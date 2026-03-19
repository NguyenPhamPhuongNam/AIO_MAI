# data_div2k_pairs.py
import os
import glob
import random
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def _pil_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _img_to_tensor_255(img: Image.Image) -> torch.Tensor:
    """
    PIL -> float32 tensor CHW, range [0..255]
    """
    arr = np.array(img, dtype=np.float32)  # HWC
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _augment(lr: torch.Tensor, hr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    lr, hr: CHW
    random H flip (W axis), V flip (H axis), transpose (H<->W)
    """
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=[2]); hr = torch.flip(hr, dims=[2])  # horizontal flip (W)
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=[1]); hr = torch.flip(hr, dims=[1])  # vertical flip (H)
    if random.random() < 0.5:
        lr = lr.transpose(1, 2); hr = hr.transpose(1, 2)              # swap H and W

    return lr.contiguous(), hr.contiguous()


def _exists_any(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _dir_has_png(d: str) -> bool:
    return len(glob.glob(os.path.join(d, "*.png"))) > 0


def _prefer_nested_if_empty(d: Optional[str]) -> Optional[str]:
    """
    If d exists but contains no *.png, try:
      - d/<basename(d)>  (e.g. .../DIV2K_valid_HR/DIV2K_valid_HR)
      - or if it has exactly one subdir and that subdir contains *.png
    Otherwise return d as-is.
    """
    if (d is None) or (not os.path.isdir(d)):
        return d
    if _dir_has_png(d):
        return d

    # case 1: nested same basename (folder inside has same name)
    inner = os.path.join(d, os.path.basename(d))
    if os.path.isdir(inner) and _dir_has_png(inner):
        return inner

    # case 2: only one subdir and it has png
    subs = [p for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p)]
    if len(subs) == 1 and _dir_has_png(subs[0]):
        return subs[0]

    return d


def _find_dir_candidates(root: str, patterns: List[str]) -> Optional[str]:
    """
    Return the first existing directory match for given patterns.
    Note: we later post-process with _prefer_nested_if_empty() and pairing-based selection for LR.
    """
    for pat in patterns:
        cand = os.path.join(root, pat)
        if os.path.isdir(cand):
            return cand
    for pat in patterns:
        if "*" in pat:
            hits = glob.glob(os.path.join(root, pat))
            hits = [h for h in hits if os.path.isdir(h)]
            if hits:
                return hits[0]
    return None


def _gather_dir_candidates(root: str, patterns: List[str]) -> List[str]:
    """
    Gather ALL existing dirs matching patterns (including wildcard patterns).
    """
    out: List[str] = []
    for pat in patterns:
        cand = os.path.join(root, pat)
        if os.path.isdir(cand):
            out.append(cand)
    for pat in patterns:
        if "*" in pat:
            hits = glob.glob(os.path.join(root, pat))
            hits = [h for h in hits if os.path.isdir(h)]
            out.extend(hits)

    # unique keep order
    seen = set()
    uniq = []
    for p in out:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def _list_hr_stems(hr_dir: str, max_n: int = 32) -> List[str]:
    hr_dir = _prefer_nested_if_empty(hr_dir) or hr_dir
    paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths[:max_n]]
    return stems


def _lr_match_score(lr_dir: str, stems: List[str], scale: int) -> int:
    """
    Score by how many stems have corresponding LR file in this lr_dir.
    Accept both:
      - {stem}x{scale}.png
      - {stem}.png
    """
    if lr_dir is None or not os.path.isdir(lr_dir):
        return -1

    lr_dir = _prefer_nested_if_empty(lr_dir) or lr_dir
    if not os.path.isdir(lr_dir):
        return -1

    score = 0
    for s in stems:
        p1 = os.path.join(lr_dir, f"{s}x{scale}.png")
        p2 = os.path.join(lr_dir, f"{s}.png")
        if os.path.exists(p1) or os.path.exists(p2):
            score += 1
    return score


def _pick_best_lr_dir(root: str, patterns: List[str], hr_dir: str, scale: int) -> Optional[str]:
    """
    From multiple candidate LR dirs, pick the one that best matches HR stems.
    This fixes many nested/extracted zip layouts automatically.
    """
    cands = _gather_dir_candidates(root, patterns)
    if not cands:
        return None

    stems = _list_hr_stems(hr_dir, max_n=32)
    if not stems:
        # fallback: just first existing
        return _prefer_nested_if_empty(cands[0]) or cands[0]

    best = None
    best_score = -1
    for d in cands:
        dd = _prefer_nested_if_empty(d) or d
        sc = _lr_match_score(dd, stems, scale)
        if sc > best_score:
            best_score = sc
            best = dd

    return best


class DIV2KPairX3(Dataset):
    """
    Paired DIV2K loader (x3 SR).

    Expected layout:
      HR: {hr_dir}/{id}.png          e.g. 0001.png, 0801.png
      LR: {lr_x3_dir}/{id}x3.png     e.g. 0001x3.png

    Also supports nested layouts (kept as-is):
      HR: {hr_dir}/{basename(hr_dir)}/{id}.png
      Example: .../DIV2K_valid_HR/DIV2K_valid_HR/*.png

    - Train: random crop LR patch (default 128), HR crop aligned (x3)
    - Val: return full image (no crop)

    Stage3 KD cache support:
      - id_list_txt: only load HR ids listed in this file (one stem per line)
      - return_meta: return (lr, hr, meta) where meta has stem/x/y/ps/scale

    IMPORTANT PATCH:
      If return_meta=True, augmentation is automatically disabled to keep
      teacher_cache crop coordinates consistent.
    """
    def __init__(
        self,
        hr_dir: str,
        lr_x3_dir: str,
        train: bool = True,
        scale: int = 3,
        lr_patch: int = 128,
        augment: bool = True,
        repeat: int = 1,
        id_list_txt: Optional[str] = None,
        return_meta: bool = False,
    ):
        super().__init__()

        # auto-jump into nested dirs if outer contains no png
        hr_dir = _prefer_nested_if_empty(hr_dir)
        lr_x3_dir = _prefer_nested_if_empty(lr_x3_dir)

        self.hr_dir = hr_dir
        self.lr_dir = lr_x3_dir
        self.train = bool(train)
        self.scale = int(scale)
        self.lr_patch = int(lr_patch)
        self.repeat = max(1, int(repeat))
        self.return_meta = bool(return_meta)

        # disable augment when return_meta=True (KD cache must align)
        if self.return_meta and augment:
            augment = False
        self.augment = bool(augment) and self.train

        self.hr_paths = sorted(glob.glob(os.path.join(self.hr_dir, "*.png")))
        if not self.hr_paths:
            raise FileNotFoundError(f"No HR png found in: {self.hr_dir}")

        # Filter by stem list (optional)
        if id_list_txt is not None:
            if not os.path.isfile(id_list_txt):
                raise FileNotFoundError(f"id_list_txt not found: {id_list_txt}")
            with open(id_list_txt, "r", encoding="utf-8") as f:
                ids = [line.strip() for line in f if line.strip()]
            allowed = set(ids)
            self.hr_paths = [
                p for p in self.hr_paths
                if os.path.splitext(os.path.basename(p))[0] in allowed
            ]
            if not self.hr_paths:
                raise FileNotFoundError("No HR png found after filtering by id_list_txt.")

        # sanity check a few pairs
        for p in self.hr_paths[:10]:
            stem = os.path.splitext(os.path.basename(p))[0]
            lr_p = self._resolve_lr_path(stem)
            if lr_p is None:
                raise FileNotFoundError(
                    f"Missing LR pair for {p} -> expected {stem}x{self.scale}.png (or fallback {stem}.png)"
                )

    def _resolve_lr_path(self, stem: str) -> Optional[str]:
        """
        Prefer DIV2K standard: {stem}x{scale}.png
        Fallback: {stem}.png
        Also: if lr_dir itself is not the final leaf, try lr_dir/X{scale} and lr_dir/<basename>/X{scale}
        """
        cand = []

        # base (expected leaf dir)
        cand.append(os.path.join(self.lr_dir, f"{stem}x{self.scale}.png"))
        cand.append(os.path.join(self.lr_dir, f"{stem}.png"))

        # common nested variants
        xdir = os.path.join(self.lr_dir, f"X{self.scale}")
        cand.append(os.path.join(xdir, f"{stem}x{self.scale}.png"))
        cand.append(os.path.join(xdir, f"{stem}.png"))

        inner = os.path.join(self.lr_dir, os.path.basename(self.lr_dir))
        inner_x = os.path.join(inner, f"X{self.scale}")
        cand.append(os.path.join(inner, f"{stem}x{self.scale}.png"))
        cand.append(os.path.join(inner, f"{stem}.png"))
        cand.append(os.path.join(inner_x, f"{stem}x{self.scale}.png"))
        cand.append(os.path.join(inner_x, f"{stem}.png"))

        return _exists_any(cand)

    def __len__(self) -> int:
        return len(self.hr_paths) * self.repeat

    def __getitem__(self, idx: int):
        hr_path = self.hr_paths[idx % len(self.hr_paths)]
        stem = os.path.splitext(os.path.basename(hr_path))[0]

        lr_path = self._resolve_lr_path(stem)
        if lr_path is None:
            raise FileNotFoundError(f"LR pair not found for stem={stem} in {self.lr_dir}")

        hr = _img_to_tensor_255(_pil_rgb(hr_path))
        lr = _img_to_tensor_255(_pil_rgb(lr_path))

        # Align HR size to LR*scale (handle slight mismatch only).
        _, h_lr, w_lr = lr.shape
        _, h_hr_raw, w_hr_raw = hr.shape
        exp_h_hr, exp_w_hr = h_lr * self.scale, w_lr * self.scale

        if h_hr_raw < exp_h_hr or w_hr_raw < exp_w_hr:
            raise RuntimeError(
                f"HR smaller than expected for stem={stem}: "
                f"got HR=({h_hr_raw},{w_hr_raw}) expected at least ({exp_h_hr},{exp_w_hr})"
            )

        # Allow tiny mismatch from some exported datasets, but catch clearly wrong pairs.
        if (h_hr_raw - exp_h_hr) >= self.scale or (w_hr_raw - exp_w_hr) >= self.scale:
            raise RuntimeError(
                f"HR/LR pair mismatch too large for stem={stem}: "
                f"LR=({h_lr},{w_lr}) scale={self.scale} -> expected HR=({exp_h_hr},{exp_w_hr}), "
                f"got HR=({h_hr_raw},{w_hr_raw})"
            )

        hr = hr[:, :exp_h_hr, :exp_w_hr]

        # Validation: return full image
        if not self.train:
            return lr, hr

        # Training: crop aligned patches
        ps = self.lr_patch
        if h_lr < ps or w_lr < ps:
            pad_h = max(0, ps - h_lr)
            pad_w = max(0, ps - w_lr)
            lr = F.pad(lr, (0, pad_w, 0, pad_h), mode="reflect")
            hr = F.pad(hr, (0, pad_w * self.scale, 0, pad_h * self.scale), mode="reflect")
            _, h_lr, w_lr = lr.shape

        x = random.randint(0, w_lr - ps)
        y = random.randint(0, h_lr - ps)

        lr_crop = lr[:, y:y + ps, x:x + ps]
        hr_crop = hr[:, y * self.scale:(y + ps) * self.scale,
                    x * self.scale:(x + ps) * self.scale]

        if self.augment:
            lr_crop, hr_crop = _augment(lr_crop, hr_crop)
        else:
            lr_crop = lr_crop.contiguous()
            hr_crop = hr_crop.contiguous()

        if self.return_meta:
            meta: Dict[str, Any] = {
                "stem": str(stem),
                "x": int(x),
                "y": int(y),
                "ps": int(ps),
                "scale": int(self.scale),
            }
            return lr_crop, hr_crop, meta

        return lr_crop, hr_crop


def resolve_div2k_paths(data_root: str, scale: int = 3):
    """
    Robust resolver for MANY DIV2K zip-extracted layouts.

    HR candidates include nested:
      - DIV2K_valid_HR
      - DIV2K_valid_HR/DIV2K_valid_HR
    LR candidates include many nested styles, and we pick the best one by checking pairing with HR stems.
    """
    # HR: prefer nested-first (so you can keep .../DIV2K_valid_HR/DIV2K_valid_HR)
    train_hr = _find_dir_candidates(data_root, ["DIV2K_train_HR/DIV2K_train_HR", "DIV2K_train_HR"])
    valid_hr = _find_dir_candidates(data_root, ["DIV2K_valid_HR/DIV2K_valid_HR", "DIV2K_valid_HR"])
    train_hr = _prefer_nested_if_empty(train_hr)
    valid_hr = _prefer_nested_if_empty(valid_hr)

    if not train_hr or not valid_hr:
        raise FileNotFoundError(
            "Cannot auto-resolve DIV2K HR paths. Found:\n"
            f" train_hr={train_hr}\n valid_hr={valid_hr}\n"
        )

    # LR: gather lots of possible dirs, then choose the one that matches HR stems best
    train_lr_patterns = [
        f"DIV2K_train_LR_bicubic_X{scale}",
        f"DIV2K_train_LR_bicubic/X{scale}",
        f"DIV2K_train_LR_bicubic/X{scale}/X{scale}",
        f"DIV2K_train_LR_bicubic/DIV2K_train_LR_bicubic/X{scale}",
        f"DIV2K_train_LR_bicubic/DIV2K_train_LR_bicubic/X{scale}/X{scale}",
        f"DIV2K_train_LR_bicubic/*/X{scale}",          # wildcard zip roots
        f"DIV2K_train_LR_bicubic*/*/X{scale}",         # even messier
    ]
    valid_lr_patterns = [
        f"DIV2K_valid_LR_bicubic_X{scale}",
        f"DIV2K_valid_LR_bicubic/X{scale}",
        f"DIV2K_valid_LR_bicubic/X{scale}/X{scale}",
        f"DIV2K_valid_LR_bicubic/DIV2K_valid_LR_bicubic/X{scale}",
        f"DIV2K_valid_LR_bicubic/DIV2K_valid_LR_bicubic/X{scale}/X{scale}",
        f"DIV2K_valid_LR_bicubic/*/X{scale}",
        f"DIV2K_valid_LR_bicubic*/*/X{scale}",
    ]

    train_lr = _pick_best_lr_dir(data_root, train_lr_patterns, train_hr, scale)
    valid_lr = _pick_best_lr_dir(data_root, valid_lr_patterns, valid_hr, scale)

    train_lr = _prefer_nested_if_empty(train_lr)
    valid_lr = _prefer_nested_if_empty(valid_lr)

    if not train_lr or not valid_lr:
        raise FileNotFoundError(
            "Cannot auto-resolve DIV2K LR paths. Found:\n"
            f" train_lr={train_lr}\n valid_lr={valid_lr}\n"
        )

    # final sanity: ensure folders contain png
    if not _dir_has_png(train_lr):
        raise FileNotFoundError(f"Resolved train_lr has no png: {train_lr}")
    if not _dir_has_png(valid_lr):
        raise FileNotFoundError(f"Resolved valid_lr has no png: {valid_lr}")

    return train_hr, train_lr, valid_hr, valid_lr