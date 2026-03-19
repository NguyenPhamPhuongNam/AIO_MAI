import argparse, torch
from torch.utils.data import DataLoader

from data_div2k_pairs import DIV2KPairX3, resolve_div2k_paths
from train_antsr_fp32 import validate_metrics_all, load_teacher_mambair

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--mambair_repo", required=True)
    ap.add_argument("--mambair_teacher_ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val_max", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")

    train_hr, train_lr, valid_hr, valid_lr = resolve_div2k_paths(args.data_root, scale=3)
    val_ds = DIV2KPairX3(valid_hr, valid_lr, train=False, lr_patch=128, augment=False, repeat=1)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=(device.type=="cuda"))

    teacher = load_teacher_mambair(args.mambair_repo, args.mambair_teacher_ckpt, device).eval()

    # IMPORTANT: teacher expects LR in [0..1], outputs in [0..1]
    # We wrap it so validate_metrics_all sees SR in [0..255]
    class WrapTeacher(torch.nn.Module):
        def __init__(self, t):
            super().__init__()
            self.t = t
        def forward(self, lr_255):
            sr01 = self.t(lr_255 / 255.0)
            return (sr01 * 255.0)

    wrapped = WrapTeacher(teacher).to(device)

    m = validate_metrics_all(
        wrapped, val_loader, device,
        scale=3,
        max_images=args.val_max,
        shaves=(0,3),
        report_ssim=False
    )
    print("Teacher MambaIR metrics (DIV2K val):")
    for k in sorted(m.keys()):
        print(k, m[k])

if __name__ == "__main__":
    main()