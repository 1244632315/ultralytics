from pathlib import Path

import cv2
import numpy as np


ROOT = Path("dataset/obb_xt_clean")


def robust_stretch_u8(gray_u8: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    a = gray_u8.astype(np.float32)
    lo, hi = np.percentile(a, [p_lo, p_hi])
    if hi <= lo + 1e-6:
        return np.zeros_like(gray_u8, dtype=np.uint8)
    a = np.clip(a, lo, hi)
    a = (a - lo) * (255.0 / (hi - lo))
    return a.astype(np.uint8)


def ensure_symlink(dst: Path, target_rel: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(target_rel)


def build_split(split_x: str, split_t: str, split_out: str) -> tuple[int, int]:
    src_x = ROOT / "images" / split_x
    src_t = ROOT / "images" / split_t
    dst = ROOT / "images" / split_out
    dst.mkdir(parents=True, exist_ok=True)

    x_files = {p.name: p for p in src_x.glob("*.png")}
    t_files = {p.name: p for p in src_t.glob("*.png")}
    names = sorted(set(x_files) & set(t_files))

    for name in names:
        x = cv2.imread(str(x_files[name]), cv2.IMREAD_GRAYSCALE)
        t = cv2.imread(str(t_files[name]), cv2.IMREAD_GRAYSCALE)
        if x is None or t is None:
            continue
        xn = robust_stretch_u8(x)
        tn = robust_stretch_u8(t)
        xt = np.stack([xn, tn, xn], axis=-1)
        cv2.imwrite(str(dst / name), xt)

    return len(names), len(list(dst.glob("*.png")))


def main() -> None:
    n_in_tr, n_out_tr = build_split("train", "train_t", "train_xt_norm1")
    n_in_va, n_out_va = build_split("val", "val_t", "val_xt_norm1")

    labels_dir = ROOT / "labels"
    ensure_symlink(labels_dir / "train_xt_norm1", "../labels/train")
    ensure_symlink(labels_dir / "val_xt_norm1", "../labels/val")

    print(
        {
            "train_common": n_in_tr,
            "train_written": n_out_tr,
            "val_common": n_in_va,
            "val_written": n_out_va,
            "out_train": str(ROOT / "images" / "train_xt_norm1"),
            "out_val": str(ROOT / "images" / "val_xt_norm1"),
        }
    )


if __name__ == "__main__":
    main()
