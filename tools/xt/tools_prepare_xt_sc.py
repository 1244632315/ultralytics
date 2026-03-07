from pathlib import Path

import cv2
import numpy as np


ROOT = Path("dataset/obb_xt_clean")


def ensure_symlink(dst: Path, target_rel: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(target_rel)


def encode_sin_cos_from_t(t_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tau = t_u8.astype(np.float32) / 255.0
    sin_ch = ((np.sin(np.pi * tau) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    cos_ch = ((np.cos(np.pi * tau) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return sin_ch, cos_ch


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
        sin_ch, cos_ch = encode_sin_cos_from_t(t)
        xt_sc = np.stack([x, sin_ch, cos_ch], axis=-1)
        cv2.imwrite(str(dst / name), xt_sc)

    return len(names), len(list(dst.glob("*.png")))


def main() -> None:
    n_in_tr, n_out_tr = build_split("train", "train_t", "train_xt_sc")
    n_in_va, n_out_va = build_split("val", "val_t", "val_xt_sc")

    labels_dir = ROOT / "labels"
    ensure_symlink(labels_dir / "train_xt_sc", "../labels/train")
    ensure_symlink(labels_dir / "val_xt_sc", "../labels/val")

    print(
        {
            "train_common": n_in_tr,
            "train_written": n_out_tr,
            "val_common": n_in_va,
            "val_written": n_out_va,
            "out_train": str(ROOT / "images" / "train_xt_sc"),
            "out_val": str(ROOT / "images" / "val_xt_sc"),
        }
    )


if __name__ == "__main__":
    main()
