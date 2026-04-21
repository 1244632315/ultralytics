from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize x-only sequential MSOD samples.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/xt_seq_msod_x/MSOD"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/xt_seq_msod_preview"))
    return parser.parse_args()


def overlay_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mask.max() <= 0:
        return base
    overlay = base.copy()
    overlay[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(base, 0.7, overlay, 0.3, 0.0)


def build_strip(seq_dir: Path, mask_dir: Path) -> np.ndarray:
    img_files = sorted(seq_dir.glob("*.tif"))
    mask_files = {p.stem: p for p in mask_dir.glob("*.png")}
    tiles = []
    for img_path in img_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_files[img_path.stem]), cv2.IMREAD_GRAYSCALE) if img_path.stem in mask_files else None
        if img is None:
            continue
        if mask is None:
            mask = np.zeros_like(img)
        top = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(top, img_path.stem, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        bottom = overlay_mask(img, mask)
        tiles.append(np.vstack([top, bottom]))
    if not tiles:
        raise RuntimeError(f"No image tiles found in {seq_dir}")
    return np.hstack(tiles)


def main() -> None:
    args = parse_args()
    img_root = args.dataset_root / args.split / "images"
    mask_root = args.dataset_root / args.split / "masks"
    seq_dirs = sorted([p for p in img_root.iterdir() if p.is_dir()])
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence directories found in {img_root}")

    rng = random.Random(args.seed)
    chosen = seq_dirs if len(seq_dirs) <= args.count else rng.sample(seq_dirs, args.count)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for seq_dir in chosen:
        strip = build_strip(seq_dir, mask_root / seq_dir.name)
        dst = args.out_dir / f"{args.split}_{seq_dir.name}.png"
        cv2.imwrite(str(dst), strip)
        print(dst)


if __name__ == "__main__":
    main()
