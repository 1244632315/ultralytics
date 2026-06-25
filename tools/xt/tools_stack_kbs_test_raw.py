from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stack KBS test sequences into per-sequence 8-bit PNG projections.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+")
    parser.add_argument("--test-json", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "test.json")
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "runs" / "kbs_eval")
    parser.add_argument("--run-name", type=str, default="kbs_test_raw_stack_p05_995")
    parser.add_argument("--stack-mode", choices=["max", "sum", "mean"], default="max")
    parser.add_argument("--percentile", type=float, nargs=2, default=(0.5, 99.5))
    return parser.parse_args()


def trunc_img(img: np.ndarray, ratio: tuple[float, float]) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    vmin, vmax = np.percentile(img, ratio)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8) * 255.0
    return img.astype(np.uint8)


def load_seq_ids(test_json: Path) -> list[str]:
    ids = json.loads(test_json.read_text(encoding="utf-8"))
    return [f"{int(x):03d}" for x in ids]


def load_frames(dataset_root: Path, seq_id: str) -> list[np.ndarray]:
    frames = []
    for path in sorted((dataset_root / f"{seq_id}_img").glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(img.astype(np.float32))
    if not frames:
        raise FileNotFoundError(f"No frames found for {seq_id}")
    return frames


def stack_frames(frames: list[np.ndarray], mode: str) -> np.ndarray:
    stack = np.stack(frames, axis=0).astype(np.float32)
    if mode == "sum":
        return stack.sum(axis=0)
    if mode == "mean":
        return stack.mean(axis=0)
    return stack.max(axis=0)


def main() -> None:
    args = parse_args()
    seq_ids = load_seq_ids(args.test_json)
    out_root = args.save_dir / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    meta = {
        "dataset_root": str(args.dataset_root),
        "test_json": str(args.test_json),
        "stack_mode": str(args.stack_mode),
        "percentile": [float(args.percentile[0]), float(args.percentile[1])],
        "num_sequences": len(seq_ids),
        "outputs": {},
    }

    for seq_id in seq_ids:
        frames = load_frames(args.dataset_root, seq_id)
        stacked = stack_frames(frames, args.stack_mode)
        out = trunc_img(stacked, tuple(args.percentile))
        out_path = out_root / f"{seq_id}.png"
        cv2.imwrite(str(out_path), out)
        meta["outputs"][seq_id] = {
            "num_frames": len(frames),
            "path": str(out_path),
            "shape": [int(out.shape[0]), int(out.shape[1])],
        }
        print(f"saved {out_path}")

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
