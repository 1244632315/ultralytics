from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.xt.tools_eval_kbs_dnt_seg import build_model, percentile_rescale, normalize_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DnTNet predictions on KBS real sequences.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+")
    parser.add_argument("--test-json", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "test.json")
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT
        / "compares"
        / "MSAMNet-master"
        / "result"
        / "MSOD_RAWBG_DnTNet_18_04_2026_19_38_29_wDS"
        / "mIoU_best_DnTNet_MSOD_RAWBG_epoch.pth.tar",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--t-frame", type=int, default=3)
    parser.add_argument("--rescale-percentile", type=float, nargs=2, default=(1.0, 99.0))
    parser.add_argument("--seq-ids", nargs="*", default=["042", "063", "090", "077", "059"])
    parser.add_argument("--frames-per-seq", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "kbs_eval" / "kbs_dnt_vis")
    return parser.parse_args()


def load_test_ids(test_json: Path) -> set[str]:
    ids = json.loads(test_json.read_text(encoding="utf-8"))
    return {f"{int(x):03d}" for x in ids}


def load_raw_and_input(dataset_root: Path, seq_id: str, frame_idx: int, t_frame: int, rescale_percentile: tuple[float, float]) -> tuple[np.ndarray, torch.Tensor]:
    frames = []
    raw_display = None
    for offset in range(t_frame):
        hist_idx = max(frame_idx - offset, 0)
        path = dataset_root / f"{seq_id}_img" / f"{hist_idx:02d}.png"
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        img_rescaled = percentile_rescale(img, rescale_percentile[0], rescale_percentile[1])
        if offset == 0:
            raw_display = img_rescaled.astype(np.uint8)
        frames.append(normalize_frame(img_rescaled))
    x = torch.from_numpy(np.stack(frames[::-1], axis=0)).unsqueeze(0)
    if raw_display is None:
        raise RuntimeError("failed to load display image")
    return raw_display, x


def load_gt_mask(dataset_root: Path, seq_id: str, frame_idx: int) -> np.ndarray:
    path = dataset_root / f"{seq_id}_gt" / f"{frame_idx:02d}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)


def colorize_overlay(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = base.copy()
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    out[tp] = (0, 255, 0)
    out[fp] = (0, 0, 255)
    out[fn] = (255, 255, 0)
    return out


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def make_panel(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray, seq_id: str, frame_idx: int) -> np.ndarray:
    fp = ((pred == 1) & (gt == 0)).astype(np.uint8)
    overlay = colorize_overlay(gray, gt, pred)
    tiles = [
        annotate(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), f"{seq_id} frame {frame_idx:02d} raw"),
        annotate(mask_to_bgr(gt), "GT"),
        annotate(mask_to_bgr(pred), "Pred"),
        annotate(mask_to_bgr(fp), "FP only"),
        annotate(overlay, "Overlay TP=green FP=red FN=cyan"),
    ]
    blank = np.zeros_like(tiles[0])
    tiles.append(blank)
    row1 = np.hstack(tiles[:3])
    row2 = np.hstack(tiles[3:6])
    return np.vstack([row1, row2])


def select_frame_indices(seq_len: int, count: int) -> list[int]:
    if seq_len <= 0:
        return []
    if count >= seq_len:
        return list(range(seq_len))
    return sorted({int(round(i)) for i in np.linspace(0, seq_len - 1, num=count)})


def main() -> None:
    args = parse_args()
    valid_ids = load_test_ids(args.test_json)
    seq_ids = [s for s in args.seq_ids if s in valid_ids]
    if not seq_ids:
        raise ValueError("No requested seq_ids are present in test.json")

    want_cpu = args.device.lower().startswith("cpu")
    device = torch.device("cpu" if want_cpu or not torch.cuda.is_available() else args.device)
    model = build_model(args.weights, device, args.t_frame)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    with torch.no_grad():
        for seq_id in seq_ids:
            img_dir = args.dataset_root / f"{seq_id}_img"
            frames = sorted(int(p.stem) for p in img_dir.glob("*.png"))
            picked = select_frame_indices(len(frames), args.frames_per_seq)
            picked_frame_ids = [frames[i] for i in picked]

            panels = []
            seq_tp = seq_fp = seq_fn = 0
            for frame_idx in picked_frame_ids:
                gray, x = load_raw_and_input(args.dataset_root, seq_id, frame_idx, args.t_frame, tuple(args.rescale_percentile))
                gt = load_gt_mask(args.dataset_root, seq_id, frame_idx)
                logits = model(x.to(device))
                pred = (logits > 0).detach().cpu().numpy()[0, 0].astype(np.uint8)
                panels.append(make_panel(gray, gt, pred, seq_id, frame_idx))
                seq_tp += int(np.logical_and(pred == 1, gt == 1).sum())
                seq_fp += int(np.logical_and(pred == 1, gt == 0).sum())
                seq_fn += int(np.logical_and(pred == 0, gt == 1).sum())

            mosaic = np.vstack(panels)
            out_path = args.out_dir / f"{seq_id}_panel.png"
            cv2.imwrite(str(out_path), mosaic)
            precision = seq_tp / (seq_tp + seq_fp + 1e-12)
            recall = seq_tp / (seq_tp + seq_fn + 1e-12)
            summary.append(
                {
                    "sequence_id": seq_id,
                    "frames": picked_frame_ids,
                    "precision_sampled": float(precision),
                    "recall_sampled": float(recall),
                    "panel": str(out_path),
                }
            )
            print(f"saved {out_path}")

    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
