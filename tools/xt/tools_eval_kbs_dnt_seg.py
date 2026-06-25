from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import sep
import torch
from torch.utils.data import DataLoader, Dataset

import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = REPO_ROOT / "compares" / "MSAMNet-master"
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

from model.DnTNet import DnTNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DnTNet on real KBS sequences with standard segmentation metrics.")
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=0, help="0 means use native resolution.")
    parser.add_argument("--rescale-percentile", type=float, nargs=2, default=(1.0, 99.0))
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "runs" / "kbs_eval")
    return parser.parse_args()


def percentile_rescale(img: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    vmin, vmax = np.percentile(img, (pmin, pmax))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin) * 255.0
    return img.astype(np.float32)


def normalize_frame(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    bkg = sep.Background(img)
    img = img - np.array(bkg, dtype=np.float32)
    img[img < 0] = 0
    std = float(img.std())
    if std < 1e-6:
        std = 1.0
    img = (img - float(img.mean())) / std
    return img.astype(np.float32)


@dataclass
class SampleMeta:
    seq_id: str
    frame_idx: int


class KBSTestDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        test_json: Path,
        t_frame: int,
        input_size: int = 0,
        rescale_percentile: tuple[float, float] = (1.0, 99.0),
    ):
        self.dataset_root = dataset_root
        self.t_frame = int(t_frame)
        self.input_size = int(input_size)
        self.rescale_percentile = tuple(float(x) for x in rescale_percentile)
        ids = json.loads(test_json.read_text(encoding="utf-8"))
        self.seq_ids = [f"{int(x):03d}" for x in ids]
        self.items: list[SampleMeta] = []
        for seq_id in self.seq_ids:
            img_dir = dataset_root / f"{seq_id}_img"
            frames = sorted(img_dir.glob("*.png"))
            for path in frames:
                self.items.append(SampleMeta(seq_id=seq_id, frame_idx=int(path.stem)))

    def __len__(self) -> int:
        return len(self.items)

    def _load_frame(self, seq_id: str, frame_idx: int) -> np.ndarray:
        path = self.dataset_root / f"{seq_id}_img" / f"{frame_idx:02d}.png"
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if self.input_size > 0 and img.shape[:2] != (self.input_size, self.input_size):
            img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img = percentile_rescale(img, self.rescale_percentile[0], self.rescale_percentile[1])
        return normalize_frame(img)

    def _load_mask(self, seq_id: str, frame_idx: int) -> np.ndarray:
        path = self.dataset_root / f"{seq_id}_gt" / f"{frame_idx:02d}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(path)
        if self.input_size > 0 and mask.shape[:2] != (self.input_size, self.input_size):
            mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        return mask

    def __getitem__(self, index: int):
        meta = self.items[index]
        frames = []
        for offset in range(self.t_frame):
            hist_idx = max(meta.frame_idx - offset, 0)
            frames.append(self._load_frame(meta.seq_id, hist_idx))
        x = torch.from_numpy(np.stack(frames[::-1], axis=0))
        y = torch.from_numpy(self._load_mask(meta.seq_id, meta.frame_idx)[None, ...])
        return x, y, meta.seq_id, meta.frame_idx


def collate_fn(batch):
    xs, ys, seq_ids, frame_idxs = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), list(seq_ids), list(frame_idxs)


def build_model(weights: Path, device: torch.device, t_frame: int) -> torch.nn.Module:
    model = DnTNet(seq_length=t_frame)
    checkpoint = torch.load(weights.expanduser().resolve(), map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Unsupported checkpoint format: {weights}")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    iou = tp / (tp + fp + fn + 1e-12)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-12)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "dice_f1": float(dice),
        "pixel_accuracy": float(acc),
    }


def main() -> None:
    args = parse_args()
    want_cpu = args.device.lower().startswith("cpu")
    device = torch.device("cpu" if want_cpu or not torch.cuda.is_available() else args.device)

    dataset = KBSTestDataset(
        args.dataset_root,
        args.test_json,
        args.t_frame,
        args.input_size,
        tuple(args.rescale_percentile),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    model = build_model(args.weights, device, args.t_frame)

    tp = fp = fn = tn = 0
    per_sequence: dict[str, dict[str, int]] = {seq_id: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for seq_id in dataset.seq_ids}

    with torch.no_grad():
        for x, y, seq_ids, _frame_idxs in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits > 0
            gt = y > 0.5

            pred_np = pred.detach().cpu().numpy().astype(np.bool_)
            gt_np = gt.detach().cpu().numpy().astype(np.bool_)
            for i, seq_id in enumerate(seq_ids):
                p = pred_np[i, 0]
                g = gt_np[i, 0]
                tp_i = int(np.logical_and(p, g).sum())
                fp_i = int(np.logical_and(p, np.logical_not(g)).sum())
                fn_i = int(np.logical_and(np.logical_not(p), g).sum())
                tn_i = int(np.logical_and(np.logical_not(p), np.logical_not(g)).sum())
                tp += tp_i
                fp += fp_i
                fn += fn_i
                tn += tn_i
                per_sequence[seq_id]["tp"] += tp_i
                per_sequence[seq_id]["fp"] += fp_i
                per_sequence[seq_id]["fn"] += fn_i
                per_sequence[seq_id]["tn"] += tn_i

    summary = {
        "dataset_root": str(args.dataset_root),
        "test_json": str(args.test_json),
        "weights": str(args.weights.expanduser().resolve()),
        "device": str(device),
        "num_sequences": len(dataset.seq_ids),
        "num_frames": len(dataset),
        "input_size": args.input_size if args.input_size > 0 else "native",
        "t_frame": args.t_frame,
        "rescale_percentile": list(args.rescale_percentile),
        "metrics": compute_metrics(tp, fp, fn, tn),
        "per_sequence": {seq_id: compute_metrics(**stats) for seq_id, stats in per_sequence.items()},
    }

    save_dir = args.save_dir / "kbs_dnt_seg"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_json = save_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["metrics"], indent=2))
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
