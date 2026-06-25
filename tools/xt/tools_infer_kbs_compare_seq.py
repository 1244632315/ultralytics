from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = REPO_ROOT / "compares" / "MSAMNet-master"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

from model.CSAUNet import CSAUNet  # noqa: E402
from model.DNANet import DNANet, Res_CBAM_block  # noqa: E402
from model.MSAMNet import AAFE, CBAM, Connection, CoordAtt, MSAMNet, SE  # noqa: E402
from model.utils import load_param  # noqa: E402
from tools.xt.tools_eval_kbs_dnt_seg import normalize_frame, percentile_rescale  # noqa: E402


DEFAULT_WEIGHTS = {
    "CSAUNet": REPO_ROOT
    / "compares"
    / "MSAMNet-master"
    / "result"
    / "MSOD_RAWBG_CSAUNet_17_04_2026_11_18_01_wDS"
    / "mIoU_best_CSAUNet_MSOD_RAWBG_epoch.pth.tar",
    "DNANet": REPO_ROOT
    / "compares"
    / "MSAMNet-master"
    / "result"
    / "MSOD_RAWBG_DNANet_17_04_2026_14_32_33_wDS"
    / "mIoU_best_DNANet_MSOD_RAWBG_epoch.pth.tar",
    "MSAMNet": REPO_ROOT
    / "compares"
    / "MSAMNet-master"
    / "result"
    / "MSOD_RAWBG_MSAMNet_18_04_2026_08_35_33_wDS"
    / "mIoU_best_MSAMNet_MSOD_RAWBG_epoch.pth.tar",
}

FUSION_BLOCKS = {
    "CBAM": CBAM,
    "AAFE": AAFE,
    "CA": CoordAtt,
    "SE": SE,
    "None": Connection,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compare-model inference on one KBS real sequence and save masks.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+")
    parser.add_argument("--sequence-id", type=str, default="006")
    parser.add_argument("--models", nargs="+", choices=["CSAUNet", "DNANet", "MSAMNet"], default=["CSAUNet", "DNANet", "MSAMNet"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--t-frame", type=int, default=3)
    parser.add_argument("--rescale-percentile", type=float, nargs=2, default=(1.0, 99.0))
    parser.add_argument("--channel-size", type=str, default="three")
    parser.add_argument("--backbone", type=str, default="resnet_18")
    parser.add_argument("--deep-supervision", type=str, default="False")
    parser.add_argument("--fusionblock", type=str, default="AAFE")
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "runs" / "kbs_eval")
    return parser.parse_args()


def load_frame(dataset_root: Path, seq_id: str, frame_idx: int, rescale_percentile: tuple[float, float]) -> np.ndarray:
    path = dataset_root / f"{seq_id}_img" / f"{frame_idx:02d}.png"
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    img = percentile_rescale(img, rescale_percentile[0], rescale_percentile[1])
    return img.astype(np.uint8)


def load_mask(dataset_root: Path, seq_id: str, frame_idx: int) -> np.ndarray:
    path = dataset_root / f"{seq_id}_gt" / f"{frame_idx:02d}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)


def load_input_tensor(dataset_root: Path, seq_id: str, frame_idx: int, t_frame: int, rescale_percentile: tuple[float, float]) -> torch.Tensor:
    frames = []
    for offset in range(t_frame):
        hist_idx = max(frame_idx - offset, 0)
        raw = load_frame(dataset_root, seq_id, hist_idx, rescale_percentile)
        frames.append(normalize_frame(raw))
    arr = np.stack(frames[::-1], axis=0).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def build_model(name: str, weights: Path, device: torch.device, args: argparse.Namespace) -> torch.nn.Module:
    if name == "CSAUNet":
        model = CSAUNet(input_channels=args.t_frame)
    elif name == "DNANet":
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
        model = DNANet(
            num_classes=1,
            input_channels=args.t_frame,
            block=Res_CBAM_block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
            deep_supervision=args.deep_supervision,
        )
    elif name == "MSAMNet":
        model = MSAMNet(frame_length=args.t_frame, fusionBlock=FUSION_BLOCKS[args.fusionblock])
    else:
        raise ValueError(name)

    checkpoint = torch.load(weights.expanduser().resolve(), map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Unsupported checkpoint format: {weights}")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_logits(output: torch.Tensor | list | tuple) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        for item in reversed(output):
            if isinstance(item, torch.Tensor):
                return item
        raise TypeError("model output does not contain a tensor")
    return output


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


def make_overlay(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    out[tp] = (0, 255, 0)
    out[fp] = (0, 0, 255)
    out[fn] = (255, 255, 0)
    return out


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def main() -> None:
    args = parse_args()
    seq_id = f"{int(args.sequence_id):03d}"
    want_cpu = args.device.lower().startswith("cpu")
    device = torch.device("cpu" if want_cpu or not torch.cuda.is_available() else args.device)
    frames = sorted(int(p.stem) for p in (args.dataset_root / f"{seq_id}_img").glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames found for sequence {seq_id}")

    summary = {}
    for model_name in args.models:
        weights = DEFAULT_WEIGHTS[model_name]
        save_root = args.save_dir / f"kbs_{model_name.lower()}_seq{seq_id}"
        pred_dir = save_root / "pred_masks"
        overlay_dir = save_root / "overlays"
        gt_dir = save_root / "gt_masks"
        raw_dir = save_root / "raw_rescaled"
        for path in [pred_dir, overlay_dir, gt_dir, raw_dir]:
            path.mkdir(parents=True, exist_ok=True)

        model = build_model(model_name, weights, device, args)

        tp = fp = fn = tn = 0
        frame_metrics = []
        with torch.no_grad():
            for frame_idx in frames:
                gray = load_frame(args.dataset_root, seq_id, frame_idx, tuple(args.rescale_percentile))
                gt = load_mask(args.dataset_root, seq_id, frame_idx)
                x = load_input_tensor(args.dataset_root, seq_id, frame_idx, args.t_frame, tuple(args.rescale_percentile)).to(device)
                logits = get_logits(model(x))
                pred = (logits > 0).detach().cpu().numpy()[0, 0].astype(np.uint8)

                tp_i = int(np.logical_and(pred == 1, gt == 1).sum())
                fp_i = int(np.logical_and(pred == 1, gt == 0).sum())
                fn_i = int(np.logical_and(pred == 0, gt == 1).sum())
                tn_i = int(np.logical_and(pred == 0, gt == 0).sum())
                tp += tp_i
                fp += fp_i
                fn += fn_i
                tn += tn_i

                save_mask_png(pred_dir / f"{frame_idx:02d}.png", pred)
                save_mask_png(gt_dir / f"{frame_idx:02d}.png", gt)
                cv2.imwrite(str(raw_dir / f"{frame_idx:02d}.png"), gray)
                cv2.imwrite(str(overlay_dir / f"{frame_idx:02d}.png"), make_overlay(gray, gt, pred))
                frame_metrics.append({"frame_idx": frame_idx, **compute_metrics(tp_i, fp_i, fn_i, tn_i)})

        model_summary = {
            "model": model_name,
            "sequence_id": seq_id,
            "weights": str(weights),
            "device": str(device),
            "t_frame": args.t_frame,
            "rescale_percentile": list(args.rescale_percentile),
            "num_frames": len(frames),
            "metrics": compute_metrics(tp, fp, fn, tn),
            "frame_metrics": frame_metrics,
            "save_root": str(save_root),
        }
        (save_root / "summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")
        summary[model_name] = model_summary
        print(json.dumps({"model": model_name, "sequence_id": seq_id, **model_summary["metrics"]}, indent=2))
        print(f"saved {save_root}")

    out_json = args.save_dir / f"kbs_compare_seq{seq_id}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
