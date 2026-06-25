from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = REPO_ROOT / "compares" / "MSAMNet-master"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

from model.CSAUNet import CSAUNet  # noqa: E402
from model.DNANet import DNANet, Res_CBAM_block  # noqa: E402
from model.DnTNet import DnTNet  # noqa: E402
from model.MSAMNet import AAFE, CBAM, Connection, CoordAtt, MSAMNet, SE  # noqa: E402
from model.utils import load_param  # noqa: E402
from tools.xt.tools_eval_kbs_dnt_seg import KBSTestDataset, collate_fn, compute_metrics, percentile_rescale  # noqa: E402


FUSION_BLOCKS = {
    "CBAM": CBAM,
    "AAFE": AAFE,
    "CA": CoordAtt,
    "SE": SE,
    "None": Connection,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sequence-projected KBS visualizations for compare segmentation models.")
    parser.add_argument("--model", choices=["CSAUNet", "DNANet", "DnTNet", "MSAMNet"], required=True)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+")
    parser.add_argument("--test-json", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "test.json")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--t-frame", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=0, help="0 means use native resolution.")
    parser.add_argument("--rescale-percentile", type=float, nargs=2, default=(1.0, 99.0))
    parser.add_argument("--channel-size", type=str, default="three")
    parser.add_argument("--backbone", type=str, default="resnet_18")
    parser.add_argument("--deep-supervision", type=str, default="False")
    parser.add_argument("--fusionblock", type=str, default="AAFE")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "runs" / "kbs_eval")
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.model == "CSAUNet":
        model = CSAUNet(input_channels=args.t_frame)
    elif args.model == "DNANet":
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
        model = DNANet(
            num_classes=1,
            input_channels=args.t_frame,
            block=Res_CBAM_block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
            deep_supervision=args.deep_supervision,
        )
    elif args.model == "DnTNet":
        model = DnTNet(seq_length=args.t_frame)
    else:
        model = MSAMNet(frame_length=args.t_frame, fusionBlock=FUSION_BLOCKS[args.fusionblock])

    checkpoint = torch.load(args.weights.expanduser().resolve(), map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Unsupported checkpoint format: {args.weights}")
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


def load_raw_rescaled_frame(
    dataset_root: Path,
    seq_id: str,
    frame_idx: int,
    input_size: int,
    rescale_percentile: tuple[float, float],
) -> np.ndarray:
    path = dataset_root / f"{seq_id}_img" / f"{int(frame_idx):02d}.png"
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if input_size > 0 and img.shape[:2] != (input_size, input_size):
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_AREA)
    img = percentile_rescale(img, rescale_percentile[0], rescale_percentile[1])
    return img.astype(np.uint8)


def make_overlay(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    out[tp] = (0, 255, 0)
    out[fp] = (0, 0, 255)
    out[fn] = (255, 255, 0)
    return out


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
    model = build_model(args, device)
    run_name = args.run_name.strip() or f"kbs_{args.model.lower()}_test_seqproj"
    save_root = args.save_dir / run_name
    save_root.mkdir(parents=True, exist_ok=True)
    pred_dir = save_root / "pred_union"
    gt_dir = save_root / "gt_union"
    raw_dir = save_root / "raw_projection"
    overlay_dir = save_root / "overlay_projection"
    for path in [pred_dir, gt_dir, raw_dir, overlay_dir]:
        path.mkdir(parents=True, exist_ok=True)

    seq_state: dict[str, dict[str, np.ndarray]] = {}

    with torch.no_grad():
        processed = 0
        total = len(dataset)
        for x, y, seq_ids, frame_idxs in loader:
            x = x.to(device)
            logits = get_logits(model(x))
            pred = logits > 0
            gt = y > 0.5

            pred_np = pred.detach().cpu().numpy().astype(np.bool_)
            gt_np = gt.detach().cpu().numpy().astype(np.bool_)
            for i, seq_id in enumerate(seq_ids):
                frame_idx = int(frame_idxs[i])
                p = pred_np[i, 0].astype(np.uint8)
                g = gt_np[i, 0].astype(np.uint8)
                gray = load_raw_rescaled_frame(
                    args.dataset_root,
                    seq_id,
                    frame_idx,
                    args.input_size,
                    tuple(args.rescale_percentile),
                )
                if seq_id not in seq_state:
                    seq_state[seq_id] = {
                        "pred": p.copy(),
                        "gt": g.copy(),
                        "raw": gray.copy(),
                    }
                else:
                    seq_state[seq_id]["pred"] = np.maximum(seq_state[seq_id]["pred"], p)
                    seq_state[seq_id]["gt"] = np.maximum(seq_state[seq_id]["gt"], g)
                    seq_state[seq_id]["raw"] = np.maximum(seq_state[seq_id]["raw"], gray)
                processed += 1
                if processed % 50 == 0 or processed == total:
                    print(f"[{args.model}] processed {processed}/{total} frames", flush=True)

    per_sequence = {}
    tp = fp = fn = tn = 0
    for seq_id, state in seq_state.items():
        pred_u8 = state["pred"].astype(np.uint8)
        gt_u8 = state["gt"].astype(np.uint8)
        raw_u8 = state["raw"].astype(np.uint8)
        overlay = make_overlay(raw_u8, gt_u8, pred_u8)

        tp_i = int(np.logical_and(pred_u8 == 1, gt_u8 == 1).sum())
        fp_i = int(np.logical_and(pred_u8 == 1, gt_u8 == 0).sum())
        fn_i = int(np.logical_and(pred_u8 == 0, gt_u8 == 1).sum())
        tn_i = int(np.logical_and(pred_u8 == 0, gt_u8 == 0).sum())
        tp += tp_i
        fp += fp_i
        fn += fn_i
        tn += tn_i

        cv2.imwrite(str(pred_dir / f"{seq_id}.png"), pred_u8 * 255)
        cv2.imwrite(str(gt_dir / f"{seq_id}.png"), gt_u8 * 255)
        cv2.imwrite(str(raw_dir / f"{seq_id}.png"), raw_u8)
        cv2.imwrite(str(overlay_dir / f"{seq_id}.png"), overlay)

        per_sequence[seq_id] = {
            "metrics": compute_metrics(tp_i, fp_i, fn_i, tn_i),
            "pred_union_path": str(pred_dir / f"{seq_id}.png"),
            "gt_union_path": str(gt_dir / f"{seq_id}.png"),
            "raw_projection_path": str(raw_dir / f"{seq_id}.png"),
            "overlay_projection_path": str(overlay_dir / f"{seq_id}.png"),
        }

    summary = {
        "model": args.model,
        "dataset_root": str(args.dataset_root),
        "test_json": str(args.test_json),
        "weights": str(args.weights.expanduser().resolve()),
        "device": str(device),
        "num_sequences": len(seq_state),
        "num_frames": len(dataset),
        "input_size": args.input_size if args.input_size > 0 else "native",
        "t_frame": args.t_frame,
        "rescale_percentile": list(args.rescale_percentile),
        "projection_mode": "sequence_union_on_max_percentile_background",
        "metrics": compute_metrics(tp, fp, fn, tn),
        "per_sequence": per_sequence,
    }
    out_json = save_root / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
