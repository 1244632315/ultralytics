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
from model.DnTNet import DnTNet  # noqa: E402
from model.MSAMNet import AAFE, CBAM, Connection, CoordAtt, MSAMNet, SE  # noqa: E402
from model.utils import load_param  # noqa: E402
from tools.xt.tools_eval_kbs_dnt_seg import normalize_frame, percentile_rescale  # noqa: E402


FUSION_BLOCKS = {
    "CBAM": CBAM,
    "AAFE": AAFE,
    "CA": CoordAtt,
    "SE": SE,
    "None": Connection,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KBS real sequences at trajectory-box level.")
    parser.add_argument("--model", choices=["CSAUNet", "DNANet", "DnTNet", "MSAMNet"], required=True)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+")
    parser.add_argument("--test-json", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "test.json")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--t-frame", type=int, default=3)
    parser.add_argument("--input-size", type=int, default=0, help="0 means use native resolution.")
    parser.add_argument("--rescale-percentile", type=float, nargs=2, default=(1.0, 99.0))
    parser.add_argument("--pred-thr", type=float, default=0.0, help="Logit threshold. 0 means sigmoid 0.5.")
    parser.add_argument("--component-min-pixels", type=int, default=4)
    parser.add_argument("--cluster-max-link-distance", type=float, default=96.0)
    parser.add_argument("--cluster-max-frame-gap", type=int, default=2)
    parser.add_argument("--pred-min-track-len", type=int, default=2)
    parser.add_argument("--pred-min-total-pixels", type=int, default=12)
    parser.add_argument("--gt-min-track-len", type=int, default=1)
    parser.add_argument("--gt-min-total-pixels", type=int, default=4)
    parser.add_argument("--match-iou", type=float, default=0.5)
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


def load_sequence_ids(test_json: Path) -> list[str]:
    ids = json.loads(test_json.read_text(encoding="utf-8"))
    return [f"{int(x):03d}" for x in ids]


def collect_frame_ids(dataset_root: Path, seq_id: str) -> list[int]:
    img_dir = dataset_root / f"{seq_id}_img"
    return [int(path.stem) for path in sorted(img_dir.glob("*.png"))]


def load_raw_frame(dataset_root: Path, seq_id: str, frame_idx: int, input_size: int) -> np.ndarray:
    path = dataset_root / f"{seq_id}_img" / f"{frame_idx:02d}.png"
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if input_size > 0 and img.shape[:2] != (input_size, input_size):
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_AREA)
    return img


def load_gt_mask(dataset_root: Path, seq_id: str, frame_idx: int, input_size: int) -> np.ndarray:
    path = dataset_root / f"{seq_id}_gt" / f"{frame_idx:02d}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if input_size > 0 and mask.shape[:2] != (input_size, input_size):
        mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def load_model_input(
    dataset_root: Path,
    seq_id: str,
    frame_idx: int,
    t_frame: int,
    input_size: int,
    rescale_percentile: tuple[float, float],
) -> torch.Tensor:
    frames = []
    for offset in range(t_frame):
        hist_idx = max(frame_idx - offset, 0)
        raw = load_raw_frame(dataset_root, seq_id, hist_idx, input_size)
        img = percentile_rescale(raw, rescale_percentile[0], rescale_percentile[1])
        img = normalize_frame(img)
        frames.append(img)
    x = np.stack(frames[::-1], axis=0).astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0)


def extract_frame_components(mask: np.ndarray, frame_id: int, min_pixels: int) -> list[dict]:
    work = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(work, connectivity=8)
    rows: list[dict] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        ys, xs = np.where(labels == label_idx)
        comp_mask = np.zeros_like(work, dtype=np.uint8)
        comp_mask[ys, xs] = 1
        rows.append(
            {
                "frame_id": int(frame_id),
                "area": area,
                "centroid": np.asarray(centroids[label_idx], dtype=np.float32),
                "mask": comp_mask,
            }
        )
    return rows


def _trajectory_link_cost(track: dict, comp: dict) -> float:
    dt = int(comp["frame_id"]) - int(track["last_frame"])
    if dt <= 0:
        return float("inf")
    last_centroid = np.asarray(track["last_centroid"], dtype=np.float32)
    if len(track["centroids"]) >= 2:
        prev_centroid = np.asarray(track["centroids"][-2], dtype=np.float32)
        velocity = last_centroid - prev_centroid
        pred_centroid = last_centroid + velocity * float(dt)
    else:
        pred_centroid = last_centroid
    return float(np.linalg.norm(np.asarray(comp["centroid"], dtype=np.float32) - pred_centroid))


def bbox_from_binary_mask(mask: np.ndarray, min_pixels: int) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    if (x2 - x1) < 1e-6 or (y2 - y1) < 1e-6:
        return None
    return [x1, y1, x2, y2]


def cluster_components_into_track_boxes(
    frame_components: list[list[dict]],
    shape: tuple[int, int],
    max_link_distance: float,
    max_frame_gap: int,
    min_track_len: int,
    min_total_pixels: int,
    min_pixels_for_box: int,
) -> tuple[list[list[float]], list[dict], np.ndarray]:
    tracks: list[dict] = []
    next_track_id = 0

    for comps in frame_components:
        used_tracks: set[int] = set()
        comp_order = sorted(comps, key=lambda row: -row["area"])
        for comp in comp_order:
            best_idx = None
            best_cost = float("inf")
            for idx, track in enumerate(tracks):
                if idx in used_tracks:
                    continue
                gap = int(comp["frame_id"]) - int(track["last_frame"])
                if gap <= 0 or gap > max_frame_gap:
                    continue
                cost = _trajectory_link_cost(track, comp)
                if cost <= max_link_distance and cost < best_cost:
                    best_cost = cost
                    best_idx = idx
            if best_idx is None:
                tracks.append(
                    {
                        "track_id": next_track_id,
                        "last_frame": int(comp["frame_id"]),
                        "last_centroid": np.asarray(comp["centroid"], dtype=np.float32),
                        "centroids": [np.asarray(comp["centroid"], dtype=np.float32)],
                        "frame_ids": [int(comp["frame_id"])],
                        "components": [comp],
                    }
                )
                used_tracks.add(len(tracks) - 1)
                next_track_id += 1
            else:
                track = tracks[best_idx]
                track["last_frame"] = int(comp["frame_id"])
                track["last_centroid"] = np.asarray(comp["centroid"], dtype=np.float32)
                track["centroids"].append(np.asarray(comp["centroid"], dtype=np.float32))
                track["frame_ids"].append(int(comp["frame_id"]))
                track["components"].append(comp)
                used_tracks.add(best_idx)

    boxes: list[list[float]] = []
    rows: list[dict] = []
    union_mask = np.zeros(shape, dtype=np.uint8)

    for track in tracks:
        if len(track["components"]) < min_track_len:
            continue
        track_mask = np.zeros(shape, dtype=np.uint8)
        total_pixels = 0
        for comp in track["components"]:
            track_mask = np.maximum(track_mask, comp["mask"])
            total_pixels += int(comp["area"])
        if total_pixels < min_total_pixels:
            continue
        box = bbox_from_binary_mask(track_mask, min_pixels=min_pixels_for_box)
        if box is None:
            continue
        boxes.append(box)
        union_mask = np.maximum(union_mask, track_mask)
        rows.append(
            {
                "track_id": int(track["track_id"]),
                "frames": track["frame_ids"],
                "length": len(track["components"]),
                "total_pixels": total_pixels,
                "bbox_xyxy": [float(v) for v in box],
            }
        )
    return boxes, rows, union_mask


def box_iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def match_boxes(pred_boxes: list[list[float]], gt_boxes: list[list[float]], match_iou: float) -> tuple[int, int, int, list[dict]]:
    pairs = []
    for gt_idx, gt_box in enumerate(gt_boxes):
        for pred_idx, pred_box in enumerate(pred_boxes):
            iou = box_iou_xyxy(gt_box, pred_box)
            if iou >= match_iou:
                pairs.append((iou, gt_idx, pred_idx))
    pairs.sort(reverse=True, key=lambda row: row[0])
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[dict] = []
    for iou, gt_idx, pred_idx in pairs:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append({"gt_idx": gt_idx, "pred_idx": pred_idx, "iou": float(iou)})
    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matches


def draw_boxes(base_img: np.ndarray, gt_boxes: list[list[float]], pred_boxes: list[list[float]]) -> np.ndarray:
    if base_img.ndim == 2:
        canvas = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        canvas = base_img.copy()
    for box in gt_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
    for box in pred_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
    return canvas


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_summary(
    args: argparse.Namespace,
    device: torch.device,
    total_tp: int,
    total_fp: int,
    total_fn: int,
    total_frames: int,
    per_sequence: dict[str, dict],
) -> dict:
    num_sequences = len(per_sequence)
    return {
        "model": args.model,
        "dataset_root": str(args.dataset_root),
        "test_json": str(args.test_json),
        "weights": str(args.weights.expanduser().resolve()),
        "device": str(device),
        "num_sequences": num_sequences,
        "num_frames": total_frames,
        "rescale_percentile": list(args.rescale_percentile),
        "pred_threshold_logit": float(args.pred_thr),
        "matching_iou": float(args.match_iou),
        "trajectory_definition": {
            "pred_min_track_len": int(args.pred_min_track_len),
            "pred_min_total_pixels": int(args.pred_min_total_pixels),
            "gt_min_track_len": int(args.gt_min_track_len),
            "gt_min_total_pixels": int(args.gt_min_total_pixels),
            "cluster_max_link_distance": float(args.cluster_max_link_distance),
            "cluster_max_frame_gap": int(args.cluster_max_frame_gap),
            "box_type": "axis_aligned_bbox_from_clustered_track_mask",
        },
        "metrics": {
            "num_gt_tracks": int(total_tp + total_fn),
            "num_pred_tracks": int(total_tp + total_fp),
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "recall": float(total_tp / (total_tp + total_fn + 1e-12)),
            "false_alarms_per_sequence": float(total_fp / max(1, num_sequences)),
            "false_alarms_per_frame": float(total_fp / max(1, total_frames)),
            "precision": float(total_tp / (total_tp + total_fp + 1e-12)),
        },
        "per_sequence": per_sequence,
    }


def main() -> None:
    args = parse_args()
    want_cpu = args.device.lower().startswith("cpu")
    device = torch.device("cpu" if want_cpu or not torch.cuda.is_available() else args.device)
    model = build_model(args, device)
    seq_ids = load_sequence_ids(args.test_json)

    run_name = args.run_name.strip() or f"kbs_{args.model.lower()}_bbox_eval"
    save_root = args.save_dir / run_name
    ensure_dir(save_root)
    ensure_dir(save_root / "vis_boxes")
    ensure_dir(save_root / "pred_union")
    ensure_dir(save_root / "gt_union")

    total_tp = total_fp = total_fn = 0
    total_frames = 0
    per_sequence = {}

    with torch.no_grad():
        for seq_idx, seq_id in enumerate(seq_ids, start=1):
            frame_ids = collect_frame_ids(args.dataset_root, seq_id)
            if not frame_ids:
                continue
            total_frames += len(frame_ids)

            pred_frame_masks = []
            gt_frame_masks = []
            base_frames = []

            for frame_id in frame_ids:
                raw = load_raw_frame(args.dataset_root, seq_id, frame_id, args.input_size)
                base = percentile_rescale(raw, args.rescale_percentile[0], args.rescale_percentile[1]).astype(np.uint8)
                x = load_model_input(
                    args.dataset_root,
                    seq_id,
                    frame_id,
                    args.t_frame,
                    args.input_size,
                    tuple(args.rescale_percentile),
                ).to(device)
                logits = get_logits(model(x))
                pred_mask = (logits[0, 0] > args.pred_thr).detach().cpu().numpy().astype(np.uint8)
                gt_mask = load_gt_mask(args.dataset_root, seq_id, frame_id, args.input_size)

                pred_frame_masks.append(pred_mask)
                gt_frame_masks.append(gt_mask)
                base_frames.append(base)

            shape = gt_frame_masks[0].shape
            pred_components = [
                extract_frame_components(mask, frame_id, min_pixels=args.component_min_pixels)
                for mask, frame_id in zip(pred_frame_masks, frame_ids)
            ]
            gt_components = [
                extract_frame_components(mask, frame_id, min_pixels=max(1, args.component_min_pixels))
                for mask, frame_id in zip(gt_frame_masks, frame_ids)
            ]

            pred_boxes, pred_tracks, pred_union = cluster_components_into_track_boxes(
                pred_components,
                shape=shape,
                max_link_distance=args.cluster_max_link_distance,
                max_frame_gap=args.cluster_max_frame_gap,
                min_track_len=args.pred_min_track_len,
                min_total_pixels=args.pred_min_total_pixels,
                min_pixels_for_box=args.component_min_pixels,
            )
            gt_boxes, gt_tracks, gt_union = cluster_components_into_track_boxes(
                gt_components,
                shape=shape,
                max_link_distance=args.cluster_max_link_distance,
                max_frame_gap=args.cluster_max_frame_gap,
                min_track_len=args.gt_min_track_len,
                min_total_pixels=args.gt_min_total_pixels,
                min_pixels_for_box=max(1, args.component_min_pixels),
            )

            tp, fp, fn, matches = match_boxes(pred_boxes, gt_boxes, match_iou=args.match_iou)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            base_stack = np.max(np.stack(base_frames, axis=0), axis=0)
            vis = draw_boxes(base_stack, gt_boxes, pred_boxes)
            cv2.imwrite(str(save_root / "vis_boxes" / f"{seq_id}.png"), vis)
            cv2.imwrite(str(save_root / "pred_union" / f"{seq_id}.png"), (pred_union * 255).astype(np.uint8))
            cv2.imwrite(str(save_root / "gt_union" / f"{seq_id}.png"), (gt_union * 255).astype(np.uint8))

            per_sequence[seq_id] = {
                "num_frames": len(frame_ids),
                "num_gt_tracks": len(gt_boxes),
                "num_pred_tracks": len(pred_boxes),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "recall": float(tp / len(gt_boxes)) if gt_boxes else 1.0,
                "false_alarms": fp,
                "pred_boxes_xyxy": [[float(v) for v in box] for box in pred_boxes],
                "gt_boxes_xyxy": [[float(v) for v in box] for box in gt_boxes],
                "pred_tracks": pred_tracks,
                "gt_tracks": gt_tracks,
                "matches": matches,
            }
            print(
                f"[{args.model}] seq {seq_idx}/{len(seq_ids)} {seq_id}: "
                f"gt={len(gt_boxes)} pred={len(pred_boxes)} tp={tp} fp={fp} fn={fn}",
                flush=True,
            )
            partial_summary = build_summary(
                args=args,
                device=device,
                total_tp=total_tp,
                total_fp=total_fp,
                total_fn=total_fn,
                total_frames=total_frames,
                per_sequence=per_sequence,
            )
            partial_path = save_root / "summary.partial.json"
            partial_path.write_text(json.dumps(partial_summary, indent=2), encoding="utf-8")

    summary = build_summary(
        args=args,
        device=device,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        total_frames=total_frames,
        per_sequence=per_sequence,
    )

    out_json = save_root / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
