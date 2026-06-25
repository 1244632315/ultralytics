from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
cv2.setNumThreads(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KBS bbox trajectory summary from saved per-frame masks.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-root", type=Path, required=True, help="Run root that contains pred_masks/gt_masks/raw_rescaled.")
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "runs" / "kbs_eval")
    parser.add_argument("--component-min-pixels", type=int, default=4)
    parser.add_argument("--cluster-max-link-distance", type=float, default=96.0)
    parser.add_argument("--cluster-max-frame-gap", type=int, default=2)
    parser.add_argument("--pred-min-track-len", type=int, default=2)
    parser.add_argument("--pred-min-total-pixels", type=int, default=12)
    parser.add_argument("--gt-min-track-len", type=int, default=1)
    parser.add_argument("--gt-min-total-pixels", type=int, default=4)
    parser.add_argument("--match-iou", type=float, default=0.5)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def build_summary(
    args: argparse.Namespace,
    total_tp: int,
    total_fp: int,
    total_fn: int,
    total_frames: int,
    per_sequence: dict[str, dict],
) -> dict:
    num_sequences = len(per_sequence)
    return {
        "model": args.model,
        "dataset_root": str(args.input_root),
        "test_json": str(args.test_json),
        "weights": str(args.input_root),
        "device": "saved_masks",
        "num_sequences": num_sequences,
        "num_frames": total_frames,
        "rescale_percentile": [],
        "pred_threshold_logit": 0.0,
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


def load_sequence_ids(test_json: Path) -> list[str]:
    ids = json.loads(test_json.read_text(encoding="utf-8"))
    return [f"{int(x):03d}" for x in ids]


def parse_frame_idx(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_mask_series(mask_root: Path, seq_id: str) -> tuple[list[int], list[np.ndarray]]:
    files = sorted(mask_root.glob(f"{seq_id}_*.png"), key=parse_frame_idx)
    if not files:
        raise FileNotFoundError(f"No masks found for sequence {seq_id} under {mask_root}")
    frame_ids = [parse_frame_idx(path) for path in files]
    masks = []
    for path in files:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(path)
        masks.append((mask > 0).astype(np.uint8))
    return frame_ids, masks


def load_raw_projection(raw_root: Path, seq_id: str, shape: tuple[int, int]) -> np.ndarray:
    if not raw_root.exists():
        return np.zeros(shape, dtype=np.uint8)
    files = sorted(raw_root.glob(f"{seq_id}_*.png"), key=parse_frame_idx)
    if not files:
        return np.zeros(shape, dtype=np.uint8)
    frames = []
    for path in files:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            continue
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.astype(np.uint8))
    if not frames:
        return np.zeros(shape, dtype=np.uint8)
    return np.max(np.stack(frames, axis=0), axis=0)


def main() -> None:
    args = parse_args()
    seq_ids = load_sequence_ids(args.test_json)
    pred_root = args.input_root / "pred_masks"
    gt_root = args.input_root / "gt_masks"
    raw_root = args.input_root / "raw_rescaled"

    run_name = args.run_name.strip() or f"{args.input_root.name}_bbox"
    save_root = args.save_dir / run_name
    ensure_dir(save_root)
    ensure_dir(save_root / "vis_boxes")
    ensure_dir(save_root / "pred_union")
    ensure_dir(save_root / "gt_union")

    total_tp = total_fp = total_fn = 0
    total_frames = 0
    per_sequence: dict[str, dict] = {}

    for seq_idx, seq_id in enumerate(seq_ids, start=1):
        pred_frame_ids, pred_frame_masks = load_mask_series(pred_root, seq_id)
        gt_frame_ids, gt_frame_masks = load_mask_series(gt_root, seq_id)
        if pred_frame_ids != gt_frame_ids:
            raise ValueError(f"Frame mismatch for seq {seq_id}: {pred_frame_ids[:5]} != {gt_frame_ids[:5]}")
        frame_ids = pred_frame_ids
        total_frames += len(frame_ids)

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

        base_stack = load_raw_projection(raw_root, seq_id, shape)
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

    summary = build_summary(
        args=args,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        total_frames=total_frames,
        per_sequence=per_sequence,
    )
    summary["source_masks_root"] = str(args.input_root)
    out_json = save_root / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
