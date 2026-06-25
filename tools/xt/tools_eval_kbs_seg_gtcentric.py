from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
cv2.setNumThreads(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GT-centric trajectory evaluation for segmentation masks on KBS.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pred-root", type=Path, required=True, help="Directory containing pred_masks/*.png")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+")
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "runs" / "trajectory_eval")
    parser.add_argument("--gt-component-min-pixels", type=int, default=1)
    parser.add_argument("--cluster-max-link-distance", type=float, default=96.0)
    parser.add_argument("--cluster-max-frame-gap", type=int, default=2)
    parser.add_argument("--gt-min-track-len", type=int, default=1)
    parser.add_argument("--gt-min-total-pixels", type=int, default=4)
    parser.add_argument("--frame-hit-thr", type=float, default=0.1)
    parser.add_argument("--det-tc", type=float, default=0.5)
    parser.add_argument("--complete-tc", type=float, default=0.8)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_sequence_ids(test_json: Path) -> list[str]:
    ids = json.loads(test_json.read_text(encoding="utf-8"))
    return [f"{int(x):03d}" for x in ids]


def collect_frame_ids(dataset_root: Path, seq_id: str) -> list[int]:
    return [int(path.stem) for path in sorted((dataset_root / f"{seq_id}_img").glob("*.png"))]


def load_gt_mask(dataset_root: Path, seq_id: str, frame_idx: int) -> np.ndarray:
    mask = cv2.imread(str(dataset_root / f"{seq_id}_gt" / f"{frame_idx:02d}.png"), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(seq_id, frame_idx)
    return (mask > 0).astype(np.uint8)


def load_pred_mask(pred_root: Path, seq_id: str, frame_idx: int) -> np.ndarray:
    path = pred_root / f"{seq_id}_{frame_idx:02d}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)


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


def cluster_gt_tracks(
    frame_components: list[list[dict]],
    shape: tuple[int, int],
    max_link_distance: float,
    max_frame_gap: int,
    min_track_len: int,
    min_total_pixels: int,
) -> list[dict]:
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

    valid_tracks: list[dict] = []
    for track in tracks:
        if len(track["components"]) < min_track_len:
            continue
        total_pixels = sum(int(comp["area"]) for comp in track["components"])
        if total_pixels < min_total_pixels:
            continue
        frame_masks = {int(comp["frame_id"]): comp["mask"] for comp in track["components"]}
        valid_tracks.append(
            {
                "track_id": int(track["track_id"]),
                "frames": [int(v) for v in track["frame_ids"]],
                "frame_masks": frame_masks,
                "total_pixels": total_pixels,
                "length": len(track["components"]),
            }
        )
    return valid_tracks


def compute_false_components(pred_mask: np.ndarray, gt_mask: np.ndarray) -> int:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_mask.astype(np.uint8), connectivity=8)
    count = 0
    for label_idx in range(1, num_labels):
        comp = labels == label_idx
        if not np.any(comp):
            continue
        if np.any(gt_mask[comp] > 0):
            continue
        count += 1
    return count


def main() -> None:
    args = parse_args()
    seq_ids = load_sequence_ids(args.test_json)
    pred_root = args.pred_root / "pred_masks" if (args.pred_root / "pred_masks").is_dir() else args.pred_root
    run_name = args.run_name.strip() or f"{args.model}_gtcentric"
    out_root = args.save_dir / run_name
    ensure_dir(out_root)

    total_gt_tracks = 0
    detected = 0
    complete = 0
    total_tc_hard = 0.0
    total_tc_soft = 0.0
    endpoint_hits = 0
    total_false_components = 0
    total_frames = 0
    per_sequence: dict[str, dict] = {}

    for seq_id in seq_ids:
        frame_ids = collect_frame_ids(args.dataset_root, seq_id)
        gt_masks = [load_gt_mask(args.dataset_root, seq_id, frame_id) for frame_id in frame_ids]
        pred_masks = [load_pred_mask(pred_root, seq_id, frame_id) for frame_id in frame_ids]
        total_frames += len(frame_ids)

        gt_components = [
            extract_frame_components(mask, frame_id, min_pixels=args.gt_component_min_pixels)
            for mask, frame_id in zip(gt_masks, frame_ids)
        ]
        gt_tracks = cluster_gt_tracks(
            gt_components,
            shape=gt_masks[0].shape,
            max_link_distance=args.cluster_max_link_distance,
            max_frame_gap=args.cluster_max_frame_gap,
            min_track_len=args.gt_min_track_len,
            min_total_pixels=args.gt_min_total_pixels,
        )

        frame_to_gt = {frame_id: mask for frame_id, mask in zip(frame_ids, gt_masks)}
        frame_to_pred = {frame_id: mask for frame_id, mask in zip(frame_ids, pred_masks)}
        seq_rows = []
        seq_detected = 0
        seq_complete = 0
        seq_tc_hard = 0.0
        seq_tc_soft = 0.0
        seq_ehr = 0
        seq_false_components = 0

        for frame_id in frame_ids:
            seq_false_components += compute_false_components(frame_to_pred[frame_id], frame_to_gt[frame_id])
        total_false_components += seq_false_components

        for gt_track in gt_tracks:
            covs = []
            hits = []
            for frame_id in gt_track["frames"]:
                gt_mask = gt_track["frame_masks"][frame_id].astype(bool)
                pred_mask = frame_to_pred[frame_id].astype(bool)
                gt_area = int(gt_mask.sum())
                cov = 0.0 if gt_area <= 0 else float(np.logical_and(gt_mask, pred_mask).sum() / gt_area)
                covs.append(cov)
                hits.append(1 if cov >= args.frame_hit_thr else 0)
            tc_hard = float(sum(hits) / max(1, len(hits)))
            tc_soft = float(sum(covs) / max(1, len(covs)))
            ehr = bool(hits[0] == 1 and hits[-1] == 1) if hits else False

            total_gt_tracks += 1
            total_tc_hard += tc_hard
            total_tc_soft += tc_soft
            seq_tc_hard += tc_hard
            seq_tc_soft += tc_soft
            if tc_hard >= args.det_tc:
                detected += 1
                seq_detected += 1
            if tc_hard >= args.complete_tc:
                complete += 1
                seq_complete += 1
            if ehr:
                endpoint_hits += 1
                seq_ehr += 1

            seq_rows.append(
                {
                    "gt_track_id": gt_track["track_id"],
                    "frames": gt_track["frames"],
                    "length": gt_track["length"],
                    "tc_hard": tc_hard,
                    "tc_soft": tc_soft,
                    "detected": bool(tc_hard >= args.det_tc),
                    "complete": bool(tc_hard >= args.complete_tc),
                    "endpoint_hit": bool(ehr),
                }
            )

        n_gt = len(gt_tracks)
        per_sequence[seq_id] = {
            "num_frames": len(frame_ids),
            "num_gt_tracks": n_gt,
            "tdr": float(seq_detected / n_gt) if n_gt else 0.0,
            "ctr": float(seq_complete / n_gt) if n_gt else 0.0,
            "mean_tc_hard": float(seq_tc_hard / n_gt) if n_gt else 0.0,
            "mean_tc_soft": float(seq_tc_soft / n_gt) if n_gt else 0.0,
            "ehr": float(seq_ehr / n_gt) if n_gt else 0.0,
            "false_components": int(seq_false_components),
            "false_components_per_frame": float(seq_false_components / max(1, len(frame_ids))),
            "gt_rows": seq_rows,
        }
        print(
            f"[{args.model}] seq {seq_id}: gt={n_gt} "
            f"tdr={per_sequence[seq_id]['tdr']:.4f} "
            f"mean_tc_soft={per_sequence[seq_id]['mean_tc_soft']:.4f} "
            f"fcpf={per_sequence[seq_id]['false_components_per_frame']:.4f}",
            flush=True,
        )

    summary = {
        "model": args.model,
        "pred_root": str(args.pred_root),
        "test_json": str(args.test_json),
        "frame_hit_thr": float(args.frame_hit_thr),
        "det_tc": float(args.det_tc),
        "complete_tc": float(args.complete_tc),
        "metrics": {
            "num_sequences": len(seq_ids),
            "num_frames": total_frames,
            "num_gt_tracks": total_gt_tracks,
            "tdr": float(detected / max(1, total_gt_tracks)),
            "ctr": float(complete / max(1, total_gt_tracks)),
            "mean_tc_hard": float(total_tc_hard / max(1, total_gt_tracks)),
            "mean_tc_soft": float(total_tc_soft / max(1, total_gt_tracks)),
            "ehr": float(endpoint_hits / max(1, total_gt_tracks)),
            "false_components_total": int(total_false_components),
            "false_components_per_sequence": float(total_false_components / max(1, len(seq_ids))),
            "false_components_per_frame": float(total_false_components / max(1, total_frames)),
        },
        "per_sequence": per_sequence,
    }
    out_json = out_root / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
