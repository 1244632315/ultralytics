from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
cv2.setNumThreads(1)


from tools.xt.tools_eval_kbs_detector_bbox import build_notebook_aligned_xt, load_sequence_frames  # noqa: E402
from tools.xt.tools_eval_kbs_detector_reconstruct_seg import connected_component_from_seed  # noqa: E402
from tools.xt.tools_eval_kbs_seg_gtcentric import (  # noqa: E402
    cluster_gt_tracks,
    collect_frame_ids,
    compute_false_components,
    extract_frame_components,
    load_gt_mask,
    load_sequence_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a traditional x_only + DBSCAN trajectory baseline on KBS.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+")
    parser.add_argument(
        "--test-json",
        type=Path,
        default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "test_expanded_excluding_ft10.json",
    )
    parser.add_argument("--seq-ids", nargs="*", default=None)
    parser.add_argument("--x-percentile", type=float, nargs=2, default=(0.01, 99.99))
    parser.add_argument("--star-threshold", type=float, default=1.5)
    parser.add_argument(
        "--t-mode",
        type=str,
        default="argmax_legacy",
        choices=["argmax_legacy", "argmax_fullrange", "hybrid", "hybrid_support"],
    )
    parser.add_argument("--time-bg-percentile", type=float, default=25.0)
    parser.add_argument("--time-conf-threshold", type=float, default=0.10)
    parser.add_argument("--time-hist-energy-threshold", type=float, default=0.05)
    parser.add_argument("--support-energy-threshold", type=float, default=0.20)
    parser.add_argument("--support-peak-percentile", type=float, default=99.5)
    parser.add_argument("--support-dilate", type=int, default=1)
    parser.add_argument("--x-cand-percentile", type=float, default=99.85)
    parser.add_argument("--x-cand-min", type=int, default=0)
    parser.add_argument("--dbscan-eps", type=float, default=3.5)
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
    parser.add_argument("--cluster-min-points", type=int, default=8)
    parser.add_argument("--cluster-min-span", type=float, default=8.0)
    parser.add_argument("--cluster-band-half", type=float, default=3.0)
    parser.add_argument("--cluster-dilate", type=int, default=1)
    parser.add_argument("--seed-min-percentile", type=float, default=99.5)
    parser.add_argument("--seed-min-intensity", type=float, default=60.0)
    parser.add_argument("--grow-seed-ratio", type=float, default=0.55)
    parser.add_argument("--grow-std-weight", type=float, default=0.8)
    parser.add_argument("--frame-min-pixels", type=int, default=1)
    parser.add_argument("--gt-component-min-pixels", type=int, default=1)
    parser.add_argument("--cluster-max-link-distance", type=float, default=96.0)
    parser.add_argument("--cluster-max-frame-gap", type=int, default=2)
    parser.add_argument("--gt-min-track-len", type=int, default=1)
    parser.add_argument("--gt-min-total-pixels", type=int, default=4)
    parser.add_argument("--frame-hit-thr", type=float, default=0.1)
    parser.add_argument("--det-tc", type=float, default=0.5)
    parser.add_argument("--complete-tc", type=float, default=0.8)
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "runs" / "trajectory_eval")
    parser.add_argument("--run-name", type=str, default="kbs_xonly_dbscan_gtcentric")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dbscan_points(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    n = int(points.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.int32)
    if n == 1:
        return np.array([-1], dtype=np.int32)
    d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    neighbors = [np.flatnonzero(d2[i] <= float(eps) * float(eps)) for i in range(n)]
    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = neighbors[i]
        if nbrs.size < int(min_samples):
            continue
        labels[i] = cluster_id
        seeds = list(nbrs.tolist())
        cursor = 0
        while cursor < len(seeds):
            j = seeds[cursor]
            cursor += 1
            if not visited[j]:
                visited[j] = True
                nbrs_j = neighbors[j]
                if nbrs_j.size >= int(min_samples):
                    for k in nbrs_j.tolist():
                        if k not in seeds:
                            seeds.append(k)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels


def fit_line_support(
    shape: tuple[int, int],
    points_xy: np.ndarray,
    band_half: float,
    dilate_iter: int,
) -> tuple[np.ndarray, float]:
    h, w = shape
    if points_xy.shape[0] == 0:
        return np.zeros(shape, dtype=np.uint8), 0.0
    pts = points_xy.astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    direction = np.array([vx, vy], dtype=np.float32)
    center = np.array([x0, y0], dtype=np.float32)
    rel = pts - center[None, :]
    proj = rel @ direction
    p0 = center + direction * float(proj.min())
    p1 = center + direction * float(proj.max())
    span = float(proj.max() - proj.min())

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    apx = xs - p0[0]
    apy = ys - p0[1]
    abx = float(p1[0] - p0[0])
    aby = float(p1[1] - p0[1])
    ab2 = max(abx * abx + aby * aby, 1e-6)
    t = np.clip((apx * abx + apy * aby) / ab2, 0.0, 1.0)
    projx = p0[0] + t * abx
    projy = p0[1] + t * aby
    dist = np.sqrt((xs - projx) ** 2 + (ys - projy) ** 2)
    band_mask = (dist <= float(band_half)).astype(np.uint8)

    pts_mask = np.zeros(shape, dtype=np.uint8)
    xs_i = np.clip(np.round(points_xy[:, 0]).astype(np.int32), 0, w - 1)
    ys_i = np.clip(np.round(points_xy[:, 1]).astype(np.int32), 0, h - 1)
    pts_mask[ys_i, xs_i] = 1
    if dilate_iter > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        pts_mask = cv2.dilate(pts_mask, kernel, iterations=int(dilate_iter))

    return np.maximum(band_mask, pts_mask), span


def detect_clusters_from_x(x_u8: np.ndarray, args: argparse.Namespace) -> list[dict]:
    thr = max(float(args.x_cand_min), float(np.percentile(x_u8, args.x_cand_percentile)))
    ys, xs = np.where(x_u8.astype(np.float32) >= thr)
    if len(xs) == 0:
        return []
    points_xy = np.stack([xs, ys], axis=1).astype(np.float32)
    labels = dbscan_points(points_xy, eps=float(args.dbscan_eps), min_samples=int(args.dbscan_min_samples))
    clusters = []
    for cid in sorted(set(labels.tolist())):
        if cid < 0:
            continue
        cur = points_xy[labels == cid]
        if cur.shape[0] < int(args.cluster_min_points):
            continue
        support_mask, span = fit_line_support(
            shape=x_u8.shape,
            points_xy=cur,
            band_half=float(args.cluster_band_half),
            dilate_iter=int(args.cluster_dilate),
        )
        if span < float(args.cluster_min_span):
            continue
        clusters.append(
            {
                "cluster_id": int(cid),
                "points_xy": cur,
                "support_mask": support_mask,
                "span": float(span),
            }
        )
    return clusters


def grow_component_from_seed(
    frame_u8: np.ndarray,
    allowed: np.ndarray,
    seed_xy: tuple[int, int],
    seed_ratio: float,
    std_weight: float,
) -> np.ndarray:
    x, y = seed_xy
    seed_val = float(frame_u8[y, x])
    vals = frame_u8[allowed > 0].astype(np.float32)
    if vals.size == 0:
        return np.zeros_like(frame_u8, dtype=np.uint8)
    local_mean = float(vals.mean())
    local_std = float(vals.std())
    lower = max(seed_val * float(seed_ratio), local_mean + local_std * float(std_weight))
    lower = min(lower, seed_val)
    thresh = ((allowed > 0) & (frame_u8.astype(np.float32) >= lower)).astype(np.uint8)
    thresh[y, x] = 1
    return connected_component_from_seed(thresh, seed_xy).astype(np.uint8)


def reconstruct_frame_masks(
    frame_u8s: list[np.ndarray],
    clusters: list[dict],
    args: argparse.Namespace,
) -> list[np.ndarray]:
    if not frame_u8s:
        return []
    seed_floor = max(float(args.seed_min_intensity), float(np.percentile(frame_u8s[0], args.seed_min_percentile)))
    outputs = [np.zeros_like(frame_u8s[0], dtype=np.uint8) for _ in frame_u8s]
    for fi, frame_u8 in enumerate(frame_u8s):
        cur_out = np.zeros_like(frame_u8, dtype=np.uint8)
        for cluster in clusters:
            allowed = cluster["support_mask"]
            vals = frame_u8[allowed > 0]
            if vals.size == 0:
                continue
            max_idx = int(vals.argmax())
            allowed_y, allowed_x = np.where(allowed > 0)
            seed_x = int(allowed_x[max_idx])
            seed_y = int(allowed_y[max_idx])
            seed_val = float(frame_u8[seed_y, seed_x])
            if seed_val < seed_floor:
                continue
            comp = grow_component_from_seed(
                frame_u8=frame_u8,
                allowed=allowed,
                seed_xy=(seed_x, seed_y),
                seed_ratio=float(args.grow_seed_ratio),
                std_weight=float(args.grow_std_weight),
            )
            if int(comp.sum()) < int(args.frame_min_pixels):
                continue
            cur_out = np.maximum(cur_out, comp)
        outputs[fi] = cur_out
    return outputs


def compute_track_metrics(gt_track: dict, frame_to_pred: dict[int, np.ndarray], frame_hit_thr: float) -> tuple[float, float, bool]:
    covs = []
    hits = []
    for frame_id in gt_track["frames"]:
        gt_mask = gt_track["frame_masks"][frame_id].astype(bool)
        pred_mask = frame_to_pred[frame_id].astype(bool)
        gt_area = int(gt_mask.sum())
        cov = 0.0 if gt_area <= 0 else float(np.logical_and(gt_mask, pred_mask).sum() / gt_area)
        covs.append(cov)
        hits.append(1 if cov >= frame_hit_thr else 0)
    tc_hard = float(sum(hits) / max(1, len(hits)))
    tc_soft = float(sum(covs) / max(1, len(covs)))
    ehr = bool(hits and hits[0] == 1 and hits[-1] == 1)
    return tc_hard, tc_soft, ehr


def main() -> None:
    args = parse_args()
    seq_ids = load_sequence_ids(args.test_json)
    if args.seq_ids:
        wanted = {f"{int(x):03d}" for x in args.seq_ids}
        seq_ids = [x for x in seq_ids if x in wanted]

    out_root = args.save_dir / args.run_name
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
        if not frame_ids:
            continue
        frames = load_sequence_frames(args.dataset_root, seq_id)
        x_u8, _t_u8, _xt_sc, frame_u8s = build_notebook_aligned_xt(
            frames,
            x_percentile=tuple(args.x_percentile),
            star_threshold=float(args.star_threshold),
            t_mode=str(args.t_mode),
            time_bg_percentile=float(args.time_bg_percentile),
            time_conf_threshold=float(args.time_conf_threshold),
            time_hist_energy_threshold=float(args.time_hist_energy_threshold),
            support_energy_threshold=float(args.support_energy_threshold),
            support_peak_percentile=float(args.support_peak_percentile),
            support_dilate=int(args.support_dilate),
        )
        clusters = detect_clusters_from_x(x_u8, args)
        pred_masks = reconstruct_frame_masks(frame_u8s=frame_u8s, clusters=clusters, args=args)
        gt_masks = [load_gt_mask(args.dataset_root, seq_id, frame_id) for frame_id in frame_ids]
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
            tc_hard, tc_soft, ehr = compute_track_metrics(
                gt_track=gt_track,
                frame_to_pred=frame_to_pred,
                frame_hit_thr=float(args.frame_hit_thr),
            )
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
            "num_clusters": len(clusters),
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
            f"[xonly_dbscan] seq {seq_id}: gt={n_gt} clusters={len(clusters)} "
            f"tdr={per_sequence[seq_id]['tdr']:.4f} "
            f"mean_tc_soft={per_sequence[seq_id]['mean_tc_soft']:.4f} "
            f"fcpf={per_sequence[seq_id]['false_components_per_frame']:.4f}",
            flush=True,
        )

    summary = {
        "model": "xonly_dbscan",
        "test_json": str(args.test_json),
        "frame_hit_thr": float(args.frame_hit_thr),
        "det_tc": float(args.det_tc),
        "complete_tc": float(args.complete_tc),
        "params": {
            "x_cand_percentile": float(args.x_cand_percentile),
            "x_cand_min": int(args.x_cand_min),
            "dbscan_eps": float(args.dbscan_eps),
            "dbscan_min_samples": int(args.dbscan_min_samples),
            "cluster_min_points": int(args.cluster_min_points),
            "cluster_min_span": float(args.cluster_min_span),
            "cluster_band_half": float(args.cluster_band_half),
            "cluster_dilate": int(args.cluster_dilate),
            "seed_min_percentile": float(args.seed_min_percentile),
            "seed_min_intensity": float(args.seed_min_intensity),
            "grow_seed_ratio": float(args.grow_seed_ratio),
            "grow_std_weight": float(args.grow_std_weight),
        },
        "metrics": {
            "num_sequences": len(per_sequence),
            "num_frames": total_frames,
            "num_gt_tracks": total_gt_tracks,
            "tdr": float(detected / max(1, total_gt_tracks)),
            "ctr": float(complete / max(1, total_gt_tracks)),
            "mean_tc_hard": float(total_tc_hard / max(1, total_gt_tracks)),
            "mean_tc_soft": float(total_tc_soft / max(1, total_gt_tracks)),
            "ehr": float(endpoint_hits / max(1, total_gt_tracks)),
            "false_components_total": int(total_false_components),
            "false_components_per_sequence": float(total_false_components / max(1, len(per_sequence))),
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
