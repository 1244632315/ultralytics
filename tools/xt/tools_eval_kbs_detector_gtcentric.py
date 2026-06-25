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

from ultralytics import YOLO  # noqa: E402
from tools.xt.tools_eval_kbs_detector_bbox import (  # noqa: E402
    build_notebook_aligned_xt,
    load_sequence_frames,
)
from tools.xt.tools_eval_kbs_detector_reconstruct_seg import reconstruct_sequence_masks  # noqa: E402
from tools.xt.tools_eval_kbs_seg_gtcentric import (  # noqa: E402
    cluster_gt_tracks,
    collect_frame_ids,
    compute_false_components,
    extract_frame_components,
    load_gt_mask,
    load_sequence_ids,
)


cv2.setNumThreads(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GT-centric trajectory evaluation for detector-reconstructed masks on KBS.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+")
    parser.add_argument(
        "--test-json",
        type=Path,
        default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "test_expanded_excluding_ft10.json",
    )
    parser.add_argument(
        "--ours-weights",
        type=Path,
        default=REPO_ROOT / "runs" / "obb" / "cmp_xt_sc_endpoint_v1_gpu_full" / "weights" / "best.pt",
    )
    parser.add_argument(
        "--xonly-weights",
        type=Path,
        default=REPO_ROOT / "runs" / "obb" / "cmp_x_only_gpu_full" / "weights" / "best.pt",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=50)
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
    parser.add_argument("--line-band-ratio", type=float, default=0.7)
    parser.add_argument("--line-band-min", type=float, default=2.0)
    parser.add_argument("--time-tol", type=float, default=18.0)
    parser.add_argument("--seed-intensity-weight", type=float, default=0.35)
    parser.add_argument("--grow-seed-ratio", type=float, default=0.55)
    parser.add_argument("--grow-std-weight", type=float, default=0.8)
    parser.add_argument("--gt-component-min-pixels", type=int, default=1)
    parser.add_argument("--cluster-max-link-distance", type=float, default=96.0)
    parser.add_argument("--cluster-max-frame-gap", type=int, default=2)
    parser.add_argument("--gt-min-track-len", type=int, default=1)
    parser.add_argument("--gt-min-total-pixels", type=int, default=4)
    parser.add_argument("--frame-hit-thr", type=float, default=0.1)
    parser.add_argument("--det-tc", type=float, default=0.5)
    parser.add_argument("--complete-tc", type=float, default=0.8)
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["ours_xt_sc", "yolo11n_obb_xonly"],
        choices=["ours_xt_sc", "yolo11n_obb_xonly"],
    )
    parser.add_argument("--seq-ids", nargs="*", default=None)
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "runs" / "trajectory_eval")
    parser.add_argument("--run-name", type=str, default="kbs_detector_gtcentric_exp82")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_track_metrics(
    gt_track: dict,
    frame_to_pred: dict[int, np.ndarray],
    frame_hit_thr: float,
) -> tuple[float, float, bool]:
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


def evaluate_detector(
    args: argparse.Namespace,
    det_name: str,
    model: YOLO,
    seq_ids: list[str],
    out_root: Path,
) -> dict:
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
        x_u8, t_u8, xt_sc, frame_u8s = build_notebook_aligned_xt(
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
        x_rgb = cv2.cvtColor(x_u8, cv2.COLOR_GRAY2BGR)
        inp = xt_sc if det_name == "ours_xt_sc" else x_rgb
        tmp_path = out_root / f"{det_name}_{seq_id}_input.png"
        cv2.imwrite(str(tmp_path), inp)

        results = model.predict(
            source=str(tmp_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
            save=False,
        )
        if not results:
            raise RuntimeError(f"No prediction result for {det_name} seq {seq_id}")

        pred_masks = reconstruct_sequence_masks(
            frame_u8s=frame_u8s,
            t_u8=t_u8,
            result=results[0],
            line_band_ratio=float(args.line_band_ratio),
            line_band_min=float(args.line_band_min),
            time_tol=float(args.time_tol),
            seed_intensity_weight=float(args.seed_intensity_weight),
            grow_seed_ratio=float(args.grow_seed_ratio),
            grow_std_weight=float(args.grow_std_weight),
        )
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
            f"[{det_name}] seq {seq_id}: gt={n_gt} "
            f"tdr={per_sequence[seq_id]['tdr']:.4f} "
            f"mean_tc_soft={per_sequence[seq_id]['mean_tc_soft']:.4f} "
            f"fcpf={per_sequence[seq_id]['false_components_per_frame']:.4f}",
            flush=True,
        )

    return {
        "model": det_name,
        "test_json": str(args.test_json),
        "frame_hit_thr": float(args.frame_hit_thr),
        "det_tc": float(args.det_tc),
        "complete_tc": float(args.complete_tc),
        "weights": str((args.ours_weights if det_name == "ours_xt_sc" else args.xonly_weights).expanduser().resolve()),
        "device": str(args.device),
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "t_mode": str(args.t_mode),
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


def main() -> None:
    args = parse_args()
    seq_ids = load_sequence_ids(args.test_json)
    if args.seq_ids:
        wanted = {f"{int(x):03d}" for x in args.seq_ids}
        seq_ids = [x for x in seq_ids if x in wanted]

    out_root = args.save_dir / args.run_name
    ensure_dir(out_root)

    models_all = {
        "ours_xt_sc": YOLO(str(args.ours_weights.expanduser().resolve())),
        "yolo11n_obb_xonly": YOLO(str(args.xonly_weights.expanduser().resolve())),
    }

    summaries: dict[str, dict] = {}
    for det_name in args.detectors:
        summary = evaluate_detector(
            args=args,
            det_name=det_name,
            model=models_all[det_name],
            seq_ids=seq_ids,
            out_root=out_root,
        )
        summaries[det_name] = summary
        out_json = out_root / f"{det_name}_summary.json"
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary["metrics"], indent=2))
        print(f"saved {out_json}")

    combined = {
        "dataset_root": str(args.dataset_root),
        "test_json": str(args.test_json),
        "detectors": {
            name: {
                "weights": summary["weights"],
                "metrics": summary["metrics"],
                "summary_path": str(out_root / f"{name}_summary.json"),
            }
            for name, summary in summaries.items()
        },
    }
    combined_path = out_root / "summary.json"
    combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))
    print(f"saved {combined_path}")


if __name__ == "__main__":
    main()
