from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import sep


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402
from tools.xt.tools_build_msod_stack_xt_sc import encode_xt_sc  # noqa: E402
from tools.xt.tools_eval_kbs_compare_bbox import (  # noqa: E402
    bbox_from_binary_mask,
    box_iou_xyxy,
    cluster_components_into_track_boxes,
    collect_frame_ids,
    extract_frame_components,
    load_gt_mask,
    load_sequence_ids,
    match_boxes,
)
from tools.xt.tools_eval_kbs_detector_reconstruct_seg import reconstruct_sequence_masks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector methods on KBS at trajectory bbox level.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+")
    parser.add_argument("--test-json", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "test.json")
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
    parser.add_argument("--component-min-pixels", type=int, default=4)
    parser.add_argument("--cluster-max-link-distance", type=float, default=96.0)
    parser.add_argument("--cluster-max-frame-gap", type=int, default=2)
    parser.add_argument("--pred-min-track-len", type=int, default=2)
    parser.add_argument("--pred-min-total-pixels", type=int, default=12)
    parser.add_argument("--gt-min-track-len", type=int, default=1)
    parser.add_argument("--gt-min-total-pixels", type=int, default=4)
    parser.add_argument("--gt-box-pad", type=float, default=4.0)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--pred-track-mode", choices=["direct_obb", "reconstruct"], default="direct_obb")
    parser.add_argument("--line-band-ratio", type=float, default=0.7)
    parser.add_argument("--line-band-min", type=float, default=2.0)
    parser.add_argument("--time-tol", type=float, default=18.0)
    parser.add_argument("--seed-intensity-weight", type=float, default=0.35)
    parser.add_argument("--grow-seed-ratio", type=float, default=0.55)
    parser.add_argument("--grow-std-weight", type=float, default=0.8)
    parser.add_argument("--seq-ids", nargs="*", default=None)
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["ours_xt_sc", "yolo11n_obb_xonly"],
        choices=["ours_xt_sc", "yolo11n_obb_xonly"],
    )
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "runs" / "kbs_eval")
    parser.add_argument("--run-name", type=str, default="kbs_detector_bbox_test")
    return parser.parse_args()


def trunc_img(img: np.ndarray, ratio: tuple[float, float]) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    vmin, vmax = np.percentile(img, ratio)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8) * 255.0
    return img.astype(np.uint8)


def load_sequence_frames(dataset_root: Path, seq_id: str) -> list[np.ndarray]:
    frames = []
    for path in sorted((dataset_root / f"{seq_id}_img").glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(img.astype(np.float32))
    if not frames:
        raise FileNotFoundError(f"No frames for sequence {seq_id}")
    return frames


def _build_support_mask(
    peak: np.ndarray,
    energy_conf: np.ndarray,
    support_energy_threshold: float,
    support_peak_percentile: float,
    support_dilate: int,
) -> np.ndarray:
    peak_thr = float(np.percentile(peak, support_peak_percentile))
    support = (energy_conf >= float(support_energy_threshold)) | (peak >= peak_thr)
    support = support.astype(np.uint8)
    if support_dilate > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        support = cv2.dilate(support, kernel, iterations=int(support_dilate))
    return support > 0


def build_kbs_t_map(
    stack: np.ndarray,
    mode: str,
    time_bg_percentile: float,
    time_conf_threshold: float,
    time_hist_energy_threshold: float,
    support_energy_threshold: float,
    support_peak_percentile: float,
    support_dilate: int,
) -> np.ndarray:
    n_frames = int(stack.shape[0])
    if n_frames <= 1:
        return np.zeros(stack.shape[1:], dtype=np.uint8)

    idx_hard = np.argmax(stack, axis=0).astype(np.int32)
    if mode == "argmax_legacy":
        denom = float(max(n_frames, 1))
        return np.clip(np.round(idx_hard.astype(np.float32) / denom * 255.0), 0, 255).astype(np.uint8)
    if mode == "argmax_fullrange":
        denom = float(max(n_frames - 1, 1))
        return np.clip(np.round(idx_hard.astype(np.float32) / denom * 255.0), 0, 255).astype(np.uint8)

    bg = np.percentile(stack, time_bg_percentile, axis=0).astype(np.float32)
    evidence = np.clip(stack - bg[None, ...], 0.0, None)
    idx_fg = evidence.argmax(axis=0).astype(np.int32)
    peak = evidence.max(axis=0)
    energy = evidence.sum(axis=0)
    energy_scale = float(np.percentile(energy, 99.9)) + 1e-6
    energy_conf = np.clip(energy / energy_scale, 0.0, 1.0)
    dominance = peak / (energy + 1e-6)
    conf = dominance * energy_conf

    hist_src = idx_fg[energy_conf < float(time_hist_energy_threshold)]
    if hist_src.size < 32:
        hist_src = idx_fg.reshape(-1)
    hist = np.bincount(hist_src.astype(np.int32), minlength=n_frames).astype(np.float64)
    if hist.sum() <= 0:
        hist = np.ones(n_frames, dtype=np.float64)
    prob = hist / hist.sum()

    rng = np.random.default_rng(0)
    idx_out = idx_fg.copy()
    low = conf < float(time_conf_threshold)
    if np.any(low):
        idx_out[low] = rng.choice(n_frames, size=int(low.sum()), p=prob)

    if mode == "hybrid_support":
        support = _build_support_mask(
            peak=peak,
            energy_conf=energy_conf,
            support_energy_threshold=support_energy_threshold,
            support_peak_percentile=support_peak_percentile,
            support_dilate=support_dilate,
        )
        idx_out = idx_out.copy()
        idx_out[~support] = 0

    denom = float(max(n_frames - 1, 1))
    return np.clip(np.round(idx_out.astype(np.float32) / denom * 255.0), 0, 255).astype(np.uint8)


def build_notebook_aligned_xt(
    frames: list[np.ndarray],
    x_percentile: tuple[float, float],
    star_threshold: float,
    t_mode: str,
    time_bg_percentile: float,
    time_conf_threshold: float,
    time_hist_energy_threshold: float,
    support_energy_threshold: float,
    support_peak_percentile: float,
    support_dilate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    stack = np.stack(frames, axis=0).astype(np.float32)
    med = np.median(stack, axis=0).astype(np.float32)
    oup = np.max(stack, axis=0).astype(np.float32)
    bkg = sep.Background(np.ascontiguousarray(med, np.float32))
    _, ext_map = sep.extract(
        med - np.array(bkg),
        float(star_threshold),
        err=bkg.globalrms,
        deblend_cont=1,
        segmentation_map=True,
    )
    star_mask = np.zeros_like(med, dtype=bool) if ext_map is None else (ext_map > 0)
    x_raw = oup.copy()
    if star_mask.any():
        x_raw[star_mask] = float(np.percentile(x_raw, 1.0))
    x_u8 = trunc_img(x_raw, x_percentile)
    t_u8 = build_kbs_t_map(
        stack=stack,
        mode=t_mode,
        time_bg_percentile=time_bg_percentile,
        time_conf_threshold=time_conf_threshold,
        time_hist_energy_threshold=time_hist_energy_threshold,
        support_energy_threshold=support_energy_threshold,
        support_peak_percentile=support_peak_percentile,
        support_dilate=support_dilate,
    )
    xt_sc = encode_xt_sc(x_u8, t_u8)

    frame_u8s = []
    for frame in frames:
        cur = frame.copy()
        if star_mask.any():
            cur[star_mask] = float(np.percentile(cur, 1.0))
        frame_u8s.append(trunc_img(cur, x_percentile))
    return x_u8, t_u8, xt_sc, frame_u8s


def obb_result_to_aabbs(result) -> list[list[float]]:
    obb = getattr(result, "obb", None)
    if obb is None or obb.xyxyxyxy is None:
        return []
    polys = obb.xyxyxyxy.detach().cpu().numpy()
    boxes = []
    for poly in polys:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        x1 = float(pts[:, 0].min())
        y1 = float(pts[:, 1].min())
        x2 = float(pts[:, 0].max() + 1.0)
        y2 = float(pts[:, 1].max() + 1.0)
        boxes.append([x1, y1, x2, y2])
    return boxes


def draw_boxes(base_img: np.ndarray, gt_boxes: list[list[float]], pred_boxes: list[list[float]]) -> np.ndarray:
    canvas = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    for box in gt_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
    for box in pred_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
    return canvas


def expand_box_xyxy(box: list[float], shape: tuple[int, int], pad: float) -> list[float]:
    h, w = shape
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, x1 - pad)
    y1 = max(0.0, y1 - pad)
    x2 = min(float(w), x2 + pad)
    y2 = min(float(h), y2 + pad)
    return [x1, y1, x2, y2]


def summarize(tp: int, fp: int, fn: int, num_sequences: int, total_frames: int) -> dict[str, float]:
    return {
        "num_gt_tracks": int(tp + fn),
        "num_pred_tracks": int(tp + fp),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "recall": float(tp / (tp + fn + 1e-12)),
        "false_alarms_per_sequence": float(fp / max(1, num_sequences)),
        "false_alarms_per_frame": float(fp / max(1, total_frames)),
        "precision": float(tp / (tp + fp + 1e-12)),
    }


def main() -> None:
    args = parse_args()
    seq_ids = load_sequence_ids(args.test_json)
    if args.seq_ids:
        wanted = {f"{int(x):03d}" for x in args.seq_ids}
        seq_ids = [x for x in seq_ids if x in wanted]

    out_root = args.save_dir / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    models_all = {
        "ours_xt_sc": YOLO(str(args.ours_weights.expanduser().resolve())),
        "yolo11n_obb_xonly": YOLO(str(args.xonly_weights.expanduser().resolve())),
    }
    models = {name: models_all[name] for name in args.detectors}

    det_summaries = {}
    for det_name, model in models.items():
        save_root = out_root / det_name
        (save_root / "vis_boxes").mkdir(parents=True, exist_ok=True)
        total_tp = total_fp = total_fn = 0
        total_frames = 0
        per_sequence = {}

        for seq_idx, seq_id in enumerate(seq_ids, start=1):
            frame_ids = collect_frame_ids(args.dataset_root, seq_id)
            if not frame_ids:
                continue
            total_frames += len(frame_ids)
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
            tmp_path = save_root / f"{seq_id}_input.png"
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
            direct_pred_boxes = obb_result_to_aabbs(results[0])

            gt_frame_masks = [load_gt_mask(args.dataset_root, seq_id, frame_id, 0) for frame_id in frame_ids]
            gt_components = [
                extract_frame_components(mask, frame_id, min_pixels=max(1, args.component_min_pixels))
                for mask, frame_id in zip(gt_frame_masks, frame_ids)
            ]
            gt_boxes, gt_tracks, _gt_union = cluster_components_into_track_boxes(
                gt_components,
                shape=gt_frame_masks[0].shape,
                max_link_distance=args.cluster_max_link_distance,
                max_frame_gap=args.cluster_max_frame_gap,
                min_track_len=args.gt_min_track_len,
                min_total_pixels=args.gt_min_total_pixels,
                min_pixels_for_box=max(1, args.component_min_pixels),
            )
            gt_boxes = [expand_box_xyxy(box, gt_frame_masks[0].shape, float(args.gt_box_pad)) for box in gt_boxes]
            for track_row, box in zip(gt_tracks, gt_boxes):
                track_row["bbox_xyxy"] = [float(v) for v in box]

            pred_tracks = []
            pred_union = np.zeros_like(gt_frame_masks[0], dtype=np.uint8)
            if args.pred_track_mode == "reconstruct":
                pred_frame_masks = reconstruct_sequence_masks(
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
                pred_components = [
                    extract_frame_components(mask, frame_id, min_pixels=args.component_min_pixels)
                    for mask, frame_id in zip(pred_frame_masks, frame_ids)
                ]
                pred_boxes, pred_tracks, pred_union = cluster_components_into_track_boxes(
                    pred_components,
                    shape=gt_frame_masks[0].shape,
                    max_link_distance=args.cluster_max_link_distance,
                    max_frame_gap=args.cluster_max_frame_gap,
                    min_track_len=args.pred_min_track_len,
                    min_total_pixels=args.pred_min_total_pixels,
                    min_pixels_for_box=args.component_min_pixels,
                )
            else:
                pred_boxes = direct_pred_boxes

            tp, fp, fn, matches = match_boxes(pred_boxes, gt_boxes, match_iou=args.match_iou)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            vis = draw_boxes(x_u8, gt_boxes, pred_boxes)
            cv2.imwrite(str(save_root / "vis_boxes" / f"{seq_id}.png"), vis)
            per_sequence[seq_id] = {
                "num_frames": len(frame_ids),
                "num_gt_tracks": len(gt_boxes),
                "num_pred_tracks": len(pred_boxes),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "recall": float(tp / len(gt_boxes)) if gt_boxes else 1.0,
                "false_alarms": fp,
                "pred_track_mode": str(args.pred_track_mode),
                "direct_pred_boxes_xyxy": [[float(v) for v in box] for box in direct_pred_boxes],
                "pred_boxes_xyxy": [[float(v) for v in box] for box in pred_boxes],
                "gt_boxes_xyxy": [[float(v) for v in box] for box in gt_boxes],
                "pred_tracks": pred_tracks,
                "gt_tracks": gt_tracks,
                "matches": matches,
            }
            print(
                f"[{det_name}] seq {seq_idx}/{len(seq_ids)} {seq_id}: "
                f"gt={len(gt_boxes)} pred={len(pred_boxes)} tp={tp} fp={fp} fn={fn}",
                flush=True,
            )

        summary = {
            "detector": det_name,
            "dataset_root": str(args.dataset_root),
            "test_json": str(args.test_json),
            "weights": str((args.ours_weights if det_name == 'ours_xt_sc' else args.xonly_weights).expanduser().resolve()),
            "device": args.device,
            "imgsz": int(args.imgsz),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "t_mode": str(args.t_mode),
            "pred_track_mode": str(args.pred_track_mode),
            "matching_iou": float(args.match_iou),
            "gt_box_pad": float(args.gt_box_pad),
            "num_sequences": len(per_sequence),
            "num_frames": total_frames,
            "metrics": summarize(total_tp, total_fp, total_fn, len(per_sequence), total_frames),
            "per_sequence": per_sequence,
        }
        out_json = save_root / "summary.json"
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary["metrics"], indent=2))
        print(f"saved {out_json}")
        det_summaries[det_name] = summary

    all_summary = {
        "dataset_root": str(args.dataset_root),
        "test_json": str(args.test_json),
        "detectors": {
            k: {
                "weights": v["weights"],
                "metrics": v["metrics"],
                "summary_path": str(out_root / k / "summary.json"),
            }
            for k, v in det_summaries.items()
        },
    }
    all_path = out_root / "summary.json"
    all_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    print(json.dumps(all_summary, indent=2))
    print(f"saved {all_path}")


if __name__ == "__main__":
    main()
