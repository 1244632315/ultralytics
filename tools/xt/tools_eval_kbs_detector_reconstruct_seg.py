from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import sep


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402
from tools.xt.tools_build_msod_stack_xt_sc import encode_xt_sc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype detector-guided framewise mask reconstruction on KBS real sequences."
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+")
    parser.add_argument("--test-json", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "test.json")
    parser.add_argument("--seq-ids", nargs="*", default=None)
    parser.add_argument(
        "--ours-weights",
        type=Path,
        default=REPO_ROOT / "runs" / "obb" / "cmp_xt_sc_endpoint_v1_gpu_full" / "weights" / "best.pt",
    )
    parser.add_argument(
        "--xonly-weights",
        type=Path,
        default=REPO_ROOT / "runs" / "obb" / "cmp_v11n_x_cmp100_xonly_v8v11v26_v11scalex_20260402" / "weights" / "best.pt",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--x-percentile", type=float, nargs=2, default=(0.01, 99.99))
    parser.add_argument("--star-threshold", type=float, default=1.5)
    parser.add_argument("--line-band-ratio", type=float, default=0.7)
    parser.add_argument("--line-band-min", type=float, default=2.0)
    parser.add_argument("--time-tol", type=float, default=18.0)
    parser.add_argument("--seed-intensity-weight", type=float, default=0.35)
    parser.add_argument("--grow-seed-ratio", type=float, default=0.55)
    parser.add_argument("--grow-std-weight", type=float, default=0.8)
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "runs" / "kbs_eval")
    parser.add_argument("--run-name", type=str, default="kbs_detector_guided_reconstruct_smoke")
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
    out: list[np.ndarray] = []
    for path in sorted((dataset_root / f"{seq_id}_img").glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out.append(img.astype(np.float32))
    if not out:
        raise FileNotFoundError(f"No image frames for sequence {seq_id}")
    return out


def load_seq_ids(test_json: Path) -> list[str]:
    ids = json.loads(test_json.read_text(encoding="utf-8"))
    return [f"{int(x):03d}" for x in ids]


def load_gt_frame_masks(dataset_root: Path, seq_id: str) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for path in sorted((dataset_root / f"{seq_id}_gt").glob("*.png")):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(path)
        out.append((mask > 0).astype(np.uint8))
    if not out:
        raise FileNotFoundError(f"No gt masks for sequence {seq_id}")
    return out


def build_notebook_aligned_xt(
    frames: list[np.ndarray],
    x_percentile: tuple[float, float],
    star_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    stack = np.stack(frames, axis=0).astype(np.float32)
    med = np.median(stack, axis=0).astype(np.float32)
    oup = np.max(stack, axis=0).astype(np.float32)
    max_idx = (np.argmax(stack, axis=0).astype(np.float32) / float(max(stack.shape[0], 1))) * 255.0
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
    t_u8 = np.clip(np.round(max_idx), 0, 255).astype(np.uint8)
    xt_sc = encode_xt_sc(x_u8, t_u8)

    frame_u8s: list[np.ndarray] = []
    for frame in frames:
        cur = frame.copy()
        if star_mask.any():
            cur[star_mask] = float(np.percentile(cur, 1.0))
        frame_u8s.append(trunc_img(cur, x_percentile))
    return x_u8, t_u8, xt_sc, star_mask.astype(np.uint8), frame_u8s


def draw_obbs(gray: np.ndarray, result) -> np.ndarray:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    obb = getattr(result, "obb", None)
    if obb is None or obb.xyxyxyxy is None:
        return canvas
    polys = obb.xyxyxyxy.detach().cpu().numpy()
    confs = None if getattr(obb, "conf", None) is None else obb.conf.detach().cpu().numpy()
    for i, poly in enumerate(polys):
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 255, 255), 2)
        if confs is not None:
            x0, y0 = pts.reshape(-1, 2)[0]
            cv2.putText(canvas, f"{float(confs[i]):.2f}", (int(x0), int(y0) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return canvas


def frame_time_value(frame_idx: int, num_frames: int) -> float:
    return 255.0 * float(frame_idx) / float(max(num_frames, 1))


def poly_to_mask(poly: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.round(np.asarray(poly, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def obb_geometry(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    if w >= h:
        long_len, short_len = float(w), float(h)
        theta = np.deg2rad(angle)
    else:
        long_len, short_len = float(h), float(w)
        theta = np.deg2rad(angle + 90.0)
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    center = np.array([cx, cy], dtype=np.float32)
    p0 = center - direction * (long_len * 0.5)
    p1 = center + direction * (long_len * 0.5)
    return p0, p1, long_len, short_len


def narrow_band_mask(shape: tuple[int, int], p0: np.ndarray, p1: np.ndarray, half_width: float) -> np.ndarray:
    h, w = shape
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
    return (dist <= float(half_width)).astype(np.uint8)


def find_seed(
    frame_u8: np.ndarray,
    t_u8: np.ndarray,
    allowed: np.ndarray,
    target_t: float,
    time_tol: float,
    seed_intensity_weight: float,
) -> tuple[int, int] | None:
    ys, xs = np.where(allowed > 0)
    if len(xs) == 0:
        return None
    dt = np.abs(t_u8[ys, xs].astype(np.float32) - float(target_t))
    intens = frame_u8[ys, xs].astype(np.float32) / 255.0
    score = -dt + intens * float(time_tol) * float(seed_intensity_weight)
    keep = dt <= float(time_tol)
    if keep.any():
        ys = ys[keep]
        xs = xs[keep]
        score = score[keep]
    best = int(np.argmax(score))
    return int(xs[best]), int(ys[best])


def connected_component_from_seed(bin_mask: np.ndarray, seed_xy: tuple[int, int]) -> np.ndarray:
    x, y = seed_xy
    if not (0 <= y < bin_mask.shape[0] and 0 <= x < bin_mask.shape[1]):
        return np.zeros_like(bin_mask, dtype=np.uint8)
    if bin_mask[y, x] == 0:
        return np.zeros_like(bin_mask, dtype=np.uint8)
    visited = np.zeros_like(bin_mask, dtype=np.uint8)
    q: deque[tuple[int, int]] = deque([(x, y)])
    visited[y, x] = 1
    while q:
        cx, cy = q.popleft()
        for ny in range(max(0, cy - 1), min(bin_mask.shape[0], cy + 2)):
            for nx in range(max(0, cx - 1), min(bin_mask.shape[1], cx + 2)):
                if visited[ny, nx] or bin_mask[ny, nx] == 0:
                    continue
                visited[ny, nx] = 1
                q.append((nx, ny))
    return visited


def grow_from_seed(
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
    return connected_component_from_seed(thresh, seed_xy)


def reconstruct_sequence_masks(
    frame_u8s: list[np.ndarray],
    t_u8: np.ndarray,
    result,
    line_band_ratio: float,
    line_band_min: float,
    time_tol: float,
    seed_intensity_weight: float,
    grow_seed_ratio: float,
    grow_std_weight: float,
) -> list[np.ndarray]:
    out = [np.zeros_like(frame_u8s[0], dtype=np.uint8) for _ in frame_u8s]
    obb = getattr(result, "obb", None)
    if obb is None or obb.xyxyxyxy is None:
        return out
    polys = obb.xyxyxyxy.detach().cpu().numpy()
    for poly in polys:
        poly_mask = poly_to_mask(poly, frame_u8s[0].shape)
        p0, p1, _, short_len = obb_geometry(poly)
        band_half = max(float(line_band_min), float(short_len) * float(line_band_ratio) * 0.5)
        band_mask = narrow_band_mask(frame_u8s[0].shape, p0, p1, band_half)
        candidate_mask = ((poly_mask > 0) & (band_mask > 0)).astype(np.uint8)
        if candidate_mask.sum() == 0:
            candidate_mask = poly_mask
        for fi, frame_u8 in enumerate(frame_u8s):
            seed = find_seed(
                frame_u8,
                t_u8,
                candidate_mask,
                target_t=frame_time_value(fi, len(frame_u8s)),
                time_tol=time_tol,
                seed_intensity_weight=seed_intensity_weight,
            )
            if seed is None:
                continue
            comp = grow_from_seed(
                frame_u8,
                poly_mask,
                seed,
                seed_ratio=grow_seed_ratio,
                std_weight=grow_std_weight,
            )
            out[fi] = np.maximum(out[fi], comp.astype(np.uint8))
    return out


def compute_metrics(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for pred, gt in zip(preds, gts):
        tp += int(np.logical_and(pred == 1, gt == 1).sum())
        fp += int(np.logical_and(pred == 1, gt == 0).sum())
        fn += int(np.logical_and(pred == 0, gt == 1).sum())
        tn += int(np.logical_and(pred == 0, gt == 0).sum())
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
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def overlay_frame(frame_u8: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    out = base.copy()
    out[tp] = (0, 255, 0)
    out[fp] = (0, 0, 255)
    out[fn] = (255, 255, 0)
    return out


def save_frame_set(root: Path, frames: list[np.ndarray], suffix: str = ".png") -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        path = root / f"{i:05d}{suffix}"
        cv2.imwrite(str(path), frame)
        paths.append(str(path))
    return paths


def save_sequence_artifacts(
    seq_root: Path,
    detector_name: str,
    frame_u8s: list[np.ndarray],
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    input_vis: np.ndarray,
    obb_vis: np.ndarray,
    x_u8: np.ndarray,
    t_u8: np.ndarray,
    detector_input: np.ndarray,
) -> dict[str, str]:
    det_root = seq_root / detector_name
    raw_root = det_root / "frames"
    gt_root = det_root / "gt_masks"
    pred_root = det_root / "pred_masks"
    overlay_root = det_root / "overlays"
    save_frame_set(raw_root, frame_u8s)
    save_frame_set(gt_root, [(m * 255).astype(np.uint8) for m in gt_masks])
    save_frame_set(pred_root, [(m * 255).astype(np.uint8) for m in pred_masks])
    save_frame_set(overlay_root, [overlay_frame(f, g, p) for f, g, p in zip(frame_u8s, gt_masks, pred_masks)])
    x_path = det_root / "x.png"
    t_path = det_root / "t.png"
    inp_path = det_root / "input.png"
    obb_path = det_root / "obb.png"
    cv2.imwrite(str(x_path), x_u8)
    cv2.imwrite(str(t_path), t_u8)
    cv2.imwrite(str(inp_path), detector_input)
    cv2.imwrite(str(obb_path), obb_vis)
    return {
        "x_path": str(x_path),
        "t_path": str(t_path),
        "input_path": str(inp_path),
        "obb_path": str(obb_path),
        "frames_dir": str(raw_root),
        "gt_dir": str(gt_root),
        "pred_dir": str(pred_root),
        "overlay_dir": str(overlay_root),
    }


def main() -> None:
    args = parse_args()
    seq_ids = load_seq_ids(args.test_json)
    if args.seq_ids:
        wanted = {f"{int(x):03d}" for x in args.seq_ids}
        seq_ids = [x for x in seq_ids if x in wanted]
    out_root = args.save_dir / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    ours_model = YOLO(str(args.ours_weights.expanduser().resolve()))
    xonly_model = YOLO(str(args.xonly_weights.expanduser().resolve()))

    summary: dict[str, object] = {
        "dataset_root": str(args.dataset_root),
        "test_json": str(args.test_json),
        "seq_ids": list(seq_ids),
        "device": args.device,
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "detectors": {},
    }
    detector_totals: dict[str, dict[str, float]] = {}

    for seq_id in seq_ids:
        frames = load_sequence_frames(args.dataset_root, seq_id)
        gt_masks = load_gt_frame_masks(args.dataset_root, seq_id)
        x_u8, t_u8, xt_sc, _star_mask, frame_u8s = build_notebook_aligned_xt(
            frames,
            x_percentile=tuple(args.x_percentile),
            star_threshold=float(args.star_threshold),
        )
        x_rgb = cv2.cvtColor(x_u8, cv2.COLOR_GRAY2BGR)

        seq_root = out_root / seq_id
        seq_root.mkdir(parents=True, exist_ok=True)
        xt_path = seq_root / "tmp_xt_sc.png"
        xonly_path = seq_root / "tmp_x_rgb.png"
        cv2.imwrite(str(xt_path), xt_sc)
        cv2.imwrite(str(xonly_path), x_rgb)

        det_cfgs = {
            "ours_xt_sc": (ours_model, xt_path, xt_sc),
            "yolo11n_obb_xonly": (xonly_model, xonly_path, x_rgb),
        }

        seq_summary: dict[str, object] = {}
        for det_name, (model, source_path, input_img) in det_cfgs.items():
            results = model.predict(
                source=str(source_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
                save=False,
            )
            if not results:
                raise RuntimeError(f"No prediction result for {det_name} on seq {seq_id}")
            result = results[0]
            pred_masks = reconstruct_sequence_masks(
                frame_u8s=frame_u8s,
                t_u8=t_u8,
                result=result,
                line_band_ratio=float(args.line_band_ratio),
                line_band_min=float(args.line_band_min),
                time_tol=float(args.time_tol),
                seed_intensity_weight=float(args.seed_intensity_weight),
                grow_seed_ratio=float(args.grow_seed_ratio),
                grow_std_weight=float(args.grow_std_weight),
            )
            obb = getattr(result, "obb", None)
            num_boxes = 0 if obb is None or obb.xyxyxyxy is None else int(obb.xyxyxyxy.shape[0])
            metrics = compute_metrics(pred_masks, gt_masks)
            totals = detector_totals.setdefault(
                det_name,
                {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tn": 0,
                    "num_sequences": 0,
                    "num_pred_boxes": 0,
                },
            )
            totals["tp"] += int(metrics["tp"])
            totals["fp"] += int(metrics["fp"])
            totals["fn"] += int(metrics["fn"])
            totals["tn"] += int(metrics["tn"])
            totals["num_sequences"] += 1
            totals["num_pred_boxes"] += int(num_boxes)
            artifacts = save_sequence_artifacts(
                seq_root=seq_root,
                detector_name=det_name,
                frame_u8s=frame_u8s,
                gt_masks=gt_masks,
                pred_masks=pred_masks,
                input_vis=input_img,
                obb_vis=draw_obbs(x_u8, result),
                x_u8=x_u8,
                t_u8=t_u8,
                detector_input=input_img,
            )
            seq_summary[det_name] = {
                "num_pred_boxes": num_boxes,
                "metrics": metrics,
                **artifacts,
            }
            print(f"[{seq_id}] {det_name}: {json.dumps(metrics)}")
        summary["detectors"][seq_id] = seq_summary

    overall = {}
    for det_name, totals in detector_totals.items():
        tp = int(totals["tp"])
        fp = int(totals["fp"])
        fn = int(totals["fn"])
        tn = int(totals["tn"])
        overall[det_name] = {
            "num_sequences": int(totals["num_sequences"]),
            "num_pred_boxes": int(totals["num_pred_boxes"]),
        }
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        iou = tp / (tp + fp + fn + 1e-12)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-12)
        acc = (tp + tn) / (tp + fp + fn + tn + 1e-12)
        overall[det_name].update(
            {
                "precision": float(precision),
                "recall": float(recall),
                "iou": float(iou),
                "dice_f1": float(dice),
                "pixel_accuracy": float(acc),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    summary["overall"] = overall

    out_json = out_root / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
