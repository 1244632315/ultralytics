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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector models on KBS real sequences using sequence-level union-mask segmentation metrics.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+")
    parser.add_argument("--test-json", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "test.json")
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "runs" / "obb" / "cmp_xt_sc_endpoint_v1_gpu_full" / "weights" / "best.pt",
    )
    parser.add_argument("--mode", type=str, default="xt_sc", choices=["xt_sc", "xonly"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--seq-ids", nargs="*", default=None, help="Optional explicit sequence ids such as 006 059.")
    parser.add_argument("--x-percentile", type=float, nargs=2, default=(0.01, 99.99))
    parser.add_argument("--star-threshold", type=float, default=1.5)
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "runs" / "kbs_eval")
    parser.add_argument("--run-name", type=str, default="kbs_detector_seqmask_notebook_aligned")
    return parser.parse_args()


def trunc_img(img: np.ndarray, ratio: tuple[float, float]) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    vmin, vmax = np.percentile(img, ratio)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8) * 255.0
    return img.astype(np.uint8)


def load_seq_ids(test_json: Path) -> list[str]:
    ids = json.loads(test_json.read_text(encoding="utf-8"))
    return [f"{int(x):03d}" for x in ids]


def load_sequence_frames(dataset_root: Path, seq_id: str) -> list[np.ndarray]:
    out = []
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


def load_gt_union_mask(dataset_root: Path, seq_id: str) -> np.ndarray:
    masks = []
    for path in sorted((dataset_root / f"{seq_id}_gt").glob("*.png")):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(path)
        masks.append(mask > 0)
    if not masks:
        raise FileNotFoundError(f"No gt masks for sequence {seq_id}")
    return np.logical_or.reduce(np.stack(masks, axis=0), axis=0).astype(np.uint8)


def build_notebook_aligned_xt(
    frames: list[np.ndarray],
    x_percentile: tuple[float, float],
    star_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return x_u8, t_u8, xt_sc, star_mask.astype(np.uint8)


def rasterize_obb_predictions(result, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    obb = getattr(result, "obb", None)
    if obb is None or obb.xyxyxyxy is None:
        return mask
    polys = obb.xyxyxyxy.detach().cpu().numpy()
    for poly in polys:
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def draw_obb_predictions(img: np.ndarray, result) -> np.ndarray:
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    obb = getattr(result, "obb", None)
    if obb is None or obb.xyxyxyxy is None:
        return canvas
    polys = obb.xyxyxyxy.detach().cpu().numpy()
    confs = None if getattr(obb, "conf", None) is None else obb.conf.detach().cpu().numpy()
    for i, poly in enumerate(polys):
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        if confs is not None:
            x0, y0 = pts.reshape(-1, 2)[0]
            cv2.putText(
                canvas,
                f"{float(confs[i]):.2f}",
                (int(x0), int(y0) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return canvas


def make_overlay(x_u8: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(x_u8, cv2.COLOR_GRAY2BGR)
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    out = base.copy()
    out[tp] = (0, 255, 0)
    out[fp] = (0, 0, 255)
    out[fn] = (255, 255, 0)
    return out


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
    seq_ids = load_seq_ids(args.test_json)
    if args.seq_ids:
        wanted = {f"{int(x):03d}" for x in args.seq_ids}
        seq_ids = [x for x in seq_ids if x in wanted]
    out_root = args.save_dir / args.run_name
    vis_root = out_root / "vis_mask"
    obb_vis_root = out_root / "vis_obb"
    xt_root = out_root / "xt_sc"
    x_root = out_root / "x"
    t_root = out_root / "t"
    star_root = out_root / "star_mask"
    pred_root = out_root / "pred_mask"
    gt_root = out_root / "gt_union"
    for path in [vis_root, obb_vis_root, xt_root, x_root, t_root, star_root, pred_root, gt_root]:
        path.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights.expanduser().resolve()))

    tp = fp = fn = tn = 0
    per_sequence = {}

    for seq_id in seq_ids:
        frames = load_sequence_frames(args.dataset_root, seq_id)
        x_u8, t_u8, xt_sc, star_mask = build_notebook_aligned_xt(
            frames,
            x_percentile=tuple(args.x_percentile),
            star_threshold=float(args.star_threshold),
        )
        x_path = x_root / f"{seq_id}.png"
        t_path = t_root / f"{seq_id}.png"
        star_path = star_root / f"{seq_id}.png"
        cv2.imwrite(str(x_path), x_u8)
        cv2.imwrite(str(t_path), t_u8)
        cv2.imwrite(str(star_path), star_mask * 255)
        xt_path = xt_root / f"{seq_id}.png"
        cv2.imwrite(str(xt_path), xt_sc)

        if args.mode == "xonly":
            inp = cv2.cvtColor(x_u8, cv2.COLOR_GRAY2BGR)
        else:
            inp = xt_sc

        results = model.predict(
            source=inp,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
            save=False,
        )
        if not results:
            raise RuntimeError(f"No result returned for {seq_id}")
        pred_mask = rasterize_obb_predictions(results[0], x_u8.shape)
        gt_mask = load_gt_union_mask(args.dataset_root, seq_id)

        tp_i = int(np.logical_and(pred_mask == 1, gt_mask == 1).sum())
        fp_i = int(np.logical_and(pred_mask == 1, gt_mask == 0).sum())
        fn_i = int(np.logical_and(pred_mask == 0, gt_mask == 1).sum())
        tn_i = int(np.logical_and(pred_mask == 0, gt_mask == 0).sum())
        tp += tp_i
        fp += fp_i
        fn += fn_i
        tn += tn_i

        cv2.imwrite(str(pred_root / f"{seq_id}.png"), pred_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(gt_root / f"{seq_id}.png"), gt_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(vis_root / f"{seq_id}.png"), make_overlay(x_u8, gt_mask, pred_mask))
        cv2.imwrite(str(obb_vis_root / f"{seq_id}.png"), draw_obb_predictions(x_u8, results[0]))

        obb = getattr(results[0], "obb", None)
        num_boxes = 0 if obb is None or obb.xyxyxyxy is None else int(obb.xyxyxyxy.shape[0])
        per_sequence[seq_id] = {
            "metrics": compute_metrics(tp_i, fp_i, fn_i, tn_i),
            "num_pred_boxes": num_boxes,
            "mode": str(args.mode),
            "x_path": str(x_path),
            "t_path": str(t_path),
            "xt_sc_path": str(xt_path),
            "star_mask_path": str(star_path),
            "pred_mask_path": str(pred_root / f"{seq_id}.png"),
            "gt_union_path": str(gt_root / f"{seq_id}.png"),
            "mask_overlay_path": str(vis_root / f"{seq_id}.png"),
            "obb_overlay_path": str(obb_vis_root / f"{seq_id}.png"),
        }

    summary = {
        "dataset_root": str(args.dataset_root),
        "test_json": str(args.test_json),
        "weights": str(args.weights.expanduser().resolve()),
        "mode": str(args.mode),
        "device": args.device,
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "x_percentile": list(args.x_percentile),
        "xt_builder": "notebook_aligned_real_export_logic",
        "x_builder": "max_projection_after_star_suppression",
        "t_builder": "argmax_frame_index_map",
        "star_threshold": float(args.star_threshold),
        "num_sequences": len(seq_ids),
        "metrics": compute_metrics(tp, fp, fn, tn),
        "per_sequence": per_sequence,
        "save_root": str(out_root),
    }
    out_json = out_root / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
