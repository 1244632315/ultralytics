from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = REPO_ROOT / "compares" / "MSAMNet-master"
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.dataloader import MSODataset  # noqa: E402
from model.CSAUNet import CSAUNet  # noqa: E402
from model.DNANet import DNANet, Res_CBAM_block  # noqa: E402
from model.DnTNet import DnTNet  # noqa: E402
from model.MSAMNet import AAFE, CBAM, Connection, CoordAtt, MSAMNet, SE  # noqa: E402
from model.utils import load_param  # noqa: E402
from ultralytics.utils import ops  # noqa: E402
from ultralytics.utils.metrics import ap_per_class, batch_probiou  # noqa: E402


FUSION_BLOCKS = {
    "CBAM": CBAM,
    "AAFE": AAFE,
    "CA": CoordAtt,
    "SE": SE,
    "None": Connection,
}

TRAJECTORY_KEYS = [
    ("line_obb", "line"),
    ("pt_obb", "point"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate compare-model outputs by stacking per-frame masks and matching stacked OBBs."
    )
    parser.add_argument("--model", choices=["CSAUNet", "DNANet", "DnTNet", "MSAMNet"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "xt_seq_msod_x")
    parser.add_argument("--dataset-name", type=str, default="MSOD_RAWBG")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--t-frame", type=int, default=3)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--suffix", type=str, default=".tif")
    parser.add_argument("--fusionblock", type=str, default="AAFE")
    parser.add_argument("--channel-size", type=str, default="three")
    parser.add_argument("--backbone", type=str, default="resnet_18")
    parser.add_argument("--deep-supervision", type=str, default="False")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pred-thr", type=float, default=0.0, help="Logit threshold. 0 means sigmoid 0.5.")
    parser.add_argument("--stack-frames", type=int, default=0, help="0 means all frames in sequence.")
    parser.add_argument("--stack-mode", choices=["max", "sum"], default="max")
    parser.add_argument("--obb-fit-mode", choices=["pca", "minrect", "trajectory_strip"], default="pca")
    parser.add_argument("--component-min-pixels", type=int, default=4)
    parser.add_argument("--component-dilate", type=int, default=1)
    parser.add_argument(
        "--pred-agg-mode",
        choices=["components", "gt_partition", "trajectory_cluster"],
        default="components",
        help="How to convert the stacked prediction mask into OBB detections.",
    )
    parser.add_argument("--score-reduction", choices=["max", "mean"], default="max")
    parser.add_argument("--cluster-max-link-distance", type=float, default=96.0)
    parser.add_argument("--cluster-max-frame-gap", type=int, default=2)
    parser.add_argument("--cluster-min-track-len", type=int, default=2)
    parser.add_argument("--cluster-min-total-pixels", type=int, default=12)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "runs" / "xt_compare_stack_obb")
    parser.add_argument("--save-labels", action="store_true")
    parser.add_argument("--save-stacks", action="store_true")
    return parser.parse_args()


def order_quad_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(ang)]
    i0 = np.argmin(pts[:, 0] + pts[:, 1])
    return np.roll(pts, -i0, axis=0)


def obb_from_points_pca(pts: np.ndarray, min_pixels: int = 4) -> np.ndarray | None:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) < min_pixels:
        return None
    center = pts.mean(axis=0)
    pts0 = pts - center
    cov = np.cov(pts0, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    v1 = eigvecs[:, order[0]].astype(np.float32)
    v2 = eigvecs[:, order[1]].astype(np.float32)
    if (v1[0] * v2[1] - v1[1] * v2[0]) < 0:
        v2 = -v2
    p1 = pts0 @ v1
    p2 = pts0 @ v2
    min1, max1 = float(p1.min()), float(p1.max())
    min2, max2 = float(p2.min()), float(p2.max())
    if (max1 - min1) < 1e-6 or (max2 - min2) < 1e-6:
        return None
    local = np.array([[min1, min2], [max1, min2], [max1, max2], [min1, max2]], dtype=np.float32)
    corners = center[None, :] + local[:, :1] * v1[None, :] + local[:, 1:] * v2[None, :]
    return order_quad_clockwise(corners)


def obb_from_points_minrect(pts: np.ndarray, min_pixels: int = 4) -> np.ndarray | None:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) < min_pixels:
        return None
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (_, _), (w, h), _ = rect
    if float(w) < 1e-6 or float(h) < 1e-6:
        return None
    corners = cv2.boxPoints(rect).astype(np.float32)
    return order_quad_clockwise(corners)


def obb_from_points(pts: np.ndarray, min_pixels: int = 4, fit_mode: str = "pca") -> np.ndarray | None:
    if fit_mode == "minrect":
        return obb_from_points_minrect(pts, min_pixels=min_pixels)
    return obb_from_points_pca(pts, min_pixels=min_pixels)


def polygon_area(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def to_yolo_obb_line(obb: np.ndarray, w: int, h: int, conf: float | None = None) -> str | None:
    pts = np.asarray(obb, dtype=np.float32).reshape(4, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, w)
    pts[:, 1] = np.clip(pts[:, 1], 0, h)
    pts = order_quad_clockwise(pts)
    if polygon_area(pts) < 2.0:
        return None
    pts[:, 0] /= float(w)
    pts[:, 1] /= float(h)
    vals = np.clip(pts.reshape(-1), 0.0, 1.0)
    prefix = "0 " + " ".join(f"{v:.6f}" for v in vals)
    if conf is None:
        return prefix
    return prefix + f" {conf:.6f}"


def match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor,
) -> torch.Tensor:
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    if pred_classes.numel() == 0 or true_classes.numel() == 0:
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
    correct_class = true_classes[:, None] == pred_classes
    iou = (iou * correct_class).cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)


def extract_component_obbs(
    mask: np.ndarray,
    score_map: np.ndarray | None,
    min_pixels: int,
    dilate_iter: int,
    score_reduction: str,
    fit_mode: str,
) -> tuple[list[np.ndarray], list[float]]:
    work = (mask > 0).astype(np.uint8)
    if dilate_iter > 0 and work.any():
        kernel = np.ones((3, 3), dtype=np.uint8)
        work = cv2.dilate(work, kernel, iterations=dilate_iter)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    polys: list[np.ndarray] = []
    confs: list[float] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        ys, xs = np.where(labels == label_idx)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        obb = obb_from_points(pts, min_pixels=min_pixels, fit_mode=fit_mode)
        if obb is None:
            continue
        polys.append(obb)
        if score_map is None:
            confs.append(1.0)
            continue
        values = score_map[labels == label_idx]
        if values.size == 0:
            confs.append(0.0)
        elif score_reduction == "mean":
            confs.append(float(values.mean()))
        else:
            confs.append(float(values.max()))
    return polys, confs


def global_obb_from_binary_mask(mask: np.ndarray, min_pixels: int = 4, fit_mode: str = "pca") -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    return obb_from_points(pts, min_pixels=min_pixels, fit_mode=fit_mode)


def trajectory_strip_obb_from_components(
    components: list[dict],
    sequence_frame_ids: list[int],
    min_pixels: int,
) -> np.ndarray | None:
    all_points = []
    centroids = []
    frame_ids = []
    per_comp = []
    for comp in components:
        ys, xs = np.where(comp["mask"] > 0)
        if len(xs) < min_pixels:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        ctr = np.asarray(comp["centroid"], dtype=np.float32)
        all_points.append(pts)
        centroids.append(ctr)
        frame_ids.append(int(comp["frame_id"]))
        per_comp.append((pts, ctr, int(comp["frame_id"])))
    if not all_points:
        return None
    points = np.concatenate(all_points, axis=0)
    if len(points) < min_pixels:
        return None

    centroid_arr = np.stack(centroids, axis=0)
    if len(centroid_arr) >= 2:
        center = centroid_arr.mean(axis=0)
        pts0 = centroid_arr - center
        cov = np.cov(pts0, rowvar=False)
        if not np.all(np.isfinite(cov)):
            return None
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argsort(eigvals)[-1]].astype(np.float32)
    else:
        center = points.mean(axis=0)
        pts0 = points - center
        cov = np.cov(pts0, rowvar=False)
        if not np.all(np.isfinite(cov)):
            return None
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argsort(eigvals)[-1]].astype(np.float32)

    frame_order = np.argsort(frame_ids)
    first_ctr = centroid_arr[frame_order[0]]
    last_ctr = centroid_arr[frame_order[-1]]
    if float(np.dot(last_ctr - first_ctr, direction)) < 0:
        direction = -direction
    direction /= float(np.linalg.norm(direction) + 1e-8)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)

    global_proj_u = points @ direction
    global_proj_v = points @ normal
    along_min = float(global_proj_u.min())
    along_max = float(global_proj_u.max())
    across_min = float(global_proj_v.min())
    across_max = float(global_proj_v.max())

    comp_rows = []
    for pts, ctr, frame_id in per_comp:
        proj_u = pts @ direction
        proj_v = pts @ normal
        ctr_u = float(np.dot(ctr, direction))
        ctr_v = float(np.dot(ctr, normal))
        comp_rows.append(
            {
                "frame_id": frame_id,
                "center_u": ctr_u,
                "center_v": ctr_v,
                "half_u": float(max(1e-6, 0.5 * (proj_u.max() - proj_u.min()))),
                "half_v": float(max(1e-6, 0.5 * (proj_v.max() - proj_v.min()))),
            }
        )
    comp_rows.sort(key=lambda row: row["frame_id"])

    if len(comp_rows) >= 2:
        speeds = []
        for prev, cur in zip(comp_rows[:-1], comp_rows[1:]):
            dt = cur["frame_id"] - prev["frame_id"]
            if dt > 0:
                speeds.append((cur["center_u"] - prev["center_u"]) / float(dt))
        step_u = float(np.median(speeds)) if speeds else 0.0
    else:
        step_u = 0.0

    seq_first = int(sequence_frame_ids[0])
    seq_last = int(sequence_frame_ids[-1])
    first_row = comp_rows[0]
    last_row = comp_rows[-1]
    missing_before = max(0, first_row["frame_id"] - seq_first)
    missing_after = max(0, seq_last - last_row["frame_id"])

    along_min = min(along_min, first_row["center_u"] - first_row["half_u"] - missing_before * step_u)
    along_max = max(along_max, last_row["center_u"] + last_row["half_u"] + missing_after * step_u)
    across_min = min(across_min, first_row["center_v"] - first_row["half_v"], last_row["center_v"] - last_row["half_v"])
    across_max = max(across_max, first_row["center_v"] + first_row["half_v"], last_row["center_v"] + last_row["half_v"])

    if (along_max - along_min) < 1e-6 or (across_max - across_min) < 1e-6:
        return None

    local = np.array(
        [[along_min, across_min], [along_max, across_min], [along_max, across_max], [along_min, across_max]],
        dtype=np.float32,
    )
    corners = local[:, :1] * direction[None, :] + local[:, 1:] * normal[None, :]
    return order_quad_clockwise(corners)


def load_metadata_by_frame(dataset_root: Path) -> dict[tuple[str, str, str], dict]:
    meta_path = dataset_root / "metadata.jsonl"
    if not meta_path.exists():
        return {}
    frame_meta = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            frame_meta[(row["split"], row["seq_id"], row["frame_id"])] = row
    return frame_meta


def load_dataset_scale(dataset_root: Path) -> tuple[float, float]:
    summary_path = dataset_root / "summary.json"
    if not summary_path.exists():
        return 1.0, 1.0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    patch_size = float(summary.get("patch_size", summary.get("save_size", 1)))
    save_size = float(summary.get("save_size", patch_size))
    if patch_size <= 0 or save_size <= 0:
        return 1.0, 1.0
    scale = save_size / patch_size
    return scale, scale


def rasterize_object_stack(
    frame_meta: dict[tuple[str, str, str], dict],
    split: str,
    seq_id: str,
    frame_ids: list[int],
    key: str,
    shape: tuple[int, int],
    scale_xy: tuple[float, float],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    sx, sy = scale_xy
    for frame_id in frame_ids:
        frame_key = (split, seq_id, f"{frame_id:05d}")
        row = frame_meta.get(frame_key)
        if row is None:
            continue
        obb = row.get(key)
        if obb is None:
            continue
        pts = np.asarray(obb, dtype=np.float32).reshape(4, 2)
        pts[:, 0] *= float(sx)
        pts[:, 1] *= float(sy)
        pts = np.round(pts).astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 1)
    return mask


def build_trajectory_instance_masks(
    frame_meta: dict[tuple[str, str, str], dict],
    split: str,
    seq_id: str,
    frame_ids: list[int],
    shape: tuple[int, int],
    scale_xy: tuple[float, float],
    dilate_iter: int,
) -> tuple[list[tuple[str, np.ndarray]], np.ndarray]:
    instances: list[tuple[str, np.ndarray]] = []
    union_mask = np.zeros(shape, dtype=np.uint8)
    for key, name in TRAJECTORY_KEYS:
        obj_mask = rasterize_object_stack(frame_meta, split, seq_id, frame_ids, key, shape, scale_xy)
        if dilate_iter > 0 and obj_mask.any():
            kernel = np.ones((3, 3), dtype=np.uint8)
            obj_mask = cv2.dilate(obj_mask, kernel, iterations=dilate_iter)
        if obj_mask.any():
            obj_mask = (obj_mask > 0).astype(np.uint8)
            instances.append((name, obj_mask))
            union_mask = np.maximum(union_mask, obj_mask)
    return instances, union_mask


def aggregate_instance_predictions(
    pred_stack_bin: np.ndarray,
    pred_score: np.ndarray,
    trajectory_masks: list[tuple[str, np.ndarray]],
    min_pixels: int,
    score_reduction: str,
    fit_mode: str,
) -> tuple[list[np.ndarray], list[float], list[dict]]:
    label_map = np.zeros(pred_stack_bin.shape, dtype=np.uint8)
    for idx, (_, inst_mask) in enumerate(trajectory_masks, start=1):
        label_map[(label_map == 0) & (inst_mask > 0)] = idx

    pred_polys: list[np.ndarray] = []
    pred_confs: list[float] = []
    instance_rows: list[dict] = []

    for idx, (name, _) in enumerate(trajectory_masks, start=1):
        inst_pred = ((pred_stack_bin > 0) & (label_map == idx)).astype(np.uint8)
        obb = global_obb_from_binary_mask(inst_pred, min_pixels=min_pixels, fit_mode=fit_mode)
        if obb is not None:
            values = pred_score[inst_pred > 0]
            if values.size == 0:
                conf = 0.0
            elif score_reduction == "mean":
                conf = float(values.mean())
            else:
                conf = float(values.max())
            pred_polys.append(obb)
            pred_confs.append(conf)
            instance_rows.append({"name": name, "kind": "trajectory", "pixels": int(inst_pred.sum())})

    fp_mask = ((pred_stack_bin > 0) & (label_map == 0)).astype(np.uint8)
    fp_polys, fp_confs = extract_component_obbs(fp_mask, pred_score, min_pixels, 0, score_reduction, fit_mode)
    pred_polys.extend(fp_polys)
    pred_confs.extend(fp_confs)
    for poly_idx, poly in enumerate(fp_polys):
        area = polygon_area(poly)
        instance_rows.append({"name": f"fp_{poly_idx:02d}", "kind": "fp", "area": float(area)})

    return pred_polys, pred_confs, instance_rows


def aggregate_prediction_components(
    pred_stack_bin: np.ndarray,
    pred_score: np.ndarray,
    min_pixels: int,
    dilate_iter: int,
    score_reduction: str,
    fit_mode: str,
) -> tuple[list[np.ndarray], list[float], list[dict]]:
    pred_polys, pred_confs = extract_component_obbs(
        pred_stack_bin,
        pred_score,
        min_pixels=min_pixels,
        dilate_iter=dilate_iter,
        score_reduction=score_reduction,
        fit_mode=fit_mode,
    )
    instance_rows = []
    for poly_idx, (poly, conf) in enumerate(zip(pred_polys, pred_confs)):
        instance_rows.append(
            {
                "name": f"pred_{poly_idx:02d}",
                "kind": "component",
                "area": float(polygon_area(poly)),
                "conf": float(conf),
            }
        )
    return pred_polys, pred_confs, instance_rows


def extract_frame_components(
    mask: np.ndarray,
    score_map: np.ndarray,
    frame_id: int,
    min_pixels: int,
) -> list[dict]:
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
        values = score_map[ys, xs]
        rows.append(
            {
                "frame_id": int(frame_id),
                "label_idx": int(label_idx),
                "area": area,
                "centroid": np.asarray(centroids[label_idx], dtype=np.float32),
                "mask": comp_mask,
                "score_max": float(values.max()) if values.size else 0.0,
                "score_mean": float(values.mean()) if values.size else 0.0,
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


def cluster_components_into_trajectories(
    frame_components: list[list[dict]],
    sequence_frame_ids: list[int],
    shape: tuple[int, int],
    max_link_distance: float,
    max_frame_gap: int,
    min_track_len: int,
    min_total_pixels: int,
    score_reduction: str,
    fit_mode: str,
) -> tuple[list[np.ndarray], list[float], list[dict], np.ndarray]:
    tracks: list[dict] = []
    next_track_id = 0

    for comps in frame_components:
        used_tracks: set[int] = set()
        comp_order = sorted(comps, key=lambda row: (-row["area"], -row["score_max"]))
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

    pred_polys: list[np.ndarray] = []
    pred_confs: list[float] = []
    instance_rows: list[dict] = []
    union_mask = np.zeros(shape, dtype=np.uint8)

    for track in tracks:
        if len(track["components"]) < min_track_len:
            continue
        track_mask = np.zeros(shape, dtype=np.uint8)
        scores = []
        total_pixels = 0
        for comp in track["components"]:
            track_mask = np.maximum(track_mask, comp["mask"])
            total_pixels += int(comp["area"])
            scores.append(comp["score_mean"] if score_reduction == "mean" else comp["score_max"])
        if total_pixels < min_total_pixels:
            continue
        if fit_mode == "trajectory_strip":
            obb = trajectory_strip_obb_from_components(
                track["components"], sequence_frame_ids=sequence_frame_ids, min_pixels=min_total_pixels
            )
            if obb is None:
                obb = global_obb_from_binary_mask(track_mask, min_pixels=min_total_pixels, fit_mode="pca")
        else:
            obb = global_obb_from_binary_mask(track_mask, min_pixels=min_total_pixels, fit_mode=fit_mode)
        if obb is None:
            continue
        conf = float(np.mean(scores)) if score_reduction == "mean" else float(np.max(scores))
        pred_polys.append(obb)
        pred_confs.append(conf)
        union_mask = np.maximum(union_mask, track_mask)
        instance_rows.append(
            {
                "name": f"traj_{track['track_id']:02d}",
                "kind": "trajectory_cluster",
                "frames": track["frame_ids"],
                "length": len(track["components"]),
                "total_pixels": total_pixels,
                "conf": conf,
            }
        )

    return pred_polys, pred_confs, instance_rows, union_mask


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
    if args.model == "DNANet":
        model = DNANet(
            num_classes=1,
            input_channels=args.t_frame,
            block=Res_CBAM_block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
            deep_supervision=args.deep_supervision,
        )
    elif args.model == "CSAUNet":
        model = CSAUNet(input_channels=args.t_frame)
    elif args.model == "DnTNet":
        model = DnTNet(seq_length=args.t_frame)
    else:
        model = MSAMNet(frame_length=args.t_frame, fusionBlock=FUSION_BLOCKS[args.fusionblock])

    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Unsupported checkpoint format: {args.weights}")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_input_tensor(
    split_dir: Path,
    seq_id: str,
    frame_id: int,
    t_frame: int,
    suffix: str,
) -> torch.Tensor:
    img_dir = split_dir / "images" / seq_id
    image_data = []

    cur_img = Image.open(img_dir / f"{frame_id:05d}{suffix}")
    image_data.append(MSODataset._normalize_frame(cur_img))

    for offset in range(1, t_frame):
        his_path = img_dir / f"{frame_id - offset:05d}{suffix}"
        if not his_path.exists():
            for i in range(1, t_frame):
                candidate = img_dir / f"{frame_id - offset + i:05d}{suffix}"
                if candidate.exists():
                    his_path = candidate
                    break
        his_img = Image.open(his_path)
        image_data.append(MSODataset._normalize_frame(his_img))

    arr = np.array(image_data[::-1], dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def load_gt_mask(split_dir: Path, seq_id: str, frame_id: int) -> np.ndarray:
    mask_path = split_dir / "masks" / seq_id / f"{frame_id:05d}.png"
    mask = np.array(Image.open(mask_path), dtype=np.uint8)
    return (mask > 0).astype(np.uint8)


def collect_frame_ids(split_dir: Path, seq_id: str, suffix: str) -> list[int]:
    img_dir = split_dir / "images" / seq_id
    frame_ids = []
    for path in sorted(img_dir.glob(f"*{suffix}")):
        frame_ids.append(int(path.stem))
    return frame_ids


def scale_to_uint8_sum(stack: np.ndarray) -> np.ndarray:
    if stack.size == 0:
        return stack.astype(np.uint8)
    vmin, vmax = np.percentile(stack, (0.01, 99.99))
    if vmax - vmin < 1e-6:
        return np.zeros_like(stack, dtype=np.uint8)
    out = np.clip(stack, vmin, vmax)
    out = (out - vmin) / (vmax - vmin) * 255.0
    return out.astype(np.uint8)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unwrap_model_output(output: torch.Tensor | list | tuple) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("Model returned an empty output sequence.")
        output = output[-1]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    return output


def main() -> None:
    args = parse_args()
    args.weights = args.weights.expanduser().resolve()
    split_dir = args.dataset_root / args.dataset_name / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    frame_meta = load_metadata_by_frame(args.dataset_root / args.dataset_name)
    if not frame_meta:
        raise FileNotFoundError(f"Missing metadata.jsonl under {args.dataset_root / args.dataset_name}")
    scale_xy = load_dataset_scale(args.dataset_root / args.dataset_name)

    want_cpu = args.device.lower().startswith("cpu")
    device = torch.device("cpu" if want_cpu or not torch.cuda.is_available() else args.device)
    model = build_model(args, device)
    seq_ids = sorted(p.name for p in (split_dir / "images").iterdir() if p.is_dir())
    if args.max_sequences > 0:
        seq_ids = seq_ids[: args.max_sequences]

    save_dir = args.save_dir / f"{args.model}_{args.split}_stackobb"
    ensure_dir(save_dir)
    if args.save_labels:
        ensure_dir(save_dir / "pred_labels")
        ensure_dir(save_dir / "gt_labels")
    if args.save_stacks:
        ensure_dir(save_dir / "pred_masks")
        ensure_dir(save_dir / "gt_masks")
        ensure_dir(save_dir / "stacked_images")

    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats_tp = []
    stats_conf = []
    stats_pred_cls = []
    stats_target_cls = []
    sequence_rows = []

    with torch.no_grad():
        for seq_id in seq_ids:
            frame_ids = collect_frame_ids(split_dir, seq_id, args.suffix)
            if not frame_ids:
                continue
            if args.stack_frames > 0:
                frame_ids = frame_ids[-args.stack_frames :]

            pred_stack = None
            pred_score = None
            img_stack = None
            frame_component_rows: list[list[dict]] = []

            for frame_id in frame_ids:
                inp = load_input_tensor(split_dir, seq_id, frame_id, args.t_frame, args.suffix).to(device)
                logits = unwrap_model_output(model(inp))
                probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
                pred_mask = (logits[0, 0].detach().cpu().numpy() > args.pred_thr).astype(np.uint8)
                img = np.array(Image.open(split_dir / "images" / seq_id / f"{frame_id:05d}{args.suffix}"), dtype=np.float32)
                frame_component_rows.append(
                    extract_frame_components(pred_mask, probs, frame_id=frame_id, min_pixels=args.component_min_pixels)
                )

                if pred_stack is None:
                    shape = pred_mask.shape
                    pred_stack = np.zeros(shape, dtype=np.uint8)
                    pred_score = np.zeros(shape, dtype=np.float32)
                    img_stack = np.zeros(shape, dtype=np.float32)

                if args.stack_mode == "sum":
                    pred_stack = np.clip(pred_stack.astype(np.uint16) + pred_mask.astype(np.uint16), 0, 255).astype(np.uint8)
                    img_stack += img
                else:
                    pred_stack = np.maximum(pred_stack, pred_mask)
                    img_stack = np.maximum(img_stack, img)
                pred_score = np.maximum(pred_score, probs)

            pred_stack_bin = (pred_stack > 0).astype(np.uint8)
            trajectory_masks, gt_stack_bin = build_trajectory_instance_masks(
                frame_meta, args.split, seq_id, frame_ids, pred_stack_bin.shape, scale_xy, args.component_dilate
            )
            gt_polys = [
                obb
                for _, inst_mask in trajectory_masks
                if (
                    obb := global_obb_from_binary_mask(
                        inst_mask, min_pixels=args.component_min_pixels, fit_mode=args.obb_fit_mode
                    )
                )
                is not None
            ]
            if args.pred_agg_mode == "gt_partition":
                pred_polys, pred_confs, instance_rows = aggregate_instance_predictions(
                    pred_stack_bin,
                    pred_score,
                    trajectory_masks,
                    args.component_min_pixels,
                    args.score_reduction,
                    args.obb_fit_mode,
                )
            elif args.pred_agg_mode == "trajectory_cluster":
                pred_polys, pred_confs, instance_rows, pred_stack_bin = cluster_components_into_trajectories(
                    frame_component_rows,
                    frame_ids,
                    pred_stack_bin.shape,
                    max_link_distance=args.cluster_max_link_distance,
                    max_frame_gap=args.cluster_max_frame_gap,
                    min_track_len=args.cluster_min_track_len,
                    min_total_pixels=args.cluster_min_total_pixels,
                    score_reduction=args.score_reduction,
                    fit_mode=args.obb_fit_mode,
                )
            else:
                pred_polys, pred_confs, instance_rows = aggregate_prediction_components(
                    pred_stack_bin,
                    pred_score,
                    args.component_min_pixels,
                    args.component_dilate,
                    args.score_reduction,
                    args.obb_fit_mode,
                )

            if pred_polys:
                pred_xyxyxyxy = torch.tensor(np.asarray(pred_polys, dtype=np.float32).reshape(-1, 8), device=device)
                pred_boxes = ops.xyxyxyxy2xywhr(pred_xyxyxyxy)
                pred_conf = np.asarray(pred_confs, dtype=np.float32)
                pred_cls = np.zeros(len(pred_polys), dtype=np.int64)
            else:
                pred_boxes = torch.zeros((0, 5), device=device, dtype=torch.float32)
                pred_conf = np.zeros((0,), dtype=np.float32)
                pred_cls = np.zeros((0,), dtype=np.int64)

            if gt_polys:
                gt_xyxyxyxy = torch.tensor(np.asarray(gt_polys, dtype=np.float32).reshape(-1, 8), device=device)
                gt_boxes = ops.xyxyxyxy2xywhr(gt_xyxyxyxy)
                gt_cls = np.zeros(len(gt_polys), dtype=np.int64)
            else:
                gt_boxes = torch.zeros((0, 5), device=device, dtype=torch.float32)
                gt_cls = np.zeros((0,), dtype=np.int64)

            pred_cls_t = torch.tensor(pred_cls, device=device)
            gt_cls_t = torch.tensor(gt_cls, device=device)
            if len(pred_polys) and len(gt_polys):
                iou = batch_probiou(gt_boxes, pred_boxes)
                correct = match_predictions(pred_cls_t, gt_cls_t, iou, iouv).cpu().numpy()
            else:
                correct = np.zeros((len(pred_polys), len(iouv)), dtype=bool)

            stats_tp.append(correct)
            stats_conf.append(pred_conf)
            stats_pred_cls.append(pred_cls)
            stats_target_cls.append(gt_cls)
            sequence_rows.append(
                {
                    "seq_id": seq_id,
                    "frames": frame_ids,
                    "num_pred": len(pred_polys),
                    "num_gt": len(gt_polys),
                    "num_gt_trajectories": len(trajectory_masks),
                    "pred_conf_mean": float(pred_conf.mean()) if pred_conf.size else 0.0,
                    "instances": instance_rows,
                }
            )

            if args.save_labels:
                pred_lines = [
                    line
                    for poly, conf in zip(pred_polys, pred_confs)
                    if (line := to_yolo_obb_line(poly, w=pred_stack_bin.shape[1], h=pred_stack_bin.shape[0], conf=conf))
                ]
                gt_lines = [
                    line
                    for poly in gt_polys
                    if (line := to_yolo_obb_line(poly, w=gt_stack_bin.shape[1], h=gt_stack_bin.shape[0]))
                ]
                (save_dir / "pred_labels" / f"{seq_id}.txt").write_text(
                    "\n".join(pred_lines) + ("\n" if pred_lines else ""), encoding="utf-8"
                )
                (save_dir / "gt_labels" / f"{seq_id}.txt").write_text(
                    "\n".join(gt_lines) + ("\n" if gt_lines else ""), encoding="utf-8"
                )
            if args.save_stacks:
                cv2.imwrite(str(save_dir / "pred_masks" / f"{seq_id}.png"), pred_stack_bin * 255)
                cv2.imwrite(str(save_dir / "gt_masks" / f"{seq_id}.png"), gt_stack_bin * 255)
                cv2.imwrite(str(save_dir / "stacked_images" / f"{seq_id}.tif"), scale_to_uint8_sum(img_stack))

    tp = np.concatenate(stats_tp, axis=0) if stats_tp else np.zeros((0, 10), dtype=bool)
    conf = np.concatenate(stats_conf, axis=0) if stats_conf else np.zeros((0,), dtype=np.float32)
    pred_cls = np.concatenate(stats_pred_cls, axis=0) if stats_pred_cls else np.zeros((0,), dtype=np.int64)
    target_cls = np.concatenate(stats_target_cls, axis=0) if stats_target_cls else np.zeros((0,), dtype=np.int64)

    if target_cls.size and conf.size:
        _, _, p, r, f1, ap, ap_class, _, _, _, _, _ = ap_per_class(
            tp, conf, pred_cls, target_cls, plot=False, names={0: "line"}
        )
        mp = float(p.mean()) if p.size else 0.0
        mr = float(r.mean()) if r.size else 0.0
        map50 = float(ap[:, 0].mean()) if ap.size else 0.0
        map50_95 = float(ap.mean()) if ap.size else 0.0
        f1_mean = float(f1.mean()) if f1.size else 0.0
        ap_class = ap_class.tolist()
    else:
        mp = 0.0
        mr = 0.0
        map50 = 0.0
        map50_95 = 0.0
        f1_mean = 0.0
        ap_class = []

    summary = {
        "model": args.model,
        "weights": str(args.weights),
        "dataset_root": str(args.dataset_root / args.dataset_name),
        "split": args.split,
        "t_frame": args.t_frame,
        "stack_frames": args.stack_frames if args.stack_frames > 0 else "all",
        "stack_mode": args.stack_mode,
        "pred_thr": args.pred_thr,
        "obb_fit_mode": args.obb_fit_mode,
        "component_min_pixels": args.component_min_pixels,
        "component_dilate": args.component_dilate,
        "pred_agg_mode": args.pred_agg_mode,
        "cluster_max_link_distance": args.cluster_max_link_distance,
        "cluster_max_frame_gap": args.cluster_max_frame_gap,
        "cluster_min_track_len": args.cluster_min_track_len,
        "cluster_min_total_pixels": args.cluster_min_total_pixels,
        "score_reduction": args.score_reduction,
        "gt_definition": "trajectory_instance_obb_aggregated_from_metadata",
        "pred_definition": (
            "prediction_stack_connected_components"
            if args.pred_agg_mode == "components"
            else (
                "prediction_frame_components_clustered_into_trajectories"
                if args.pred_agg_mode == "trajectory_cluster"
                else "prediction_stack_partitioned_by_gt_trajectory_regions_plus_fp_residuals"
            )
        ),
        "num_sequences": len(sequence_rows),
        "num_predictions": int(sum(len(x) for x in stats_conf)),
        "num_targets": int(sum(len(x) for x in stats_target_cls)),
        "precision": mp,
        "recall": mr,
        "f1": f1_mean,
        "map50": map50,
        "map50_95": map50_95,
        "ap_class_index": ap_class,
    }

    (save_dir / "sequence_metrics.json").write_text(json.dumps(sequence_rows, indent=2), encoding="utf-8")
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
