from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import sep


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 3-sequence KBS finetune datasets for compare models and XT_sc detector.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+")
    parser.add_argument("--train-json", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "train.json")
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-ids", nargs="*", default=None, help="Optional explicit KBS ids such as 050 061 015.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-seq-count", type=int, default=0, help="0 means derive from --val-ratio.")
    parser.add_argument("--compare-out-root", type=Path, default=REPO_ROOT / "dataset" / "xt_seq_msod_x" / "KBS_FT3_REAL_P99")
    parser.add_argument("--ours-out-root", type=Path, default=REPO_ROOT / "dataset" / "obb_kbs_ft3_notebook_aligned")
    parser.add_argument("--compare-save-size", type=int, default=512)
    parser.add_argument("--frame-percentile", type=float, nargs=2, default=(1.0, 99.0))
    parser.add_argument("--x-percentile", type=float, nargs=2, default=(0.01, 99.99))
    parser.add_argument("--star-threshold", type=float, default=1.5)
    parser.add_argument("--track-max-link-distance", type=float, default=80.0)
    parser.add_argument("--track-max-frame-gap", type=int, default=2)
    parser.add_argument("--track-min-len", type=int, default=2)
    parser.add_argument("--track-min-area", type=int, default=6)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def percentile_rescale(img: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    vmin, vmax = np.percentile(img, (pmin, pmax))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8) * 255.0
    return img.astype(np.uint8)


def trunc_img(img: np.ndarray, ratio: tuple[float, float]) -> np.ndarray:
    return percentile_rescale(img, ratio[0], ratio[1])


def encode_xt_sc(x_u8: np.ndarray, t_u8: np.ndarray) -> np.ndarray:
    tau = t_u8.astype(np.float32) / 255.0
    sin8 = ((np.sin(np.pi * tau) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    cos8 = ((np.cos(np.pi * tau) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return np.stack([x_u8, sin8, cos8], axis=-1)


def load_seq_ids(args: argparse.Namespace) -> list[str]:
    if args.seq_ids:
        return [f"{int(x):03d}" for x in args.seq_ids]
    ids = json.loads(args.train_json.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    picked = rng.sample(ids, args.sample_count)
    return [f"{int(x):03d}" for x in picked]


def split_seq_ids(seq_ids: list[str], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    seq_ids = [f"{int(x):03d}" for x in seq_ids]
    if len(seq_ids) < 2:
        raise ValueError("Need at least 2 sequences to build a non-leaking train/val split.")
    if int(args.val_seq_count) > 0:
        val_count = int(args.val_seq_count)
    else:
        val_count = max(1, int(round(len(seq_ids) * float(args.val_ratio))))
    val_count = min(val_count, len(seq_ids) - 1)
    train_ids = seq_ids[:-val_count]
    val_ids = seq_ids[-val_count:]
    return train_ids, val_ids


def load_frames_and_masks(dataset_root: Path, seq_id: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    frame_paths = sorted((dataset_root / f"{seq_id}_img").glob("*.png"))
    mask_paths = sorted((dataset_root / f"{seq_id}_gt").glob("*.png"))
    if not frame_paths or len(frame_paths) != len(mask_paths):
        raise FileNotFoundError(f"Broken sequence {seq_id}")
    frames = []
    masks = []
    for f_path, m_path in zip(frame_paths, mask_paths):
        img = cv2.imread(str(f_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(m_path), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            raise FileNotFoundError(f"Missing frame or mask in {seq_id}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(img.astype(np.float32))
        masks.append((mask > 0).astype(np.uint8))
    return frames, masks


def order_quad_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(ang)]
    i0 = np.argmin(pts[:, 0] + pts[:, 1])
    return np.roll(pts, -i0, axis=0)


def polygon_area(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def obb_from_points(pts: np.ndarray, min_pixels: int = 4) -> np.ndarray | None:
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


def to_yolo_obb_line(obb: np.ndarray, w: int, h: int) -> str | None:
    pts = np.asarray(obb, dtype=np.float32).reshape(4, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, w)
    pts[:, 1] = np.clip(pts[:, 1], 0, h)
    pts = order_quad_clockwise(pts)
    if polygon_area(pts) < 2.0:
        return None
    pts[:, 0] /= float(w)
    pts[:, 1] /= float(h)
    vals = np.clip(pts.reshape(-1), 0.0, 1.0)
    return "0 " + " ".join(f"{v:.6f}" for v in vals)


def extract_frame_components(mask: np.ndarray, frame_id: int, min_area: int) -> list[dict]:
    work = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(work, connectivity=8)
    rows: list[dict] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
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


def _link_cost(track: dict, comp: dict) -> float:
    dt = int(comp["frame_id"]) - int(track["last_frame"])
    if dt <= 0:
        return float("inf")
    last_centroid = np.asarray(track["last_centroid"], dtype=np.float32)
    dist = float(np.linalg.norm(np.asarray(comp["centroid"], dtype=np.float32) - last_centroid))
    return dist / float(dt)


def cluster_gt_components(
    frame_masks: list[np.ndarray],
    max_link_distance: float,
    max_frame_gap: int,
    min_track_len: int,
    min_area: int,
) -> list[np.ndarray]:
    per_frame = [extract_frame_components(mask, idx, min_area=min_area) for idx, mask in enumerate(frame_masks)]
    active_tracks: list[dict] = []
    finished_tracks: list[dict] = []

    for comp_rows in per_frame:
        comp_rows = sorted(comp_rows, key=lambda row: row["area"], reverse=True)
        taken_tracks: set[int] = set()
        for comp in comp_rows:
            best_idx = None
            best_cost = float("inf")
            for track_idx, track in enumerate(active_tracks):
                if track_idx in taken_tracks:
                    continue
                dt = int(comp["frame_id"]) - int(track["last_frame"])
                if dt <= 0 or dt > max_frame_gap:
                    continue
                cost = _link_cost(track, comp)
                if cost <= max_link_distance and cost < best_cost:
                    best_cost = cost
                    best_idx = track_idx
            if best_idx is None:
                active_tracks.append(
                    {
                        "components": [comp],
                        "last_frame": int(comp["frame_id"]),
                        "last_centroid": np.asarray(comp["centroid"], dtype=np.float32),
                    }
                )
                taken_tracks.add(len(active_tracks) - 1)
            else:
                track = active_tracks[best_idx]
                track["components"].append(comp)
                track["last_frame"] = int(comp["frame_id"])
                track["last_centroid"] = np.asarray(comp["centroid"], dtype=np.float32)
                taken_tracks.add(best_idx)

        still_active = []
        current_frame = max([c["frame_id"] for c in comp_rows], default=None)
        for track in active_tracks:
            if current_frame is None or (current_frame - int(track["last_frame"])) < max_frame_gap:
                still_active.append(track)
            else:
                finished_tracks.append(track)
        active_tracks = still_active

    finished_tracks.extend(active_tracks)

    union_masks: list[np.ndarray] = []
    for track in finished_tracks:
        if len(track["components"]) < min_track_len:
            continue
        union = np.zeros_like(frame_masks[0], dtype=np.uint8)
        total_area = 0
        for comp in track["components"]:
            union = np.maximum(union, comp["mask"])
            total_area += int(comp["area"])
        if total_area < min_area:
            continue
        union_masks.append(union)
    return union_masks


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


def prepare_dir(root: Path, keep_existing: bool) -> None:
    if root.exists() and not keep_existing:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def write_compare_dataset(args: argparse.Namespace, train_seq_ids: list[str], val_seq_ids: list[str]) -> dict:
    root = args.compare_out_root.resolve()
    prepare_dir(root, keep_existing=args.keep_existing)
    split_records: dict[str, list[str]] = {"train": [], "val": []}
    meta = {
        "train_seq_ids": train_seq_ids,
        "val_seq_ids": val_seq_ids,
        "save_size": args.compare_save_size,
        "frame_percentile": list(args.frame_percentile),
    }

    for split in ["train", "val"]:
        for sub in ["images", "masks"]:
            (root / split / sub).mkdir(parents=True, exist_ok=True)

    for split, split_seq_ids in [("train", train_seq_ids), ("val", val_seq_ids)]:
        for seq_id in split_seq_ids:
            frames, masks = load_frames_and_masks(args.dataset_root, seq_id)
            img_dir = root / split / "images" / seq_id
            mask_dir = root / split / "masks" / seq_id
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx, (frame, mask) in enumerate(zip(frames, masks)):
                x_u8 = percentile_rescale(frame, args.frame_percentile[0], args.frame_percentile[1])
                x_u8 = cv2.resize(x_u8, (args.compare_save_size, args.compare_save_size), interpolation=cv2.INTER_AREA)
                m_u8 = cv2.resize((mask * 255).astype(np.uint8), (args.compare_save_size, args.compare_save_size), interpolation=cv2.INTER_NEAREST)
                frame_name = f"{frame_idx:05d}"
                cv2.imwrite(str(img_dir / f"{frame_name}.png"), x_u8)
                cv2.imwrite(str(mask_dir / f"{frame_name}.png"), m_u8)
                split_records[split].append(f"{seq_id}/{frame_name}")

    for split, items in split_records.items():
        (root / f"{split}.txt").write_text("".join(f"{x}\n" for x in items), encoding="utf-8")

    (root / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"dataset_dir": str(root), "train_items": len(split_records["train"]), "val_items": len(split_records["val"])}


def write_ours_dataset(args: argparse.Namespace, train_seq_ids: list[str], val_seq_ids: list[str]) -> dict:
    root = args.ours_out_root.resolve()
    prepare_dir(root, keep_existing=args.keep_existing)
    for split in ["train", "val"]:
        (root / "images" / f"{split}_xt_sc").mkdir(parents=True, exist_ok=True)
        (root / "labels" / f"{split}_xt_sc").mkdir(parents=True, exist_ok=True)
    for aux in ["x", "t", "star_mask", "gt_union"]:
        for split in ["train", "val"]:
            (root / aux / split).mkdir(parents=True, exist_ok=True)

    meta = []
    for split, split_seq_ids in [("train", train_seq_ids), ("val", val_seq_ids)]:
        for seq_id in split_seq_ids:
            frames, masks = load_frames_and_masks(args.dataset_root, seq_id)
            x_u8, t_u8, xt_sc, star_mask = build_notebook_aligned_xt(
                frames,
                x_percentile=tuple(args.x_percentile),
                star_threshold=float(args.star_threshold),
            )
            tracks = cluster_gt_components(
                masks,
                max_link_distance=float(args.track_max_link_distance),
                max_frame_gap=int(args.track_max_frame_gap),
                min_track_len=int(args.track_min_len),
                min_area=int(args.track_min_area),
            )
            gt_union = np.logical_or.reduce(np.stack(masks, axis=0), axis=0).astype(np.uint8)

            h, w = x_u8.shape[:2]
            lines = []
            for track_mask in tracks:
                ys, xs = np.where(track_mask > 0)
                if len(xs) < int(args.track_min_area):
                    continue
                obb = obb_from_points(np.stack([xs, ys], axis=1), min_pixels=int(args.track_min_area))
                if obb is None:
                    continue
                line = to_yolo_obb_line(obb, w=w, h=h)
                if line is not None and line not in lines:
                    lines.append(line)
            if not lines:
                ys, xs = np.where(gt_union > 0)
                if len(xs) >= int(args.track_min_area):
                    obb = obb_from_points(np.stack([xs, ys], axis=1), min_pixels=int(args.track_min_area))
                    if obb is not None:
                        line = to_yolo_obb_line(obb, w=w, h=h)
                        if line is not None:
                            lines.append(line)

            cv2.imwrite(str(root / "images" / f"{split}_xt_sc" / f"{seq_id}.png"), xt_sc)
            cv2.imwrite(str(root / "x" / split / f"{seq_id}.png"), x_u8)
            cv2.imwrite(str(root / "t" / split / f"{seq_id}.png"), t_u8)
            cv2.imwrite(str(root / "star_mask" / split / f"{seq_id}.png"), star_mask * 255)
            cv2.imwrite(str(root / "gt_union" / split / f"{seq_id}.png"), gt_union * 255)
            (root / "labels" / f"{split}_xt_sc" / f"{seq_id}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "",
                encoding="utf-8",
            )
            meta.append({"split": split, "seq_id": seq_id, "num_labels": len(lines), "num_tracks": len(tracks)})

    data_yaml = "\n".join(
        [
            f"path: {root.as_posix()}",
            "train: images/train_xt_sc",
            "val: images/val_xt_sc",
            "names:",
            "  0: line",
            "",
        ]
    )
    (root / "data_xt_sc.yaml").write_text(data_yaml, encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps(
            {
                "selected_seq_ids": train_seq_ids + val_seq_ids,
                "train_seq_ids": train_seq_ids,
                "val_seq_ids": val_seq_ids,
                "x_percentile": list(args.x_percentile),
                "star_threshold": float(args.star_threshold),
                "track_max_link_distance": float(args.track_max_link_distance),
                "track_max_frame_gap": int(args.track_max_frame_gap),
                "track_min_len": int(args.track_min_len),
                "track_min_area": int(args.track_min_area),
                "items": meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"dataset_dir": str(root), "data_yaml": str(root / "data_xt_sc.yaml"), "items": meta}


def main() -> None:
    args = parse_args()
    seq_ids = load_seq_ids(args)
    train_seq_ids, val_seq_ids = split_seq_ids(seq_ids, args)
    compare_summary = write_compare_dataset(args, train_seq_ids, val_seq_ids)
    ours_summary = write_ours_dataset(args, train_seq_ids, val_seq_ids)
    out = {
        "selected_seq_ids": seq_ids,
        "train_seq_ids": train_seq_ids,
        "val_seq_ids": val_seq_ids,
        "seed": int(args.seed),
        "compare": compare_summary,
        "ours": ours_summary,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
