from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO-OBB dataset by stacking frame sequences from the MSOD dataset."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/xt_seq_msod_x/MSOD_RAWBG"))
    parser.add_argument("--out-root", type=Path, default=Path("dataset/obb_xt_seq_msod_stack_rawbg"))
    parser.add_argument("--image-suffix", type=str, default=".tif")
    parser.add_argument("--stack-frames", type=int, default=0, help="0 means all frames in each sequence.")
    parser.add_argument("--image-stack-mode", choices=["sum", "max", "mean"], default="sum")
    parser.add_argument("--mask-stack-mode", choices=["max", "sum"], default="max")
    parser.add_argument("--component-min-pixels", type=int, default=4)
    parser.add_argument("--component-dilate", type=int, default=1)
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--save-stacked-masks", action="store_true")
    return parser.parse_args()


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


def extract_component_obbs(mask: np.ndarray, min_pixels: int, dilate_iter: int) -> list[np.ndarray]:
    work = (mask > 0).astype(np.uint8)
    if dilate_iter > 0 and work.any():
        kernel = np.ones((3, 3), dtype=np.uint8)
        work = cv2.dilate(work, kernel, iterations=dilate_iter)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    polys: list[np.ndarray] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        ys, xs = np.where(labels == label_idx)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        obb = obb_from_points(pts, min_pixels=min_pixels)
        if obb is not None:
            polys.append(obb)
    return polys


def global_obb_from_binary_mask(mask: np.ndarray, min_pixels: int = 4) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    return obb_from_points(pts, min_pixels=min_pixels)


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


def collect_frame_ids(img_dir: Path, suffix: str) -> list[int]:
    return sorted(int(p.stem) for p in img_dir.glob(f"*{suffix}"))


def stack_image(frames: list[np.ndarray], mode: str) -> np.ndarray:
    stack = np.stack(frames, axis=0).astype(np.float32)
    if mode == "max":
        img = stack.max(axis=0)
    elif mode == "mean":
        img = stack.mean(axis=0)
    else:
        img = stack.sum(axis=0)
    vmin, vmax = np.percentile(img, (1.0, 99.0))
    if vmax - vmin < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8) * 255.0
    return img.astype(np.uint8)


def stack_mask(masks: list[np.ndarray], mode: str) -> np.ndarray:
    stack = np.stack(masks, axis=0).astype(np.uint8)
    if mode == "sum":
        return (stack.sum(axis=0) > 0).astype(np.uint8)
    return stack.max(axis=0).astype(np.uint8)


def prepare_out_root(root: Path, keep_existing: bool, save_stacked_masks: bool) -> None:
    if root.exists() and not keep_existing:
        shutil.rmtree(root)
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        if save_stacked_masks:
            (root / "masks" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(out_root: Path) -> None:
    text = "\n".join(
        [
            f"path: {out_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: line",
            "",
        ]
    )
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


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


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")

    prepare_out_root(out_root, keep_existing=args.keep_existing, save_stacked_masks=args.save_stacked_masks)
    frame_meta = load_metadata_by_frame(dataset_root)
    scale_xy = load_dataset_scale(dataset_root)
    metadata_rows = []
    summary = {"splits": {}, "image_stack_mode": args.image_stack_mode, "mask_stack_mode": args.mask_stack_mode}

    for split in ["train", "val", "test"]:
        split_root = dataset_root / split
        img_split_dir = split_root / "images"
        mask_split_dir = split_root / "masks"
        seq_ids = sorted(p.name for p in img_split_dir.iterdir() if p.is_dir())
        count_images = 0
        count_boxes = 0

        for seq_id in tqdm(seq_ids, desc=f"Build {split}"):
            seq_img_dir = img_split_dir / seq_id
            seq_mask_dir = mask_split_dir / seq_id
            frame_ids = collect_frame_ids(seq_img_dir, args.image_suffix)
            if not frame_ids:
                continue
            if args.stack_frames > 0:
                frame_ids = frame_ids[-args.stack_frames :]

            frames = []
            masks = []
            for frame_id in frame_ids:
                frame_name = f"{frame_id:05d}"
                frames.append(np.array(Image.open(seq_img_dir / f"{frame_name}{args.image_suffix}"), dtype=np.float32))
                mask = np.array(Image.open(seq_mask_dir / f"{frame_name}.png"), dtype=np.uint8)
                masks.append((mask > 0).astype(np.uint8))

            stacked_img = stack_image(frames, mode=args.image_stack_mode)
            stacked_mask = stack_mask(masks, mode=args.mask_stack_mode)
            if frame_meta:
                object_masks = [
                    rasterize_object_stack(frame_meta, split, seq_id, frame_ids, "line_obb", stacked_mask.shape, scale_xy),
                    rasterize_object_stack(frame_meta, split, seq_id, frame_ids, "pt_obb", stacked_mask.shape, scale_xy),
                ]
                obbs = []
                for obj_mask in object_masks:
                    if args.component_dilate > 0 and obj_mask.any():
                        obj_mask = cv2.dilate(
                            obj_mask, np.ones((3, 3), dtype=np.uint8), iterations=args.component_dilate
                        )
                    obb = global_obb_from_binary_mask(obj_mask, min_pixels=args.component_min_pixels)
                    if obb is not None:
                        obbs.append(obb)
            else:
                obbs = extract_component_obbs(
                    stacked_mask, min_pixels=args.component_min_pixels, dilate_iter=args.component_dilate
                )
            label_lines = [
                line
                for obb in obbs
                if (line := to_yolo_obb_line(obb, w=stacked_img.shape[1], h=stacked_img.shape[0]))
            ]

            img_out = out_root / "images" / split / f"{seq_id}.tif"
            lbl_out = out_root / "labels" / split / f"{seq_id}.txt"
            cv2.imwrite(str(img_out), stacked_img)
            lbl_out.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            if args.save_stacked_masks:
                cv2.imwrite(str(out_root / "masks" / split / f"{seq_id}.png"), stacked_mask.astype(np.uint8) * 255)

            metadata_rows.append(
                {
                    "split": split,
                    "seq_id": seq_id,
                    "frame_ids": frame_ids,
                    "num_boxes": len(label_lines),
                    "image_path": str(img_out),
                    "label_path": str(lbl_out),
                }
            )
            count_images += 1
            count_boxes += len(label_lines)

        summary["splits"][split] = {"images": count_images, "boxes": count_boxes}

    write_data_yaml(out_root)
    (out_root / "metadata.json").write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out_root": str(out_root), **summary}, indent=2))


if __name__ == "__main__":
    main()
