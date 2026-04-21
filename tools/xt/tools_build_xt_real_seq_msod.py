from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


DEFAULT_SOURCE_DIRS = [
    "/mnt/e/Imgs/03-TJStars/150ms-2k2k/decode/",
    "/mnt/e/Imgs/03-TJStars/3000ms-4k4k/decode/",
    "/mnt/e/Imgs/03-TJStars/TJ2/decode/",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a test-only x-sequence MSOD-style dataset from real XT decode folders."
    )
    parser.add_argument("--out-root", type=Path, default=Path("dataset/xt_real_seq_msod_x"))
    parser.add_argument("--dataset-name", type=str, default="REAL_MSOD_RAWBG")
    parser.add_argument("--source-dirs", nargs="+", default=DEFAULT_SOURCE_DIRS)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--save-size", type=int, default=512)
    parser.add_argument("--index-history", type=int, default=3)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--register", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--register-downscale", type=float, default=0.25)
    parser.add_argument("--register-border-fill", choices=["min", "median"], default="min")
    parser.add_argument("--suppress-static", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--static-mask-percentile", type=float, default=99.7)
    parser.add_argument("--static-mask-dilate", type=int, default=1)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.90)
    parser.add_argument("--write-zero-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def prepare_out_root(dataset_dir: Path, keep_existing: bool, write_zero_masks: bool) -> None:
    if dataset_dir.exists() and not keep_existing:
        shutil.rmtree(dataset_dir)
    (dataset_dir / "test" / "images").mkdir(parents=True, exist_ok=True)
    if write_zero_masks:
        (dataset_dir / "test" / "masks").mkdir(parents=True, exist_ok=True)


def load_real_img(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    if not np.isfinite(img).all():
        finite = np.isfinite(img)
        fill = float(np.median(img[finite])) if finite.any() else 0.0
        img = np.where(finite, img, fill)
    return img.astype(np.float32)


def resize_for_registration(img: np.ndarray, downscale: float) -> np.ndarray:
    if downscale >= 1.0:
        out = img
    else:
        h, w = img.shape
        new_w = max(64, int(round(w * downscale)))
        new_h = max(64, int(round(h * downscale)))
        out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    vmin, vmax = np.percentile(out, (1.0, 99.0))
    if vmax - vmin < 1e-6:
        return np.zeros_like(out, dtype=np.float32)
    out = np.clip(out, vmin, vmax)
    out = (out - vmin) / (vmax - vmin)
    return out.astype(np.float32)


def estimate_translation(ref: np.ndarray, moving: np.ndarray, downscale: float) -> tuple[tuple[float, float], float]:
    ref_small = resize_for_registration(ref, downscale)
    mov_small = resize_for_registration(moving, downscale)
    shift_small, response = cv2.phaseCorrelate(ref_small, mov_small)
    if downscale <= 0:
        downscale = 1.0
    shift = (float(shift_small[0] / downscale), float(shift_small[1] / downscale))
    return shift, float(response)


def align_to_reference(
    img: np.ndarray,
    shift_xy: tuple[float, float],
    border_fill: str,
) -> tuple[np.ndarray, np.ndarray]:
    dx, dy = shift_xy
    fill = float(np.min(img)) if border_fill == "min" else float(np.median(img))
    h, w = img.shape
    matrix = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
    aligned = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    valid = cv2.warpAffine(
        np.ones((h, w), dtype=np.uint8),
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned.astype(np.float32), valid.astype(bool)


def build_patch_records(source_dirs: list[str], patch_size: int, stride: int) -> list[dict]:
    records: list[dict] = []
    for source_index, source_dir in enumerate(source_dirs):
        paths = sorted(glob.glob(str(Path(source_dir) / "*.tif")))
        if not paths:
            continue
        sample = load_real_img(paths[0])
        h, w = sample.shape
        nw = (w - patch_size) // stride + 1
        nh = (h - patch_size) // stride + 1
        for iy in range(nh):
            for ix in range(nw):
                x1, y1 = ix * stride, iy * stride
                records.append(
                    {
                        "source_index": source_index,
                        "source_dir": source_dir,
                        "x1": x1,
                        "y1": y1,
                        "x2": x1 + patch_size,
                        "y2": y1 + patch_size,
                    }
                )
    return records


def scale_to_uint8(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    img = np.clip(img, vmin, vmax)
    out = (img - vmin) / (vmax - vmin + 1e-8) * 255.0
    return out.astype(np.uint8)


def build_static_mask(median_patch: np.ndarray, percentile: float, dilate_iter: int) -> np.ndarray:
    if median_patch.size == 0:
        return np.zeros_like(median_patch, dtype=bool)
    thr = float(np.percentile(median_patch, percentile))
    mask = (median_patch >= thr).astype(np.uint8)
    if dilate_iter > 0 and mask.any():
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return mask.astype(bool)


def write_index_file(path: Path, items: list[str]) -> None:
    path.write_text("".join(f"{item}\n" for item in items), encoding="utf-8")


def load_and_align_source(
    source_dir: str,
    register: bool,
    register_downscale: float,
    register_border_fill: str,
) -> dict:
    paths = sorted(glob.glob(str(Path(source_dir) / "*.tif")))
    if not paths:
        raise FileNotFoundError(f"No TIFF files found in {source_dir}")

    ref = load_real_img(paths[0])
    aligned_frames: list[np.ndarray] = [ref]
    valid_masks: list[np.ndarray] = [np.ones_like(ref, dtype=bool)]
    shifts: list[tuple[float, float]] = [(0.0, 0.0)]
    responses: list[float] = [1.0]

    for path in tqdm(paths[1:], desc=f"Align {Path(source_dir).parent.name}"):
        img = load_real_img(path)
        if register:
            shift_xy, response = estimate_translation(ref, img, register_downscale)
            aligned, valid = align_to_reference(img, shift_xy, register_border_fill)
        else:
            shift_xy, response = (0.0, 0.0), 1.0
            aligned, valid = img, np.ones_like(img, dtype=bool)
        aligned_frames.append(aligned)
        valid_masks.append(valid)
        shifts.append(shift_xy)
        responses.append(response)

    return {
        "paths": paths,
        "aligned_frames": aligned_frames,
        "valid_masks": valid_masks,
        "shifts": shifts,
        "responses": responses,
    }


def build_dataset(args: argparse.Namespace) -> dict:
    dataset_dir = (args.out_root / args.dataset_name).expanduser().resolve()
    prepare_out_root(dataset_dir, keep_existing=args.keep_existing, write_zero_masks=args.write_zero_masks)

    patch_records = build_patch_records(args.source_dirs, args.patch_size, args.stride)
    if not patch_records:
        raise RuntimeError("No patch records found.")
    if args.max_sequences > 0:
        patch_records = patch_records[: args.max_sequences]

    source_cache = {}
    metadata_path = dataset_dir / "metadata.jsonl"
    test_index: list[str] = []
    seq_rows: list[dict] = []
    seq_global = 1

    with metadata_path.open("w", encoding="utf-8") as meta_file:
        for row in tqdm(patch_records, desc="Build real test sequences"):
            source_dir = row["source_dir"]
            if source_dir not in source_cache:
                source_cache[source_dir] = load_and_align_source(
                    source_dir,
                    register=args.register,
                    register_downscale=args.register_downscale,
                    register_border_fill=args.register_border_fill,
                )
            cached = source_cache[source_dir]

            frame_patches = []
            patch_valids = []
            for frame_img, valid_mask in zip(cached["aligned_frames"], cached["valid_masks"]):
                patch = frame_img[row["y1"] : row["y2"], row["x1"] : row["x2"]].astype(np.float32)
                valid = valid_mask[row["y1"] : row["y2"], row["x1"] : row["x2"]]
                frame_patches.append(patch)
                patch_valids.append(valid)

            overlap_valid = np.logical_and.reduce(np.stack(patch_valids, axis=0), axis=0)
            overlap_ratio = float(overlap_valid.mean())
            if overlap_ratio < float(args.min_overlap_ratio):
                continue

            stack = np.stack(frame_patches, axis=0)
            median_patch = np.median(stack, axis=0)
            static_mask = (
                build_static_mask(median_patch, args.static_mask_percentile, args.static_mask_dilate)
                if args.suppress_static
                else np.zeros_like(median_patch, dtype=bool)
            )

            if overlap_valid.any():
                fill_value = float(np.percentile(stack[:, overlap_valid], 1.0))
            else:
                fill_value = float(np.percentile(stack, 1.0))

            processed_frames: list[np.ndarray] = []
            for patch in frame_patches:
                cur = patch.copy()
                cur[~overlap_valid] = fill_value
                if static_mask.any():
                    cur[static_mask] = fill_value
                processed_frames.append(cur)

            proc_stack = np.stack(processed_frames, axis=0)
            if overlap_valid.any():
                valid_vals = proc_stack[:, overlap_valid]
            else:
                valid_vals = proc_stack.reshape(-1)
            seq_vmin, seq_vmax = np.percentile(valid_vals, (1.0, 99.0))

            seq_id = f"{seq_global:04d}"
            img_dir = dataset_dir / "test" / "images" / seq_id
            mask_dir = dataset_dir / "test" / "masks" / seq_id
            img_dir.mkdir(parents=True, exist_ok=True)
            if args.write_zero_masks:
                mask_dir.mkdir(parents=True, exist_ok=True)

            source_tag = Path(source_dir).parent.parent.name
            num_frames = len(processed_frames)
            for frame_idx, (frame_path, shift_xy, response, patch) in enumerate(
                zip(cached["paths"], cached["shifts"], cached["responses"], processed_frames),
                start=1,
            ):
                frame_u8 = scale_to_uint8(patch, float(seq_vmin), float(seq_vmax))
                if args.save_size != args.patch_size:
                    frame_u8 = cv2.resize(frame_u8, (args.save_size, args.save_size), interpolation=cv2.INTER_AREA)

                frame_name = f"{frame_idx:05d}"
                cv2.imwrite(str(img_dir / f"{frame_name}.tif"), frame_u8)
                if args.write_zero_masks:
                    zeros = np.zeros_like(frame_u8, dtype=np.uint8)
                    cv2.imwrite(str(mask_dir / f"{frame_name}.png"), zeros)

                if frame_idx >= args.index_history:
                    test_index.append(f"{seq_id}/{frame_name}")

                meta = {
                    "split": "test",
                    "seq_id": seq_id,
                    "frame_id": frame_name,
                    "source_dir": source_dir,
                    "source_tag": source_tag,
                    "source_path": frame_path,
                    "crop": [row["x1"], row["y1"], row["x2"], row["y2"]],
                    "shift_xy": [round(float(shift_xy[0]), 4), round(float(shift_xy[1]), 4)],
                    "register_response": round(float(response), 6),
                    "overlap_ratio": round(overlap_ratio, 6),
                    "static_mask_ratio": round(float(static_mask.mean()), 6),
                }
                meta_file.write(json.dumps(meta, ensure_ascii=True) + "\n")

            seq_rows.append(
                {
                    "seq_id": seq_id,
                    "source_dir": source_dir,
                    "source_tag": source_tag,
                    "crop": [row["x1"], row["y1"], row["x2"], row["y2"]],
                    "num_frames": num_frames,
                    "overlap_ratio": overlap_ratio,
                    "static_mask_ratio": float(static_mask.mean()),
                }
            )
            seq_global += 1

    write_index_file(dataset_dir / "test.txt", test_index)
    summary = {
        "dataset_dir": str(dataset_dir),
        "split": "test",
        "source_dirs": args.source_dirs,
        "sequences": seq_global - 1,
        "frames_total": int(sum(row["num_frames"] for row in seq_rows)),
        "index_history": args.index_history,
        "test_indices": len(test_index),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "save_size": args.save_size,
        "register": bool(args.register),
        "register_downscale": args.register_downscale,
        "suppress_static": bool(args.suppress_static),
        "write_zero_masks": bool(args.write_zero_masks),
        "min_overlap_ratio": args.min_overlap_ratio,
    }
    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (dataset_dir / "sequences.json").write_text(json.dumps(seq_rows, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
