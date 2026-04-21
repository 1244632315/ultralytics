from __future__ import annotations

import argparse
import glob
import json
import random
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


def _gaussian_psf(radius_px: float, size: int | None = None) -> np.ndarray:
    sigma = max(0.2, float(radius_px))
    if size is None:
        size = int(np.ceil(6 * sigma)) | 1
    c = size // 2
    y, x = np.mgrid[-c : c + 1, -c : c + 1]
    psf = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    psf /= psf.sum()
    return psf.astype(np.float32)


def _motion_kernel(length_px: float, angle_rad: float, width: float = 1.0) -> np.ndarray:
    length_px = max(1.0, float(length_px))
    size = int(np.ceil(length_px)) | 1
    c = size // 2
    y, x = np.mgrid[-c : c + 1, -c : c + 1]
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    u = ca * x + sa * y
    v = -sa * x + ca * y
    line = (np.abs(u) <= (length_px / 2)).astype(np.float32)
    blur = np.exp(-(v * v) / (2 * (max(0.3, width) ** 2)))
    ker = line * blur
    ker /= ker.sum()
    return ker.astype(np.float32)


def _fft_convolve_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ha, wa = a.shape
    hb, wb = b.shape
    h, w = ha + hb - 1, wa + wb - 1
    fa = np.fft.rfft2(a, s=(h, w))
    fb = np.fft.rfft2(b, s=(h, w))
    out = np.fft.irfft2(fa * fb, s=(h, w))
    return out.astype(np.float32)


def _place_kernel_add(img: np.ndarray, ker: np.ndarray, y: float, x: float, gain: float) -> None:
    h, w = img.shape
    kh, kw = ker.shape
    cy, cx = kh // 2, kw // 2
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    dy = y - y0
    dx = x - x0
    weights = [
        (y0, x0, (1 - dy) * (1 - dx)),
        (y0 + 1, x0, dy * (1 - dx)),
        (y0, x0 + 1, (1 - dy) * dx),
        (y0 + 1, x0 + 1, dy * dx),
    ]
    for yy, xx, ww in weights:
        top = yy - cy
        left = xx - cx
        r0 = max(0, top)
        c0 = max(0, left)
        r1 = min(h, top + kh)
        c1 = min(w, left + kw)
        if r0 >= r1 or c0 >= c1:
            continue
        kr0 = r0 - top
        kc0 = c0 - left
        kr1 = kr0 + (r1 - r0)
        kc1 = kc0 + (c1 - c0)
        img[r0:r1, c0:c1] += (gain * ww) * ker[kr0:kr1, kc0:kc1]


def order_quad_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(ang)]
    i0 = np.argmin(pts[:, 0] + pts[:, 1])
    return np.roll(pts, -i0, axis=0)


def global_obb_from_projection_image(img: np.ndarray, thr: float, min_pixels: int = 4) -> np.ndarray | None:
    if img is None or img.size == 0 or not np.isfinite(thr):
        return None
    mask = (img > thr).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build x-only sequential MSOD-style dataset from XT simulation.")
    parser.add_argument("--out-root", type=Path, default=Path("dataset/xt_seq_msod_x"))
    parser.add_argument("--dataset-name", type=str, default="MSOD_RAWBG")
    parser.add_argument("--source-dirs", nargs="+", default=DEFAULT_SOURCE_DIRS)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--save-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--index-history", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--flag-bkg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--peak-mask-ratio", type=float, default=0.15)
    parser.add_argument("--mask-dilate", type=int, default=1)
    return parser.parse_args()


def build_background_masks(path_dirs: list[str], flag_bkg: bool) -> list[np.ndarray | None]:
    if not flag_bkg:
        return [None for _ in path_dirs]

    import sep

    meds: list[np.ndarray | None] = []
    for path_dir in path_dirs:
        aligns = []
        paths = sorted(glob.glob(path_dir + "*.tif"))
        for path in tqdm(paths, desc=f"Median mask: {Path(path_dir).name}"):
            img = cv2.imread(path, -1)
            if img is not None:
                aligns.append(img)
        if not aligns:
            raise FileNotFoundError(f"No TIFF files found in {path_dir}")
        med = np.median(np.array(aligns), axis=0)
        bkg = sep.Background(np.ascontiguousarray(med, np.float32))
        bkg_img = np.array(bkg)
        _, ext_map = sep.extract(med - bkg_img, 5, err=bkg.globalrms, deblend_cont=1, segmentation_map=True)
        meds.append(ext_map)
    return meds


def build_patch_records(path_dirs: list[str], patch_size: int, stride: int) -> list[dict]:
    records = []
    for num_dir, path_dir in enumerate(path_dirs):
        paths = sorted(glob.glob(path_dir + "*.tif"))
        for path in paths:
            img = cv2.imread(path, -1)
            if img is None:
                continue
            h, w = img.shape[:2]
            nw = (w - patch_size) // stride + 1
            nh = (h - patch_size) // stride + 1
            for iy in range(nh):
                for ix in range(nw):
                    x1, y1 = ix * stride, iy * stride
                    records.append(
                        {
                            "source_index": num_dir,
                            "path": path,
                            "x1": x1,
                            "y1": y1,
                            "x2": x1 + patch_size,
                            "y2": y1 + patch_size,
                        }
                    )
    return records


def target_to_mask(target: np.ndarray, peak_mask_ratio: float, dilate_iter: int) -> np.ndarray:
    peak = float(target.max())
    if peak <= 1e-6:
        return np.zeros_like(target, dtype=np.uint8)
    thr = peak * float(peak_mask_ratio)
    mask = (target >= thr).astype(np.uint8)
    if dilate_iter > 0 and mask.any():
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return mask


def scale_to_uint8(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    img = np.clip(img, vmin, vmax)
    out = (img - vmin) / (vmax - vmin + 1e-8) * 255.0
    return out.astype(np.uint8)


def build_effective_kernel(radius_px: float, speed_px_s: float, exposure_s: float, angle_rad: float) -> np.ndarray:
    psf = _gaussian_psf(radius_px)
    blur_len = max(0.0, float(speed_px_s) * float(exposure_s))
    if blur_len < 1.0:
        return psf
    mk = _motion_kernel(blur_len, angle_rad, width=max(0.8, radius_px * 0.6))
    eff = _fft_convolve_full(psf, mk)
    eff /= eff.sum()
    return eff.astype(np.float32)


def simulate_target_sequence(
    seed: int,
    img_shape: tuple[int, int],
    radius_px: float,
    peak: float,
    speed_px_s: float,
    angle_deg: float,
    n_frames: int,
    exposure_s: float,
    frame_interval_s: float,
    start_xy: tuple[float, float] | None = None,
    jitter_px: float = 0.5,
    scintillation: float = 0.15,
    periodic_amp_px: float | None = None,
    periodic_along_amp_px: float | None = None,
    period_frames: float | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    rng = np.random.default_rng(seed)
    h, w = img_shape
    angle = np.deg2rad(float(angle_deg))
    dirx, diry = np.cos(angle), np.sin(angle)
    vx = float(speed_px_s) * dirx
    vy = float(speed_px_s) * diry
    if start_xy is None:
        x0 = rng.uniform(w * 0.25, w * 0.75)
        y0 = rng.uniform(h * 0.25, h * 0.75)
    else:
        x0, y0 = map(float, start_xy)
    if period_frames is None:
        period_frames = rng.uniform(max(6.0, n_frames * 0.3), max(10.0, n_frames * 0.9))
    if periodic_amp_px is None:
        periodic_amp_px = rng.uniform(0.3, 2.2)
    if periodic_along_amp_px is None:
        periodic_along_amp_px = rng.uniform(0.0, 1.2)
    phase_perp = rng.uniform(0.0, 2.0 * np.pi)
    phase_along = rng.uniform(0.0, 2.0 * np.pi)
    eff = build_effective_kernel(radius_px, speed_px_s, exposure_s, angle)

    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    metadata: list[dict] = []
    for frame_idx in range(n_frames):
        t = frame_idx * float(frame_interval_s)
        ph = 2.0 * np.pi * (frame_idx / max(period_frames, 1e-6))
        perp = float(periodic_amp_px) * np.sin(ph + phase_perp)
        along = float(periodic_along_amp_px) * np.sin(0.6 * ph + phase_along)
        x = x0 + vx * t + dirx * along - diry * perp + rng.normal(0.0, jitter_px)
        y = y0 + vy * t + diry * along + dirx * perp + rng.normal(0.0, jitter_px)
        pk = float(peak) * (1.0 + 0.15 * np.sin(ph + phase_along) + rng.normal(0.0, scintillation))
        pk = max(0.0, pk)
        tar = np.zeros((h, w), dtype=np.float32)
        _place_kernel_add(tar, eff, y, x, pk)
        frames.append(tar)
        masks.append(tar)
        metadata.append({"x": float(x), "y": float(y), "peak": float(pk)})
    return frames, masks, metadata


def write_index_file(path: Path, items: list[str]) -> None:
    path.write_text("".join(f"{item}\n" for item in items), encoding="utf-8")


def prepare_out_root(dataset_dir: Path, keep_existing: bool) -> None:
    if dataset_dir.exists() and not keep_existing:
        shutil.rmtree(dataset_dir)
    for split in ["train", "val", "test"]:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "masks").mkdir(parents=True, exist_ok=True)


def split_records(records: list[dict], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[dict]]:
    if not 0.0 < train_ratio < 1.0 or not 0.0 <= val_ratio < 1.0:
        raise ValueError("Invalid split ratios.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.")
    rows = list(records)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_total = len(rows)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    return {
        "train": rows[:n_train],
        "val": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }


def build_dataset(args: argparse.Namespace) -> dict:
    dataset_dir = args.out_root / args.dataset_name
    prepare_out_root(dataset_dir, keep_existing=args.keep_existing)

    background_masks = build_background_masks(args.source_dirs, flag_bkg=args.flag_bkg)
    patch_records = build_patch_records(args.source_dirs, patch_size=args.patch_size, stride=args.stride)
    if not patch_records:
        raise RuntimeError("No patch records found.")
    if args.max_sequences > 0:
        rng = random.Random(args.seed)
        patch_records = rng.sample(patch_records, min(args.max_sequences, len(patch_records)))
    split_map = split_records(patch_records, args.train_ratio, args.val_ratio, args.seed)

    all_index: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    meta_path = dataset_dir / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as meta_file:
        seq_global = 1
        for split, rows in split_map.items():
            for row in tqdm(rows, desc=f"Build {split}"):
                ori = cv2.imread(row["path"], -1)
                if ori is None:
                    continue
                patch = ori[row["y1"] : row["y2"], row["x1"] : row["x2"]].astype(np.float32)
                ext_map = background_masks[row["source_index"]]
                if ext_map is not None:
                    patch_mask = ext_map[row["y1"] : row["y2"], row["x1"] : row["x2"]] > 0
                    patch = patch.copy()
                    patch[patch_mask] = np.median(patch)

                seq_seed = args.seed + seq_global
                rng = np.random.default_rng(seq_seed)
                exp = max(rng.normal(1.0, 0.3), 0.05)

                peak_pt = max(rng.normal(0.1, 0.03), 0.001) * float(patch.max())
                pt_frames, pt_targets, pt_meta = simulate_target_sequence(
                    seed=seq_seed,
                    img_shape=patch.shape,
                    n_frames=args.seq_len,
                    speed_px_s=float(rng.integers(1, 5)),
                    exposure_s=exp,
                    frame_interval_s=exp * float(rng.integers(2, 15)),
                    radius_px=max(rng.normal(2, 0.7), 0.5),
                    peak=peak_pt,
                    angle_deg=float(rng.uniform(-90, 90)),
                )

                rng2 = np.random.default_rng(seq_seed + 1)
                peak_line = max(rng2.normal(0.1, 0.03), 0.001) * float(patch.max())
                line_frames, line_targets, line_meta = simulate_target_sequence(
                    seed=seq_seed + 1,
                    img_shape=patch.shape,
                    n_frames=args.seq_len,
                    speed_px_s=float(rng2.integers(3, 23)),
                    exposure_s=exp,
                    frame_interval_s=exp * float(rng2.integers(2, 5)),
                    radius_px=max(rng2.normal(1, 0.3), 0.2),
                    peak=peak_line,
                    angle_deg=float(rng2.uniform(-90, 90)),
                )

                stack = np.stack([patch + pt_frames[i] + line_frames[i] for i in range(args.seq_len)], axis=0)
                seq_vmin, seq_vmax = np.percentile(stack, (1.0, 99.0))
                seq_id = f"{seq_global:04d}"
                img_dir = dataset_dir / split / "images" / seq_id
                mask_dir = dataset_dir / split / "masks" / seq_id
                img_dir.mkdir(parents=True, exist_ok=True)
                mask_dir.mkdir(parents=True, exist_ok=True)

                for frame_idx in range(args.seq_len):
                    syn = stack[frame_idx]
                    img_u8 = scale_to_uint8(syn, float(seq_vmin), float(seq_vmax))
                    pt_mask = target_to_mask(pt_targets[frame_idx], args.peak_mask_ratio, args.mask_dilate)
                    line_mask = target_to_mask(line_targets[frame_idx], args.peak_mask_ratio, args.mask_dilate)
                    mask_u8 = np.clip(pt_mask + line_mask, 0, 1).astype(np.uint8) * 255

                    pt_obb = global_obb_from_projection_image(
                        pt_targets[frame_idx],
                        max(float(pt_targets[frame_idx].max()) * args.peak_mask_ratio, 1e-6),
                    )
                    line_obb = global_obb_from_projection_image(
                        line_targets[frame_idx],
                        max(float(line_targets[frame_idx].max()) * args.peak_mask_ratio, 1e-6),
                    )

                    if args.save_size != args.patch_size:
                        img_u8 = cv2.resize(img_u8, (args.save_size, args.save_size), interpolation=cv2.INTER_AREA)
                        mask_u8 = cv2.resize(mask_u8, (args.save_size, args.save_size), interpolation=cv2.INTER_NEAREST)

                    frame_name = f"{frame_idx + 1:05d}"
                    cv2.imwrite(str(img_dir / f"{frame_name}.tif"), img_u8)
                    cv2.imwrite(str(mask_dir / f"{frame_name}.png"), mask_u8)

                    if frame_idx + 1 >= args.index_history:
                        all_index[split].append(f"{seq_id}/{frame_name}")

                    meta = {
                        "split": split,
                        "seq_id": seq_id,
                        "frame_id": frame_name,
                        "source_path": row["path"],
                        "crop": [row["x1"], row["y1"], row["x2"], row["y2"]],
                        "pt_center": pt_meta[frame_idx],
                        "line_center": line_meta[frame_idx],
                        "pt_obb": None if pt_obb is None else np.asarray(pt_obb).reshape(-1).round(4).tolist(),
                        "line_obb": None if line_obb is None else np.asarray(line_obb).reshape(-1).round(4).tolist(),
                    }
                    meta_file.write(json.dumps(meta, ensure_ascii=True) + "\n")

                seq_global += 1

    for split in ["train", "val", "test"]:
        write_index_file(dataset_dir / f"{split}.txt", all_index[split])

    summary = {
        "dataset_dir": str(dataset_dir),
        "splits": {split: len(items) for split, items in all_index.items()},
        "seq_len": args.seq_len,
        "index_history": args.index_history,
        "save_size": args.save_size,
        "patch_size": args.patch_size,
        "sequences": seq_global - 1,
    }
    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
