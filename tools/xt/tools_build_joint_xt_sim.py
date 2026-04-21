from __future__ import annotations

import argparse
import glob
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import sep
from tqdm import tqdm


DEFAULT_SOURCE_DIRS = [
    "/mnt/e/Imgs/03-TJStars/150ms-2k2k/decode/",
    "/mnt/e/Imgs/03-TJStars/3000ms-4k4k/decode/",
    "/mnt/e/Imgs/03-TJStars/TJ2/decode/",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a joint simulation dataset for compare segmentation models and XT detector models."
    )
    parser.add_argument("--source-dirs", nargs="+", default=DEFAULT_SOURCE_DIRS)
    parser.add_argument("--seq-root", type=Path, default=Path("dataset/xt_seq_msod_x"))
    parser.add_argument("--seq-dataset-name", type=str, default="MSOD_SIM_JOINT")
    parser.add_argument("--obb-root", type=Path, default=Path("dataset/obb_xt_joint_sim"))
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--seq-save-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--index-history", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument(
        "--compare-flag-bkg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether compare-model sequence inputs should suppress structured background before simulation.",
    )
    parser.add_argument(
        "--ours-flag-bkg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether detector XT inputs should suppress structured background before simulation.",
    )
    parser.add_argument("--peak-mask-ratio", type=float, default=0.15)
    parser.add_argument("--mask-dilate", type=int, default=1)
    parser.add_argument(
        "--compare-seq-percentile",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        help="Percentiles used to rescale compare-model sequence frames.",
    )
    parser.add_argument(
        "--ours-ratio-x-bkg",
        type=float,
        nargs=2,
        default=(0.01, 99.99),
        help="Percentiles used to rescale detector X after background suppression.",
    )
    parser.add_argument(
        "--ours-ratio-x-raw",
        type=float,
        nargs=2,
        default=(0.5, 99.5),
        help="Percentiles used to rescale detector X when background suppression is disabled.",
    )
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


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


def polygon_area(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def global_obb_from_projection_image(img: np.ndarray, thr: float, min_pixels: int = 4) -> np.ndarray | None:
    if img is None or img.size == 0 or not np.isfinite(thr):
        return None
    mask = (img > thr).astype(np.uint8)
    if mask.any():
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
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


def scale_to_uint8(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    img = np.clip(img, vmin, vmax)
    out = (img - vmin) / (vmax - vmin + 1e-8) * 255.0
    return out.astype(np.uint8)


def trunc_img(img: np.ndarray, ratio: tuple[float, float]) -> np.ndarray:
    vmin, vmax = np.percentile(img, ratio)
    img = np.clip(img, vmin, vmax)
    out = (img - vmin) / (vmax - vmin + 1e-8) * 255.0
    return out.astype(np.uint8)


def encode_xt_sc(x_u8: np.ndarray, t_u8: np.ndarray) -> np.ndarray:
    tau = t_u8.astype(np.float32) / 255.0
    sin8 = ((np.sin(np.pi * tau) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    cos8 = ((np.cos(np.pi * tau) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return np.stack([x_u8, sin8, cos8], axis=-1)


def _quantize_t_to_global_idx(t: np.ndarray, n_local: int, n_global: int) -> np.ndarray:
    n_local = int(max(n_local, 1))
    n_global = int(max(n_global, 1))
    if n_global <= 1 or n_local <= 1:
        return np.zeros_like(t, dtype=np.int32)
    idx_local = np.round(np.clip(t, 0.0, 1.0) * float(n_local - 1)).astype(np.int32)
    idx_global = np.round(idx_local.astype(np.float32) * float(n_global - 1) / float(n_local - 1)).astype(np.int32)
    return np.clip(idx_global, 0, n_global - 1)


def compose_frame_index_projection_unified(
    t1: np.ndarray,
    w1: np.ndarray,
    t2: np.ndarray,
    w2: np.ndarray,
    n1: int,
    n2: int,
    rng: np.random.Generator,
    conf_thr: float = 0.10,
    hist_energy_thr: float = 0.05,
) -> np.ndarray:
    n_bg = max(int(n1), int(n2), 2)
    idx1 = _quantize_t_to_global_idx(t1, n1, n_bg)
    idx2 = _quantize_t_to_global_idx(t2, n2, n_bg)
    w1 = np.asarray(w1, dtype=np.float32)
    w2 = np.asarray(w2, dtype=np.float32)
    wsum = w1 + w2
    use1 = w1 >= w2
    idx_fg = np.where(use1, idx1, idx2)
    dom = np.abs(w1 - w2) / (wsum + 1e-6)
    scale = np.percentile(wsum, 99.9) + 1e-6
    energy = np.clip(wsum / scale, 0.0, 1.0)
    conf = dom * energy
    low = conf < float(conf_thr)
    hist_src = idx_fg[energy < float(hist_energy_thr)]
    if hist_src.size < 32:
        hist_src = idx_fg.reshape(-1)
    hist = np.bincount(hist_src.astype(np.int32), minlength=n_bg).astype(np.float64)
    if hist.sum() <= 0:
        hist = np.ones(n_bg, dtype=np.float64)
    idx_out = idx_fg.copy()
    if int(low.sum()) > 0:
        idx_out[low] = rng.choice(n_bg, size=int(low.sum()), p=(hist / hist.sum()))
    return np.round(idx_out.astype(np.float32) / float(n_bg - 1) * 255.0).astype(np.uint8)


def target_to_mask(target: np.ndarray, peak_mask_ratio: float, dilate_iter: int) -> np.ndarray:
    peak = float(target.max())
    if peak <= 1e-6:
        return np.zeros_like(target, dtype=np.uint8)
    mask = (target >= peak * float(peak_mask_ratio)).astype(np.uint8)
    if dilate_iter > 0 and mask.any():
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=dilate_iter)
    return mask


def suppress_background(patch: np.ndarray, ext_map: np.ndarray | None, row: dict) -> np.ndarray:
    out = patch.copy()
    if ext_map is None:
        return out
    patch_mask = ext_map[row["y1"] : row["y2"], row["x1"] : row["x2"]] > 0
    if patch_mask.any():
        out[patch_mask] = np.median(out)
    return out


def build_background_masks(path_dirs: list[str], flag_bkg: bool) -> list[np.ndarray | None]:
    if not flag_bkg:
        return [None for _ in path_dirs]
    masks: list[np.ndarray | None] = []
    for path_dir in path_dirs:
        paths = sorted(glob.glob(path_dir + "*.tif"))
        if not paths:
            raise FileNotFoundError(f"No TIFF files found in {path_dir}")
        aligns = []
        for path in tqdm(paths, desc=f"Median mask: {Path(path_dir).name}"):
            img = cv2.imread(path, -1)
            if img is not None:
                aligns.append(img)
        med = np.median(np.array(aligns), axis=0)
        bkg = sep.Background(np.ascontiguousarray(med, np.float32))
        _, ext_map = sep.extract(med - np.array(bkg), 5, err=bkg.globalrms, deblend_cont=1, segmentation_map=True)
        masks.append(ext_map)
    return masks


def build_patch_records(path_dirs: list[str], patch_size: int, stride: int) -> list[dict]:
    records = []
    for source_index, path_dir in enumerate(path_dirs):
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
                            "source_index": source_index,
                            "path": path,
                            "x1": x1,
                            "y1": y1,
                            "x2": x1 + patch_size,
                            "y2": y1 + patch_size,
                        }
                    )
    return records


def split_records(records: list[dict], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[dict]]:
    if not 0.0 < train_ratio < 1.0 or not 0.0 <= val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Invalid split ratios.")
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


def sample_target_params(seed: int, img_shape: tuple[int, int], peak_scale: float, kind: str) -> dict:
    rng = np.random.default_rng(seed)
    h, w = img_shape
    if kind == "pt":
        speed = float(rng.integers(1, 5))
        interval_mul = float(rng.integers(2, 15))
        radius = max(float(rng.normal(2.0, 0.7)), 0.5)
        peak = max(float(rng.normal(0.1, 0.03)), 0.001) * peak_scale
    else:
        speed = float(rng.integers(3, 23))
        interval_mul = float(rng.integers(2, 5))
        radius = max(float(rng.normal(1.0, 0.3)), 0.2)
        peak = max(float(rng.normal(0.1, 0.03)), 0.001) * peak_scale
    exp = max(float(rng.normal(1.0, 0.3)), 0.05)
    return {
        "img_shape": (h, w),
        "n_frames": None,
        "radius_px": radius,
        "peak": peak,
        "start_xy": (float(rng.uniform(w * 0.25, w * 0.75)), float(rng.uniform(h * 0.25, h * 0.75))),
        "speed_px_s": speed,
        "angle_deg": float(rng.uniform(-90, 90)),
        "exposure_s": exp,
        "frame_interval_s": exp * interval_mul,
        "jitter_px": 0.5,
        "scintillation": 0.15,
        "periodic_amp_px": float(rng.uniform(0.3, 2.2)),
        "periodic_along_amp_px": float(rng.uniform(0.0, 1.2)),
        "period_frames": float(rng.uniform(6.0, 10.0)),
    }


def simulate_target_bundle(seed: int, params: dict, n_frames: int) -> dict:
    rng = np.random.default_rng(seed)
    h, w = params["img_shape"]
    angle = np.deg2rad(float(params["angle_deg"]))
    dirx, diry = np.cos(angle), np.sin(angle)
    vx = float(params["speed_px_s"]) * dirx
    vy = float(params["speed_px_s"]) * diry
    x0, y0 = params["start_xy"]
    eff = _gaussian_psf(float(params["radius_px"]))
    blur_len = max(0.0, float(params["speed_px_s"]) * float(params["exposure_s"]))
    if blur_len >= 1.0:
        mk = _motion_kernel(blur_len, angle, width=max(0.8, float(params["radius_px"]) * 0.6))
        eff = _fft_convolve_full(eff, mk)
        eff /= eff.sum()

    frames = []
    centers = []
    proj = np.zeros((h, w), dtype=np.float32)
    t_num = np.zeros((h, w), dtype=np.float32)
    t_den = np.zeros((h, w), dtype=np.float32)
    phase_perp = rng.uniform(0.0, 2.0 * np.pi)
    phase_along = rng.uniform(0.0, 2.0 * np.pi)
    period_frames = max(float(params["period_frames"]), 1.0)

    for frame_idx in range(n_frames):
        t_now = frame_idx * float(params["frame_interval_s"])
        ph = 2.0 * np.pi * (frame_idx / period_frames)
        perp = float(params["periodic_amp_px"]) * np.sin(ph + phase_perp)
        along = float(params["periodic_along_amp_px"]) * np.sin(0.6 * ph + phase_along)
        x = x0 + vx * t_now + dirx * along - diry * perp + rng.normal(0.0, float(params["jitter_px"]))
        y = y0 + vy * t_now + diry * along + dirx * perp + rng.normal(0.0, float(params["jitter_px"]))
        pk = float(params["peak"]) * (1.0 + 0.15 * np.sin(ph + phase_along) + rng.normal(0.0, float(params["scintillation"])))
        pk = max(0.0, pk)

        frame = np.zeros((h, w), dtype=np.float32)
        _place_kernel_add(frame, eff, y, x, pk)
        frames.append(frame)
        proj += frame
        frame_id = (frame_idx + 1.0) / float(max(n_frames, 1))
        _place_kernel_add(t_num, eff, y, x, pk * frame_id)
        _place_kernel_add(t_den, eff, y, x, pk)
        centers.append({"x": float(x), "y": float(y), "peak": float(pk)})

    t_proj = np.clip(t_num / (t_den + 1e-6), 0.0, 1.0).astype(np.float32)
    conf = t_den / (float(t_den.max()) + 1e-12)
    low = conf < 0.08
    if np.any(low):
        ridx = rng.integers(0, max(n_frames, 1), size=int(low.sum()))
        t_proj[low] = ridx.astype(np.float32) / float(max(n_frames - 1, 1))

    return {"frames": frames, "proj": proj, "t_proj": t_proj, "centers": centers}


def prepare_seq_root(dataset_dir: Path, keep_existing: bool) -> None:
    if dataset_dir.exists() and not keep_existing:
        shutil.rmtree(dataset_dir)
    for split in ["train", "val", "test"]:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "masks").mkdir(parents=True, exist_ok=True)


def prepare_obb_root(root: Path, keep_existing: bool) -> None:
    if root.exists() and not keep_existing:
        shutil.rmtree(root)
    for split in ["train", "val", "test"]:
        for sub in [
            ("images", split),
            ("images", f"{split}_t"),
            ("images", f"{split}_xt"),
            ("images", f"{split}_xt_sc"),
            ("labels", split),
            ("labels", f"{split}_t"),
            ("labels", f"{split}_xt"),
            ("labels", f"{split}_xt_sc"),
        ]:
            (root / sub[0] / sub[1]).mkdir(parents=True, exist_ok=True)


def write_index_file(path: Path, items: list[str]) -> None:
    path.write_text("".join(f"{item}\n" for item in items), encoding="utf-8")


def write_obb_yamls(root: Path) -> None:
    mapping = {
        "data_x.yaml": "images/train",
        "data_t.yaml": "images/train_t",
        "data_xt.yaml": "images/train_xt",
        "data_xt_sc.yaml": "images/train_xt_sc",
    }
    for name, train_path in mapping.items():
        val_path = train_path.replace("train", "val")
        test_path = train_path.replace("train", "test")
        text = "\n".join(
            [
                f"path: {root.as_posix()}",
                f"train: {train_path}",
                f"val: {val_path}",
                f"test: {test_path}",
                "names:",
                "  0: line",
                "",
            ]
        )
        (root / name).write_text(text, encoding="utf-8")


def build_joint_dataset(args: argparse.Namespace) -> dict:
    seq_dir = (args.seq_root / args.seq_dataset_name).resolve()
    obb_root = args.obb_root.resolve()
    prepare_seq_root(seq_dir, keep_existing=args.keep_existing)
    prepare_obb_root(obb_root, keep_existing=args.keep_existing)

    seq_ratio = tuple(args.compare_seq_percentile)
    ours_ratio_x = tuple(args.ours_ratio_x_bkg if args.ours_flag_bkg else args.ours_ratio_x_raw)
    need_bkg_masks = args.compare_flag_bkg or args.ours_flag_bkg
    background_masks = build_background_masks(args.source_dirs, flag_bkg=need_bkg_masks)
    patch_records = build_patch_records(args.source_dirs, patch_size=args.patch_size, stride=args.stride)
    if not patch_records:
        raise RuntimeError("No patch records found.")
    if args.max_sequences > 0:
        rng = random.Random(args.seed)
        patch_records = rng.sample(patch_records, min(args.max_sequences, len(patch_records)))
    split_map = split_records(patch_records, args.train_ratio, args.val_ratio, args.seed)

    scale_xy = float(args.seq_save_size) / float(args.patch_size)
    seq_index = {"train": [], "val": [], "test": []}
    seq_meta_path = seq_dir / "metadata.jsonl"
    obb_meta = []
    seq_count = 0

    with seq_meta_path.open("w", encoding="utf-8") as seq_meta_file:
        for split, rows in split_map.items():
            for row in tqdm(rows, desc=f"Build {split}"):
                ori = cv2.imread(row["path"], -1)
                if ori is None:
                    continue
                patch_raw = ori[row["y1"] : row["y2"], row["x1"] : row["x2"]].astype(np.float32)
                ext_map = background_masks[row["source_index"]]
                patch_compare = (
                    suppress_background(patch_raw, ext_map, row) if args.compare_flag_bkg else patch_raw.copy()
                )
                patch_ours = suppress_background(patch_raw, ext_map, row) if args.ours_flag_bkg else patch_raw.copy()

                seq_id = f"{seq_count + 1:04d}"
                peak_scale = float(max(patch_raw.max(), 1.0))
                pt = simulate_target_bundle(
                    seq_count * 2 + args.seed + 1,
                    sample_target_params(seq_count * 2 + args.seed + 1, patch_raw.shape, peak_scale, "pt"),
                    args.seq_len,
                )
                line = simulate_target_bundle(
                    seq_count * 2 + args.seed + 2,
                    sample_target_params(seq_count * 2 + args.seed + 2, patch_raw.shape, peak_scale, "line"),
                    args.seq_len,
                )

                seq_stack = np.stack(
                    [patch_compare + pt["frames"][i] + line["frames"][i] for i in range(args.seq_len)],
                    axis=0,
                )
                seq_vmin, seq_vmax = np.percentile(seq_stack, seq_ratio)
                img_dir = seq_dir / split / "images" / seq_id
                mask_dir = seq_dir / split / "masks" / seq_id
                img_dir.mkdir(parents=True, exist_ok=True)
                mask_dir.mkdir(parents=True, exist_ok=True)

                for frame_idx in range(args.seq_len):
                    frame_name = f"{frame_idx + 1:05d}"
                    frame_u8 = scale_to_uint8(seq_stack[frame_idx], float(seq_vmin), float(seq_vmax))
                    pt_mask = target_to_mask(pt["frames"][frame_idx], args.peak_mask_ratio, args.mask_dilate)
                    line_mask = target_to_mask(line["frames"][frame_idx], args.peak_mask_ratio, args.mask_dilate)
                    mask_u8 = np.clip(pt_mask + line_mask, 0, 1).astype(np.uint8) * 255

                    if args.seq_save_size != args.patch_size:
                        frame_u8 = cv2.resize(frame_u8, (args.seq_save_size, args.seq_save_size), interpolation=cv2.INTER_AREA)
                        mask_u8 = cv2.resize(mask_u8, (args.seq_save_size, args.seq_save_size), interpolation=cv2.INTER_NEAREST)

                    cv2.imwrite(str(img_dir / f"{frame_name}.tif"), frame_u8)
                    cv2.imwrite(str(mask_dir / f"{frame_name}.png"), mask_u8)

                    if frame_idx + 1 >= args.index_history:
                        seq_index[split].append(f"{seq_id}/{frame_name}")

                    pt_obb = global_obb_from_projection_image(pt["frames"][frame_idx], max(float(pt["frames"][frame_idx].max()) * args.peak_mask_ratio, 1e-6))
                    line_obb = global_obb_from_projection_image(line["frames"][frame_idx], max(float(line["frames"][frame_idx].max()) * args.peak_mask_ratio, 1e-6))
                    row_meta = {
                        "split": split,
                        "seq_id": seq_id,
                        "frame_id": frame_name,
                        "source_path": row["path"],
                        "crop": [row["x1"], row["y1"], row["x2"], row["y2"]],
                        "pt_center": pt["centers"][frame_idx],
                        "line_center": line["centers"][frame_idx],
                        "pt_obb": None if pt_obb is None else np.asarray(pt_obb).reshape(-1).round(4).tolist(),
                        "line_obb": None if line_obb is None else np.asarray(line_obb).reshape(-1).round(4).tolist(),
                    }
                    seq_meta_file.write(json.dumps(row_meta, ensure_ascii=True) + "\n")

                x_raw = patch_ours + pt["proj"] + line["proj"]
                x_u8 = trunc_img(x_raw, ratio=ours_ratio_x)
                t_u8 = compose_frame_index_projection_unified(
                    pt["t_proj"],
                    pt["proj"],
                    line["t_proj"],
                    line["proj"],
                    args.seq_len,
                    args.seq_len,
                    rng=np.random.default_rng(args.seed + 100000 + seq_count),
                )
                xt_u8 = np.stack([x_u8, t_u8, x_u8], axis=-1)
                xt_sc_u8 = encode_xt_sc(x_u8, t_u8)
                x_rgb = np.stack([x_u8, x_u8, x_u8], axis=-1)

                pt_obb = global_obb_from_projection_image(pt["proj"], max(float(np.percentile(pt["proj"], 99.9)), 1e-6))
                line_obb = global_obb_from_projection_image(line["proj"], max(float(np.percentile(line["proj"], 99.9)), 1e-6))
                label_lines = []
                for obb in [line_obb, pt_obb]:
                    if obb is None:
                        continue
                    line_txt = to_yolo_obb_line(obb, w=x_u8.shape[1], h=x_u8.shape[0])
                    if line_txt is not None and line_txt not in label_lines:
                        label_lines.append(line_txt)

                stem = seq_id
                cv2.imwrite(str(obb_root / "images" / split / f"{stem}.png"), x_rgb)
                cv2.imwrite(str(obb_root / "images" / f"{split}_t" / f"{stem}.png"), t_u8)
                cv2.imwrite(str(obb_root / "images" / f"{split}_xt" / f"{stem}.png"), xt_u8)
                cv2.imwrite(str(obb_root / "images" / f"{split}_xt_sc" / f"{stem}.png"), xt_sc_u8)
                for sub in [
                    obb_root / "labels" / split,
                    obb_root / "labels" / f"{split}_t",
                    obb_root / "labels" / f"{split}_xt",
                    obb_root / "labels" / f"{split}_xt_sc",
                ]:
                    with (sub / f"{stem}.txt").open("w", encoding="utf-8") as f:
                        if label_lines:
                            f.write("\n".join(label_lines) + "\n")

                obb_meta.append(
                    {
                        "split": split,
                        "stem": stem,
                        "seq_id": seq_id,
                        "source_path": row["path"],
                        "crop": [row["x1"], row["y1"], row["x2"], row["y2"]],
                        "pt_obb": None if pt_obb is None else np.asarray(pt_obb).reshape(-1).round(4).tolist(),
                        "line_obb": None if line_obb is None else np.asarray(line_obb).reshape(-1).round(4).tolist(),
                    }
                )
                seq_count += 1

    for split in ["train", "val", "test"]:
        write_index_file(seq_dir / f"{split}.txt", seq_index[split])

    seq_summary = {
        "dataset_dir": str(seq_dir),
        "splits": {split: len(items) for split, items in seq_index.items()},
        "seq_len": args.seq_len,
        "index_history": args.index_history,
        "save_size": args.seq_save_size,
        "patch_size": args.patch_size,
        "sequences": seq_count,
        "paired_obb_root": str(obb_root),
        "compare_preprocess": {
            "flag_bkg": bool(args.compare_flag_bkg),
            "frame_rescale_percentile": list(seq_ratio),
        },
    }
    (seq_dir / "summary.json").write_text(json.dumps(seq_summary, indent=2), encoding="utf-8")

    write_obb_yamls(obb_root)
    obb_summary = {
        "dataset_dir": str(obb_root),
        "sequences": seq_count,
        "patch_size": args.patch_size,
        "source_seq_root": str(seq_dir),
        "time_encoding": "compose_frame_index_projection_unified",
        "x_consistency": "same raw patch and same simulated trajectories as sequence dataset",
        "detector_preprocess": {
            "flag_bkg": bool(args.ours_flag_bkg),
            "x_rescale_percentile": list(ours_ratio_x),
        },
    }
    (obb_root / "summary.json").write_text(json.dumps(obb_summary, indent=2), encoding="utf-8")
    (obb_root / "metadata.json").write_text(json.dumps(obb_meta, indent=2), encoding="utf-8")

    return {"seq_summary": seq_summary, "obb_summary": obb_summary}


def main() -> None:
    args = parse_args()
    summary = build_joint_dataset(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
