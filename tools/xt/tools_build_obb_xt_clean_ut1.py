from __future__ import annotations

import glob
import math
import random
from pathlib import Path

import cv2
import numpy as np
import sep
from tqdm import tqdm


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
    L = max(1.0, float(length_px))
    size = int(np.ceil(L)) | 1
    c = size // 2
    y, x = np.mgrid[-c : c + 1, -c : c + 1]
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    u = ca * x + sa * y
    v = -sa * x + ca * y
    line = (np.abs(u) <= (L / 2)).astype(np.float32)
    blur = np.exp(-(v * v) / (2 * (max(0.3, width) ** 2)))
    k = line * blur
    k /= k.sum()
    return k.astype(np.float32)


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


def simulate_trajectory_projection_with_interval(
    seed: int,
    img_shape=(256, 256),
    radius_px=1.5,
    peak=2000.0,
    start_xy=None,
    speed_px_s=8.0,
    angle_deg=None,
    n_frames=50,
    exposure_s=0.01,
    frame_interval_s=0.04,
    jitter_px=0.5,
    scintillation=0.15,
    background=50.0,
    shot_noise=False,
    read_noise=3.0,
    periodic_amp_px=None,
    periodic_along_amp_px=None,
    period_frames=None,
    return_time_proj=False,
    t_uncertainty=True,
    t_core_keep_ratio=0.90,
    t_edge_threshold=0.25,
    t_edge_sharpness=8.0,
    t_lowconf_randomize=True,
    t_lowconf_ratio=0.08,
):
    rng = np.random.default_rng(seed)
    h, w = img_shape
    if angle_deg is None:
        angle_deg = rng.uniform(0, 90)
    angle = np.deg2rad(float(angle_deg))
    dirx, diry = np.cos(angle), np.sin(angle)
    vx = float(speed_px_s) * dirx
    vy = float(speed_px_s) * diry
    if start_xy is None:
        x0 = rng.uniform(w // 4, w // 4 * 3)
        y0 = rng.uniform(h // 4, h // 4 * 3)
    else:
        x0, y0 = map(float, start_xy)

    n_frames_i = max(int(n_frames), 1)
    if period_frames is None:
        period_frames = rng.uniform(max(6.0, n_frames_i * 0.3), max(10.0, n_frames_i * 0.9))
    if periodic_amp_px is None:
        periodic_amp_px = rng.uniform(0.3, 2.2)
    if periodic_along_amp_px is None:
        periodic_along_amp_px = rng.uniform(0.0, 1.2)
    phase_perp = rng.uniform(0.0, 2.0 * np.pi)
    phase_along = rng.uniform(0.0, 2.0 * np.pi)

    psf = _gaussian_psf(radius_px)
    blur_len = max(0.0, float(speed_px_s) * float(exposure_s))
    if blur_len >= 1.0:
        mk = _motion_kernel(blur_len, angle, width=max(0.8, radius_px * 0.6))
        eff = _fft_convolve_full(psf, mk)
        eff /= eff.sum()
    else:
        eff = psf

    tar = np.zeros((h, w), dtype=np.float32)
    if return_time_proj:
        t_num = np.zeros((h, w), dtype=np.float32)
        t_den = np.zeros((h, w), dtype=np.float32)

    for k in range(n_frames_i):
        t = k * float(frame_interval_s)
        ph = 2.0 * np.pi * (k / max(period_frames, 1e-6))
        perp = float(periodic_amp_px) * np.sin(ph + phase_perp)
        along = float(periodic_along_amp_px) * np.sin(0.6 * ph + phase_along)
        x = x0 + vx * t + dirx * along - diry * perp + rng.normal(0.0, jitter_px)
        y = y0 + vy * t + diry * along + dirx * perp + rng.normal(0.0, jitter_px)
        pk = float(peak) * (1.0 + 0.15 * np.sin(ph + phase_along) + rng.normal(0.0, scintillation))
        pk = max(0.0, pk)
        _place_kernel_add(tar, eff, y, x, pk)

        if return_time_proj:
            frame_id = (k + 1.0) / float(n_frames_i)
            if t_uncertainty:
                rel = eff / (eff.max() + 1e-12)
                strength = np.clip(pk / (float(peak) + 1e-12), 0.0, 1.5)
                edge = 1.0 / (1.0 + np.exp(-(rel - float(t_edge_threshold)) * float(t_edge_sharpness)))
                p = np.clip((0.20 + 0.80 * np.clip(strength, 0.0, 1.0)) * edge, 0.0, 1.0)
                keep = rng.random(eff.shape) < p
                keep |= rel >= float(t_core_keep_ratio)
                eff_t = (eff * keep.astype(np.float32)).astype(np.float32)
                if eff_t.sum() <= 1e-8:
                    eff_t = eff.copy()
            else:
                eff_t = eff
            _place_kernel_add(t_num, eff_t, y, x, pk * frame_id)
            _place_kernel_add(t_den, eff_t, y, x, pk)

    ratio = peak / (tar.max() + 1e-6)
    tar = ratio * tar
    out = tar + float(background)
    if shot_noise:
        out = np.clip(out, 0, None)
        out = rng.poisson(out).astype(np.float32)
    if read_noise > 0:
        out = out + rng.normal(0.0, float(read_noise), size=out.shape).astype(np.float32)

    if not return_time_proj:
        return out, tar

    t_proj = t_num / (t_den + 1e-6)
    t_proj = np.clip(t_proj, 0.0, 1.0).astype(np.float32)
    if t_uncertainty and t_lowconf_randomize:
        conf = t_den / (float(t_den.max()) + 1e-12)
        low = conf < float(t_lowconf_ratio)
        if np.any(low):
            if n_frames_i > 1:
                ridx = rng.integers(0, n_frames_i, size=int(low.sum()))
                rvals = ridx.astype(np.float32) / float(n_frames_i - 1)
            else:
                rvals = np.zeros(int(low.sum()), dtype=np.float32)
            t_proj[low] = rvals
    return out, tar, t_proj


def _quantize_t_to_global_idx(t: np.ndarray, n_local: int, n_global: int) -> np.ndarray:
    n_local = int(max(n_local, 1))
    n_global = int(max(n_global, 1))
    if n_global <= 1:
        return np.zeros_like(t, dtype=np.int32)
    if n_local <= 1:
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
    wmax = np.maximum(w1, w2)

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
    prob = hist / hist.sum()

    idx_out = idx_fg.copy()
    if int(low.sum()) > 0:
        idx_out[low] = rng.choice(n_bg, size=int(low.sum()), p=prob)

    t_u8 = np.round(idx_out.astype(np.float32) / float(n_bg - 1) * 255.0).astype(np.uint8)
    return t_u8


def order_quad_clockwise(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    i0 = np.argmin(pts[:, 0] + pts[:, 1])
    pts = np.roll(pts, -i0, axis=0)
    return pts


def global_obb_from_projection_image(img: np.ndarray, thr: float, min_pixels: int = 4):
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
    if np.cross(v1, v2) < 0:
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


def polygon_area(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def to_yolo_obb_line(obb, w, h):
    if obb[0] < 0:
        return None
    pts = np.asarray(obb, dtype=np.float32).reshape(4, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, w)
    pts[:, 1] = np.clip(pts[:, 1], 0, h)
    pts = order_quad_clockwise(pts)
    if polygon_area(pts) < 2.0:
        return None
    pts[:, 0] /= float(w)
    pts[:, 1] /= float(h)
    vals = np.clip(pts.reshape(-1), 0.0, 1.0)
    return "0 " + " ".join([f"{v:.6f}" for v in vals])


def trunc_img(img, ratio=(0.5, 99.5)):
    vmin, vmax = np.percentile(img, ratio)
    img = np.clip(img, vmin, vmax)
    out = (img - vmin) / (vmax - vmin + 1e-8) * 255
    return out.astype(np.uint8)


def build_dataset():
    path_dirs = [
        "/mnt/e/Imgs/03-TJStars/150ms-2k2k/decode/",
        "/mnt/e/Imgs/03-TJStars/3000ms-4k4k/decode/",
        "/mnt/e/Imgs/03-TJStars/TJ2/decode/",
    ]
    root_dir = Path("dataset/obb_xt_clean_ut1")
    imgs_dir_x = root_dir / "imgs_x"
    imgs_dir_t = root_dir / "imgs_t"
    label_file = root_dir / "labels_obb.txt"
    patch_size = 1024
    stride = 512
    start_idx = 1
    flag_bkg = True
    train_ratio = 0.8
    seed = 42
    class_names = ["line"]
    ratio_x = (0.01, 99.99) if flag_bkg else (0.5, 99.5)

    for p in [imgs_dir_x, imgs_dir_t]:
        if p.exists():
            for q in p.glob("*"):
                q.unlink()
        p.mkdir(parents=True, exist_ok=True)

    meds = []
    for path_dir in path_dirs:
        aligns = []
        paths = sorted(glob.glob(path_dir + "*.tif"))
        for path in tqdm(paths, desc=f"Build median mask: {Path(path_dir).name}"):
            aligns.append(cv2.imread(path, -1))
        if flag_bkg:
            med = np.median(np.array(aligns), axis=0)
            bkg = sep.Background(np.ascontiguousarray(med, np.float32))
            bkg_img = np.array(bkg)
            _, ext_map = sep.extract(med - bkg_img, 5, err=bkg.globalrms, deblend_cont=1, segmentation_map=True)
            meds.append(ext_map)

    idx = start_idx
    content = []
    for num_dir, path_dir in enumerate(tqdm(path_dirs, desc="Simulating imgs")):
        mask = meds[num_dir] > 0 if flag_bkg else None
        paths = sorted(glob.glob(path_dir + "*.tif"))
        for path in paths:
            ori = cv2.imread(path, -1)
            if ori is None:
                continue
            ori = ori.astype(np.float32)
            h, w = ori.shape
            nw = (w - patch_size) // stride + 1
            nh = (h - patch_size) // stride + 1
            for i in range(nh):
                for j in range(nw):
                    x1, y1 = j * stride, i * stride
                    x2, y2 = x1 + patch_size, y1 + patch_size
                    img = ori[y1:y2, x1:x2].copy()
                    if flag_bkg:
                        mask_roi = mask[y1:y2, x1:x2]
                        img[mask_roi] = np.median(img)
                    try:
                        rng = np.random.default_rng(idx)
                        exp = max(rng.normal(1, 0.3), 0.05)
                        n_pt = int(rng.integers(5, 30))
                        peak_pt = (
                            max(rng.normal(0.1, 0.03), 0.001) * img.max()
                            if flag_bkg
                            else max(rng.normal(1, 0.5), 0.1) * img.std()
                        )
                        img_pt, tar_pt, t_pt = simulate_trajectory_projection_with_interval(
                            seed=idx,
                            img_shape=img.shape,
                            n_frames=n_pt,
                            speed_px_s=rng.integers(1, 5),
                            exposure_s=exp,
                            frame_interval_s=exp * rng.integers(2, 15),
                            radius_px=max(rng.normal(2, 0.7), 0.5),
                            peak=peak_pt,
                            angle_deg=rng.uniform(-90, 90),
                            background=0,
                            return_time_proj=True,
                        )

                        rng2 = np.random.default_rng(idx + 1)
                        n_line = int(rng2.integers(5, 30))
                        peak_line = (
                            max(rng2.normal(0.1, 0.03), 0.001) * img.max()
                            if flag_bkg
                            else max(rng2.normal(0.5, 0.5), 0.1) * img.std()
                        )
                        img_line, tar_line, t_line = simulate_trajectory_projection_with_interval(
                            seed=idx + 1,
                            img_shape=img.shape,
                            n_frames=n_line,
                            speed_px_s=rng2.integers(3, 23),
                            exposure_s=exp,
                            frame_interval_s=exp * rng2.integers(2, 5),
                            radius_px=max(rng2.normal(1, 0.3), 0.2),
                            peak=peak_line,
                            angle_deg=rng2.uniform(-90, 90),
                            background=0,
                            return_time_proj=True,
                        )

                        syn_x = (img + img_pt + img_line).astype(np.float32)
                        syn_t_u8 = compose_frame_index_projection_unified(
                            t_pt,
                            tar_pt,
                            t_line,
                            tar_line,
                            n_pt,
                            n_line,
                            rng=np.random.default_rng(idx + 11),
                            conf_thr=0.10,
                            hist_energy_thr=0.05,
                        )

                        row = [idx]
                        for tar in [tar_line, tar_pt]:
                            th = np.percentile(tar, 99.9)
                            obb = global_obb_from_projection_image(tar, th)
                            if obb is None:
                                row += [-1.0] * 8
                            else:
                                row += obb.reshape(-1).tolist()
                        content.append(row)

                        cv2.imwrite(str(imgs_dir_x / f"{idx:03d}.tif"), syn_x)
                        cv2.imwrite(str(imgs_dir_t / f"{idx:03d}.tif"), syn_t_u8)
                        idx += 1
                    except Exception as e:  # pragma: no cover
                        print(f"Failed idx={idx}: {e}")

    with label_file.open("w", encoding="utf-8") as f:
        for row in content:
            vals = ",".join([f"{row[0]:03d}"] + [f"{v:.6f}" for v in row[1:]])
            f.write(vals + "\n")

    print(f"UT1 sim done: images={len(content)} X={imgs_dir_x} T={imgs_dir_t} labels={label_file}")

    # Convert to YOLO-OBB layout
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images_x = sorted([p for p in imgs_dir_x.iterdir() if p.is_file() and p.suffix.lower() in exts])

    labels_by_idx = {}
    with label_file.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 17:
                raise ValueError(f"Bad OBB label format at line {ln}: got {len(parts)}")
            idx_i = int(parts[0])
            vals = list(map(float, parts[1:]))
            labels_by_idx[idx_i] = (vals[:8], vals[8:16])

    paired = []
    for img_x_path in images_x:
        img_idx = int(img_x_path.stem)
        t_path = imgs_dir_t / f"{img_x_path.stem}{img_x_path.suffix}"
        if img_idx not in labels_by_idx or not t_path.exists():
            continue
        paired.append((img_x_path, t_path, labels_by_idx[img_idx]))

    for split in ["train", "val"]:
        (root_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (root_dir / "images" / f"{split}_t").mkdir(parents=True, exist_ok=True)
        (root_dir / "images" / f"{split}_xt").mkdir(parents=True, exist_ok=True)
        (root_dir / "images" / f"{split}_xt_sc").mkdir(parents=True, exist_ok=True)
        (root_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root_dir / "labels" / f"{split}_xt").mkdir(parents=True, exist_ok=True)
        (root_dir / "labels" / f"{split}_xt_sc").mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rng.shuffle(paired)
    train_count = int(len(paired) * train_ratio)
    train_set = paired[:train_count]
    val_set = paired[train_count:]

    def write_split(rows, split):
        for img_x_path, t_path, (obb1, obb2) in tqdm(rows, desc=f"writing {split}"):
            img_x = cv2.imread(str(img_x_path), -1)
            img_t = cv2.imread(str(t_path), -1)
            if img_x is None or img_t is None:
                continue
            h, w = img_x.shape[:2]
            img_x8 = trunc_img(img_x, ratio_x)
            img_t8 = img_t.astype(np.uint8)
            tau = img_t8.astype(np.float32) / 255.0
            sin8 = ((np.sin(np.pi * tau) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            cos8 = ((np.cos(np.pi * tau) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            img3 = np.stack([img_x8, img_x8, img_x8], axis=-1)
            img_xt = np.stack([img_x8, img_t8, img_x8], axis=-1)
            img_xt_sc = np.stack([img_x8, sin8, cos8], axis=-1)
            stem = img_x_path.stem

            cv2.imwrite(str(root_dir / "images" / split / f"{stem}.png"), img3)
            cv2.imwrite(str(root_dir / "images" / f"{split}_t" / f"{stem}.png"), img_t8)
            cv2.imwrite(str(root_dir / "images" / f"{split}_xt" / f"{stem}.png"), img_xt)
            cv2.imwrite(str(root_dir / "images" / f"{split}_xt_sc" / f"{stem}.png"), img_xt_sc)

            lines = []
            for obb in (obb1, obb2):
                line = to_yolo_obb_line(obb, w=w, h=h)
                if line is not None:
                    lines.append(line)
            dedup = []
            seen = set()
            for line in lines:
                if line not in seen:
                    dedup.append(line)
                    seen.add(line)

            for lbl_split in [split, f"{split}_xt", f"{split}_xt_sc"]:
                dst_lbl = root_dir / "labels" / lbl_split / f"{stem}.txt"
                with dst_lbl.open("w", encoding="utf-8") as f:
                    if dedup:
                        f.write("\n".join(dedup) + "\n")

    write_split(train_set, "train")
    write_split(val_set, "val")

    yamls = {
        "data_x.yaml": "images/train",
        "data_t.yaml": "images/train_t",
        "data_xt.yaml": "images/train_xt",
        "data_xt_sc.yaml": "images/train_xt_sc",
    }
    for name, train_path in yamls.items():
        val_path = train_path.replace("train", "val")
        txt = (
            f"path: {root_dir.as_posix()}\n"
            f"train: {train_path}\n"
            f"val: {val_path}\n"
            "names:\n"
            f"  0: {class_names[0]}\n"
        )
        (root_dir / name).write_text(txt, encoding="utf-8")

    print(
        {
            "root": str(root_dir),
            "total": len(paired),
            "train": len(train_set),
            "val": len(val_set),
            "data_xt_sc": str(root_dir / "data_xt_sc.yaml"),
        }
    )


if __name__ == "__main__":
    build_dataset()
