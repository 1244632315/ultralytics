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
        description="Build a stacked XT_sc dataset from MSOD sequence data using an estimated temporal projection."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/xt_seq_msod_x/MSOD_RAWBG"))
    parser.add_argument("--ref-stack-root", type=Path, default=Path("dataset/obb_xt_seq_msod_stack_rawbg"))
    parser.add_argument("--out-root", type=Path, default=Path("dataset/obb_xt_seq_msod_stack_rawbg_xt_sc"))
    parser.add_argument("--image-suffix", type=str, default=".tif")
    parser.add_argument("--stack-frames", type=int, default=0, help="0 means all frames in each sequence.")
    parser.add_argument("--image-stack-mode", choices=["sum", "max", "mean"], default="sum")
    parser.add_argument("--time-mode", choices=["argmax", "soft_quantized", "soft_palette"], default="soft_quantized")
    parser.add_argument("--time-bg-percentile", type=float, default=25.0)
    parser.add_argument("--time-conf-percentile", type=float, default=92.0)
    parser.add_argument("--time-dom-threshold", type=float, default=0.35)
    parser.add_argument("--time-lowconf-randomize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-levels", type=int, default=32)
    parser.add_argument("--time-seed", type=int, default=0)
    parser.add_argument("--time-target-root", type=Path, default=Path("dataset/obb_xt_clean/images"))
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


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


def ensure_out_root(out_root: Path, keep_existing: bool) -> None:
    if out_root.exists() and not keep_existing:
        shutil.rmtree(out_root)
    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels").mkdir(parents=True, exist_ok=True)


def build_time_palette(target_root: Path, levels: int) -> np.ndarray:
    levels = max(int(levels), 2)
    candidates = list(sorted((target_root / "train_t").glob("*.png")))[:128]
    candidates += list(sorted((target_root / "val_t").glob("*.png")))[:64]
    values = []
    for path in candidates:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        values.append(img.reshape(-1))

    if not values:
        return np.linspace(0.0, 255.0, levels, dtype=np.float32)

    merged = np.concatenate(values, axis=0).astype(np.float32)
    qs = np.linspace(0.0, 1.0, levels)
    palette = np.quantile(merged, qs).astype(np.float32)
    palette[0] = 0.0
    palette[-1] = 255.0
    return np.clip(np.round(palette), 0, 255).astype(np.float32)


def estimate_time_projection(
    frames: list[np.ndarray],
    seq_seed: int,
    mode: str,
    palette: np.ndarray,
    bg_percentile: float,
    conf_percentile: float,
    dom_threshold: float,
    randomize_lowconf: bool,
) -> np.ndarray:
    stack = np.stack(frames, axis=0).astype(np.float32)
    n_frames = stack.shape[0]
    if n_frames <= 1:
        return np.zeros(stack.shape[1:], dtype=np.uint8)

    bg = np.percentile(stack, bg_percentile, axis=0).astype(np.float32)
    evidence = np.clip(stack - bg[None, ...], 0.0, None)
    peak = evidence.max(axis=0)
    energy = evidence.sum(axis=0)
    denom = energy + 1e-6
    idx_hard = evidence.argmax(axis=0).astype(np.int32)

    energy_scale = float(np.percentile(energy, conf_percentile))
    if energy_scale <= 1e-6:
        energy_scale = float(energy.max()) if float(energy.max()) > 1e-6 else 1.0
    energy_conf = np.clip(energy / (energy_scale + 1e-6), 0.0, 1.0)
    dominance = np.clip(peak / denom, 0.0, 1.0)

    if mode == "argmax":
        idx_out = idx_hard.copy()
        high = energy_conf >= 0.20
        low = ~high
        if randomize_lowconf and int(low.sum()) > 0:
            hist_src = idx_hard[high]
            if hist_src.size < 64:
                hist_src = idx_hard.reshape(-1)
            hist = np.bincount(hist_src.astype(np.int32), minlength=n_frames).astype(np.float64)
            if hist.sum() <= 0:
                hist = np.ones(n_frames, dtype=np.float64)
            prob = hist / hist.sum()
            rng = np.random.default_rng(seq_seed)
            idx_out[low] = rng.choice(n_frames, size=int(low.sum()), p=prob)
        tau = idx_out.astype(np.float32) / float(max(n_frames - 1, 1))
        return np.round(tau * 255.0).astype(np.uint8)

    frame_axis = np.arange(n_frames, dtype=np.float32)[:, None, None]
    mu = (evidence * frame_axis).sum(axis=0) / denom
    tau = np.clip(mu / float(max(n_frames - 1, 1)), 0.0, 1.0)

    flat_tau = tau.reshape(-1)
    flat_conf = energy_conf.reshape(-1)
    flat_dom = dominance.reshape(-1)
    high = (flat_conf >= 0.20) & (flat_dom >= float(dom_threshold))
    t_out = np.empty(flat_tau.shape[0], dtype=np.uint8)
    rng = np.random.default_rng(seq_seed)

    palette_u8 = np.clip(np.round(palette), 0, 255).astype(np.uint8)
    if high.any():
        tau_high = flat_tau[high]
        if mode == "soft_palette":
            order = np.argsort(tau_high, kind="mergesort")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(order.size)
            q = ranks.astype(np.float32) / float(max(order.size - 1, 1))
            pal_idx = np.clip(np.round(q * float(len(palette_u8) - 1)), 0, len(palette_u8) - 1).astype(np.int32)
        else:
            pal_idx = np.clip(np.round(tau_high * float(len(palette_u8) - 1)), 0, len(palette_u8) - 1).astype(np.int32)
        t_high = palette_u8[pal_idx]
        t_out[high] = t_high
        hist_src = np.bincount(t_high.astype(np.int32), minlength=256).astype(np.float64)
    else:
        hist_src = np.bincount(palette_u8.astype(np.int32), minlength=256).astype(np.float64)

    low = ~high
    if low.any():
        if randomize_lowconf:
            if hist_src.sum() <= 0:
                hist_src = np.bincount(palette_u8.astype(np.int32), minlength=256).astype(np.float64)
            prob = hist_src / hist_src.sum()
            t_out[low] = rng.choice(256, size=int(low.sum()), p=prob).astype(np.uint8)
        else:
            idx_soft = np.clip(np.round(flat_tau[low] * float(len(palette_u8) - 1)), 0, len(palette_u8) - 1).astype(np.int32)
            t_out[low] = palette_u8[idx_soft]

    return t_out.reshape(stack.shape[1:])


def encode_xt_sc(x_u8: np.ndarray, t_u8: np.ndarray) -> np.ndarray:
    tau = t_u8.astype(np.float32) / 255.0
    sin8 = ((np.sin(np.pi * tau) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    cos8 = ((np.cos(np.pi * tau) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return np.stack([x_u8, sin8, cos8], axis=-1)


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


def maybe_link_labels(out_root: Path, ref_stack_root: Path) -> None:
    out_labels = out_root / "labels"
    ref_labels = ref_stack_root / "labels"
    if not ref_labels.exists():
        raise FileNotFoundError(f"Missing reference labels: {ref_labels}")
    for split in ["train", "val", "test"]:
        src = ref_labels / split
        dst = out_labels / split
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve(), target_is_directory=True)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    ref_stack_root = args.ref_stack_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")
    if not ref_stack_root.exists():
        raise FileNotFoundError(f"Missing reference stack root: {ref_stack_root}")

    ensure_out_root(out_root, keep_existing=args.keep_existing)
    maybe_link_labels(out_root, ref_stack_root)
    palette = build_time_palette(args.time_target_root.expanduser().resolve(), args.time_levels)

    metadata_rows = []
    summary = {
        "source_dataset_root": str(dataset_root),
        "reference_stack_root": str(ref_stack_root),
        "image_stack_mode": args.image_stack_mode,
        "time_mode": args.time_mode,
        "time_bg_percentile": args.time_bg_percentile,
        "time_conf_percentile": args.time_conf_percentile,
        "time_dom_threshold": args.time_dom_threshold,
        "time_lowconf_randomize": bool(args.time_lowconf_randomize),
        "time_levels": int(args.time_levels),
        "time_palette": palette.astype(int).tolist(),
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        split_root = dataset_root / split / "images"
        seq_ids = sorted(p.name for p in split_root.iterdir() if p.is_dir())
        count = 0

        for seq_id in tqdm(seq_ids, desc=f"Build {split} xt_sc"):
            seq_dir = split_root / seq_id
            frame_ids = collect_frame_ids(seq_dir, args.image_suffix)
            if not frame_ids:
                continue
            if args.stack_frames > 0:
                frame_ids = frame_ids[-args.stack_frames :]

            frames = [
                np.array(Image.open(seq_dir / f"{frame_id:05d}{args.image_suffix}"), dtype=np.float32) for frame_id in frame_ids
            ]
            x_u8 = stack_image(frames, mode=args.image_stack_mode)
            seq_seed = args.time_seed + int(seq_id)
            t_u8 = estimate_time_projection(
                frames,
                seq_seed=seq_seed,
                mode=args.time_mode,
                palette=palette,
                bg_percentile=args.time_bg_percentile,
                conf_percentile=args.time_conf_percentile,
                dom_threshold=args.time_dom_threshold,
                randomize_lowconf=bool(args.time_lowconf_randomize),
            )
            xt_sc = encode_xt_sc(x_u8, t_u8)

            out_path = out_root / "images" / split / f"{seq_id}.png"
            cv2.imwrite(str(out_path), xt_sc)
            metadata_rows.append(
                {
                    "split": split,
                    "seq_id": seq_id,
                    "frame_ids": frame_ids,
                    "image_path": str(out_path),
                    "t_seed": seq_seed,
                }
            )
            count += 1

        summary["splits"][split] = {"images": count}

    write_data_yaml(out_root)
    (out_root / "metadata.json").write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out_root": str(out_root), **summary}, indent=2))


if __name__ == "__main__":
    main()
