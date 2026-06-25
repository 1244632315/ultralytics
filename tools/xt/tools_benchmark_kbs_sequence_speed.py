from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import sep
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = REPO_ROOT / "compares" / "MSAMNet-master"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

from ultralytics import YOLO  # noqa: E402
from model.CSAUNet import CSAUNet  # noqa: E402
from model.DNANet import DNANet, Res_CBAM_block  # noqa: E402
from model.DnTNet import DnTNet  # noqa: E402
from model.MSAMNet import AAFE, CBAM, Connection, CoordAtt, MSAMNet, SE  # noqa: E402
from model.utils import load_param  # noqa: E402
from tools.xt.tools_build_msod_stack_xt_sc import encode_xt_sc  # noqa: E402
from tools.xt.tools_eval_kbs_dnt_seg import normalize_frame, percentile_rescale  # noqa: E402


FUSION_BLOCKS = {
    "CBAM": CBAM,
    "AAFE": AAFE,
    "CA": CoordAtt,
    "SE": SE,
    "None": Connection,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark whole-sequence processing speed on KBS test sequences.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+")
    parser.add_argument("--test-json", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "test.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--t-frame", type=int, default=3)
    parser.add_argument("--input-size", type=int, default=0, help="0 means native resolution.")
    parser.add_argument("--rescale-percentile", type=float, nargs=2, default=(1.0, 99.0))
    parser.add_argument("--x-percentile", type=float, nargs=2, default=(0.01, 99.99))
    parser.add_argument("--star-threshold", type=float, default=1.5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--warmup-seqs", type=int, default=1)
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "runs" / "kbs_eval")
    parser.add_argument("--run-name", type=str, default="kbs_sequence_speed_benchmark")
    return parser.parse_args()


def load_seq_ids(test_json: Path) -> list[str]:
    ids = json.loads(test_json.read_text(encoding="utf-8"))
    return [f"{int(x):03d}" for x in ids]


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile_rescale_u8(img: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    img = percentile_rescale(img, pmin, pmax)
    return np.clip(img, 0, 255).astype(np.uint8)


def trunc_img(img: np.ndarray, ratio: tuple[float, float]) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    vmin, vmax = np.percentile(img, ratio)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8) * 255.0
    return img.astype(np.uint8)


def load_sequence_raw_frames(dataset_root: Path, seq_id: str, input_size: int) -> list[np.ndarray]:
    frames = []
    for path in sorted((dataset_root / f"{seq_id}_img").glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if input_size > 0 and img.shape[:2] != (input_size, input_size):
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_AREA)
        frames.append(img.astype(np.float32))
    if not frames:
        raise FileNotFoundError(f"No frames for {seq_id}")
    return frames


def build_notebook_aligned_xt(
    frames: list[np.ndarray],
    x_percentile: tuple[float, float],
    star_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return x_u8, t_u8, xt_sc


def build_compare_model(name: str, weights: Path, device: torch.device, t_frame: int) -> torch.nn.Module:
    if name == "CSAUNet":
        model = CSAUNet(input_channels=t_frame)
    elif name == "DNANet":
        nb_filter, num_blocks = load_param("three", "resnet_18")
        model = DNANet(
            num_classes=1,
            input_channels=t_frame,
            block=Res_CBAM_block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
            deep_supervision="False",
        )
    elif name == "DnTNet":
        model = DnTNet(seq_length=t_frame)
    elif name == "MSAMNet":
        model = MSAMNet(frame_length=t_frame, fusionBlock=FUSION_BLOCKS["AAFE"])
    else:
        raise KeyError(name)
    checkpoint = torch.load(weights.expanduser().resolve(), map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Unsupported checkpoint format: {weights}")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_logits(output: torch.Tensor | list | tuple) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        for item in reversed(output):
            if isinstance(item, torch.Tensor):
                return item
        raise TypeError("model output does not contain a tensor")
    return output


def benchmark_compare_sequence(
    model: torch.nn.Module,
    frames_raw: list[np.ndarray],
    device: torch.device,
    t_frame: int,
    rescale_percentile_pair: tuple[float, float],
) -> dict[str, float]:
    start = time.perf_counter()
    num_frames = len(frames_raw)
    with torch.no_grad():
        for frame_idx in range(num_frames):
            frames = []
            for offset in range(t_frame):
                hist_idx = max(frame_idx - offset, 0)
                img = percentile_rescale(frames_raw[hist_idx], rescale_percentile_pair[0], rescale_percentile_pair[1])
                img = normalize_frame(img)
                frames.append(img.astype(np.float32))
            x = np.stack(frames[::-1], axis=0)
            x = torch.from_numpy(x).unsqueeze(0).to(device)
            sync_if_cuda(device)
            _ = get_logits(model(x))
            sync_if_cuda(device)
    total = time.perf_counter() - start
    return {
        "num_frames": int(num_frames),
        "total_seconds": float(total),
        "seconds_per_sequence": float(total),
        "seconds_per_frame": float(total / max(1, num_frames)),
        "fps": float(num_frames / max(total, 1e-12)),
    }


def benchmark_ours_sequence(
    model: YOLO,
    frames_raw: list[np.ndarray],
    device_str: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    x_percentile_pair: tuple[float, float],
    star_threshold: float,
    tmp_dir: Path,
    seq_id: str,
) -> dict[str, float]:
    t0 = time.perf_counter()
    x_u8, t_u8, xt_sc = build_notebook_aligned_xt(
        frames_raw,
        x_percentile=x_percentile_pair,
        star_threshold=star_threshold,
    )
    xt_gen = time.perf_counter() - t0

    inp_path = tmp_dir / f"{seq_id}.png"
    cv2.imwrite(str(inp_path), xt_sc)
    t1 = time.perf_counter()
    _ = model.predict(
        source=str(inp_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device_str,
        verbose=False,
        save=False,
    )
    infer = time.perf_counter() - t1
    total = xt_gen + infer
    num_frames = len(frames_raw)
    return {
        "num_frames": int(num_frames),
        "xt_sc_generation_seconds": float(xt_gen),
        "detector_inference_seconds": float(infer),
        "total_seconds": float(total),
        "seconds_per_sequence": float(total),
        "seconds_per_frame_equivalent": float(total / max(1, num_frames)),
        "fps_equivalent": float(num_frames / max(total, 1e-12)),
        "x_path_shape": [int(x_u8.shape[0]), int(x_u8.shape[1])],
        "t_path_shape": [int(t_u8.shape[0]), int(t_u8.shape[1])],
    }


def average_rows(rows: list[dict[str, float]], extra_keys: list[str] | None = None) -> dict[str, float]:
    if not rows:
        return {}
    keys = [
        "num_frames",
        "total_seconds",
        "seconds_per_sequence",
        "seconds_per_frame",
        "fps",
    ]
    if extra_keys:
        keys.extend(extra_keys)
    out = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r]
        if vals:
            out[key] = float(sum(vals) / len(vals))
    return out


def main() -> None:
    args = parse_args()
    want_cpu = args.device.lower().startswith("cpu")
    device = torch.device("cpu" if want_cpu or not torch.cuda.is_available() else args.device)
    seq_ids = load_seq_ids(args.test_json)

    compare_weights = {
        "CSAUNet": REPO_ROOT / "compares" / "MSAMNet-master" / "result" / "MSOD_RAWBG_CSAUNet_17_04_2026_11_18_01_wDS" / "mIoU_best_CSAUNet_MSOD_RAWBG_epoch.pth.tar",
        "DNANet": REPO_ROOT / "compares" / "MSAMNet-master" / "result" / "MSOD_RAWBG_DNANet_17_04_2026_14_32_33_wDS" / "mIoU_best_DNANet_MSOD_RAWBG_epoch.pth.tar",
        "DnTNet": REPO_ROOT / "compares" / "MSAMNet-master" / "result" / "MSOD_RAWBG_DnTNet_18_04_2026_19_38_29_wDS" / "mIoU_best_DnTNet_MSOD_RAWBG_epoch.pth.tar",
        "MSAMNet": REPO_ROOT / "compares" / "MSAMNet-master" / "result" / "MSOD_RAWBG_MSAMNet_18_04_2026_08_35_33_wDS" / "mIoU_best_MSAMNet_MSOD_RAWBG_epoch.pth.tar",
    }
    ours_weights = REPO_ROOT / "runs" / "obb" / "cmp_xt_sc_endpoint_v1_gpu_full" / "weights" / "best.pt"

    compare_models = {
        name: build_compare_model(name, weights, device, args.t_frame) for name, weights in compare_weights.items()
    }
    ours_model = YOLO(str(ours_weights.expanduser().resolve()))

    out_root = args.save_dir / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_root / "tmp_inputs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # warmup
    for seq_id in seq_ids[: max(0, int(args.warmup_seqs))]:
        frames_raw = load_sequence_raw_frames(args.dataset_root, seq_id, args.input_size)
        _ = benchmark_compare_sequence(
            next(iter(compare_models.values())),
            frames_raw,
            device,
            args.t_frame,
            tuple(args.rescale_percentile),
        )
        _ = benchmark_ours_sequence(
            ours_model,
            frames_raw,
            args.device,
            args.imgsz,
            args.conf,
            args.iou,
            args.max_det,
            tuple(args.x_percentile),
            args.star_threshold,
            tmp_dir,
            seq_id=f"warmup_{seq_id}",
        )

    per_method: dict[str, list[dict[str, float]]] = {name: [] for name in compare_models}
    per_method["Proposed_XT_sc"] = []
    per_sequence = {}

    for seq_idx, seq_id in enumerate(seq_ids, start=1):
        frames_raw = load_sequence_raw_frames(args.dataset_root, seq_id, args.input_size)
        seq_row = {"num_frames": len(frames_raw), "methods": {}}
        for name, model in compare_models.items():
            row = benchmark_compare_sequence(
                model,
                frames_raw,
                device,
                args.t_frame,
                tuple(args.rescale_percentile),
            )
            per_method[name].append(row)
            seq_row["methods"][name] = row
        ours_row = benchmark_ours_sequence(
            ours_model,
            frames_raw,
            args.device,
            args.imgsz,
            args.conf,
            args.iou,
            args.max_det,
            tuple(args.x_percentile),
            args.star_threshold,
            tmp_dir,
            seq_id=seq_id,
        )
        per_method["Proposed_XT_sc"].append(ours_row)
        seq_row["methods"]["Proposed_XT_sc"] = ours_row
        per_sequence[seq_id] = seq_row
        print(f"[{seq_idx}/{len(seq_ids)}] done {seq_id}", flush=True)

    summary = {
        "dataset_root": str(args.dataset_root),
        "test_json": str(args.test_json),
        "device": str(device),
        "num_sequences": len(seq_ids),
        "sequence_ids": seq_ids,
        "benchmark_protocol": {
            "segmentation_methods": "whole-sequence end-to-end time including frame loading, percentile(1,99), normalization, and per-frame inference",
            "proposed_method": "whole-sequence end-to-end time including frame loading, XT_sc generation, and one detector inference",
        },
        "methods": {
            name: {
                "average": average_rows(
                    rows,
                    extra_keys=["xt_sc_generation_seconds", "detector_inference_seconds", "seconds_per_frame_equivalent", "fps_equivalent"],
                ),
                "per_sequence": {seq_id: per_sequence[seq_id]["methods"][name] for seq_id in seq_ids},
            }
            for name, rows in per_method.items()
        },
    }

    out_json = out_root / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    for name, payload in summary["methods"].items():
        avg = payload["average"]
        if name == "Proposed_XT_sc":
            lines.append(
                f"{name}: total/seq={avg['seconds_per_sequence']:.4f}s, xt_sc={avg['xt_sc_generation_seconds']:.4f}s, "
                f"infer={avg['detector_inference_seconds']:.4f}s, fps_eq={avg['fps_equivalent']:.2f}"
            )
        else:
            lines.append(
                f"{name}: total/seq={avg['seconds_per_sequence']:.4f}s, per-frame={avg['seconds_per_frame']:.4f}s, fps={avg['fps']:.2f}"
            )
    report_text = "\n".join(lines)
    (out_root / "report.txt").write_text(report_text + "\n", encoding="utf-8")
    print(report_text)
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
