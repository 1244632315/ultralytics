from __future__ import annotations

import argparse
import json
import random
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

from model.CSAUNet import CSAUNet  # noqa: E402
from model.DNANet import DNANet, Res_CBAM_block  # noqa: E402
from model.DnTNet import DnTNet  # noqa: E402
from model.MSAMNet import AAFE, CBAM, Connection, CoordAtt, MSAMNet, SE  # noqa: E402
from model.dataloader import MSODataset  # noqa: E402
from model.utils import load_param  # noqa: E402


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
        description="Visualize compare-model outputs on projected sequence frames by stacking per-frame masks."
    )
    parser.add_argument("--model", choices=["CSAUNet", "DNANet", "DnTNet", "MSAMNet"], default="CSAUNet")
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT
        / "compares"
        / "MSAMNet-master"
        / "result"
        / "MSOD_CSAUNet_16_04_2026_20_46_23_wDS"
        / "mIoU_best_CSAUNet_MSOD_epoch.pth.tar",
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "xt_seq_msod_x" / "MSOD_RAWBG")
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
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--component-min-pixels", type=int, default=4)
    parser.add_argument("--component-dilate", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "xt_compare_vis")
    return parser.parse_args()


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

    checkpoint = torch.load(args.weights.expanduser().resolve(), map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Unsupported checkpoint format: {args.weights}")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_input_tensor(split_dir: Path, seq_id: str, frame_id: int, t_frame: int, suffix: str) -> torch.Tensor:
    img_dir = split_dir / "images" / seq_id
    image_data = []

    cur_img = cv2.imread(str(img_dir / f"{frame_id:05d}{suffix}"), cv2.IMREAD_UNCHANGED)
    image_data.append(MSODataset._normalize_frame(cur_img))

    for offset in range(1, t_frame):
        his_path = img_dir / f"{frame_id - offset:05d}{suffix}"
        if not his_path.exists():
            for i in range(1, t_frame):
                candidate = img_dir / f"{frame_id - offset + i:05d}{suffix}"
                if candidate.exists():
                    his_path = candidate
                    break
        his_img = cv2.imread(str(his_path), cv2.IMREAD_UNCHANGED)
        image_data.append(MSODataset._normalize_frame(his_img))

    arr = np.array(image_data[::-1], dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def order_quad_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(ang)]
    i0 = np.argmin(pts[:, 0] + pts[:, 1])
    return np.roll(pts, -i0, axis=0)


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
        row = frame_meta.get((split, seq_id, f"{frame_id:05d}"))
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
    trajectory_masks: list[tuple[str, np.ndarray]],
    min_pixels: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    label_map = np.zeros(pred_stack_bin.shape, dtype=np.uint8)
    for idx, (_, inst_mask) in enumerate(trajectory_masks, start=1):
        label_map[(label_map == 0) & (inst_mask > 0)] = idx

    pred_polys: list[np.ndarray] = []
    traj_union = np.zeros(pred_stack_bin.shape, dtype=np.uint8)
    for idx, (_, _) in enumerate(trajectory_masks, start=1):
        inst_pred = ((pred_stack_bin > 0) & (label_map == idx)).astype(np.uint8)
        obb = global_obb_from_binary_mask(inst_pred, min_pixels=min_pixels)
        if obb is not None:
            pred_polys.append(obb)
            traj_union = np.maximum(traj_union, inst_pred)

    fp_mask = ((pred_stack_bin > 0) & (label_map == 0)).astype(np.uint8)
    fp_polys = extract_component_obbs(fp_mask, min_pixels=min_pixels, dilate_iter=0)
    pred_polys.extend(fp_polys)
    pred_union = np.maximum(traj_union, fp_mask)
    return pred_polys, pred_union


def scale_to_uint8_sum(stack: np.ndarray) -> np.ndarray:
    if stack.size == 0:
        return stack.astype(np.uint8)
    vmin, vmax = np.percentile(stack, (0.01, 99.99))
    if vmax - vmin < 1e-6:
        return np.zeros_like(stack, dtype=np.uint8)
    out = np.clip(stack, vmin, vmax)
    out = (out - vmin) / (vmax - vmin) * 255.0
    return out.astype(np.uint8)


def overlay_mask(gray: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mask.max() <= 0:
        return base
    overlay = base.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(base, 0.72, overlay, 0.28, 0.0)


def overlay_compare(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gt_bool = gt > 0
    pred_bool = pred > 0
    tp = gt_bool & pred_bool
    fn = gt_bool & (~pred_bool)
    fp = pred_bool & (~gt_bool)
    out[tp] = (0, 255, 255)
    out[fn] = (0, 255, 0)
    out[fp] = (0, 0, 255)
    return cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.65, out, 0.35, 0.0)


def draw_obbs(gray: np.ndarray, gt_obbs: list[np.ndarray], pred_obbs: list[np.ndarray]) -> np.ndarray:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for obb in gt_obbs:
        pts = np.round(np.asarray(obb, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    for obb in pred_obbs:
        pts = np.round(np.asarray(obb, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "GT=green Pred=red", (10, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def label_tile(img: np.ndarray, text: str) -> np.ndarray:
    tile = img.copy()
    cv2.putText(tile, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return tile


def build_projection_panel(
    model: torch.nn.Module,
    args: argparse.Namespace,
    seq_id: str,
    split_dir: Path,
    frame_meta: dict[tuple[str, str, str], dict],
    scale_xy: tuple[float, float],
    device: torch.device,
) -> np.ndarray:
    img_dir = split_dir / "images" / seq_id
    frame_ids = sorted(int(p.stem) for p in img_dir.glob(f"*{args.suffix}"))
    img_stack = None
    pred_stack = None

    with torch.no_grad():
        for frame_id in frame_ids:
            frame_name = f"{frame_id:05d}"
            gray = read_gray(img_dir / f"{frame_name}{args.suffix}")
            inp = load_input_tensor(split_dir, seq_id, frame_id, args.t_frame, args.suffix).to(device)
            logits = model(inp)[0, 0].detach().cpu().numpy()
            pred_mask = (logits > args.pred_thr).astype(np.uint8)

            if img_stack is None:
                shape = gray.shape
                img_stack = np.zeros(shape, dtype=np.float32)
                pred_stack = np.zeros(shape, dtype=np.uint8)

            img_stack += gray.astype(np.float32)
            pred_stack = np.maximum(pred_stack, pred_mask.astype(np.uint8))

    if img_stack is None or pred_stack is None:
        raise ValueError(f"No frames found for sequence {seq_id}")

    proj_gray = scale_to_uint8_sum(img_stack)
    trajectory_masks, gt_stack = build_trajectory_instance_masks(
        frame_meta, args.split, seq_id, frame_ids, pred_stack.shape, scale_xy, args.component_dilate
    )
    gt_obbs = [
        obb
        for _, inst_mask in trajectory_masks
        if (obb := global_obb_from_binary_mask(inst_mask, min_pixels=args.component_min_pixels)) is not None
    ]
    pred_obbs, pred_union = aggregate_instance_predictions(pred_stack, trajectory_masks, args.component_min_pixels)

    panel0 = label_tile(cv2.cvtColor(proj_gray, cv2.COLOR_GRAY2BGR), f"Seq {seq_id} Projected X")
    panel1 = label_tile(overlay_mask(proj_gray, gt_stack, (0, 255, 0)), f"GT Stack Mask ({len(gt_obbs)})")
    panel2 = label_tile(overlay_mask(proj_gray, pred_union, (0, 0, 255)), f"Pred Stack Mask ({len(pred_obbs)})")
    panel3 = label_tile(draw_obbs(proj_gray, gt_obbs, pred_obbs), "Projected OBB")
    panel4 = label_tile(overlay_compare(proj_gray, gt_stack, pred_union), "Projected TP/FN/FP")
    panel5 = label_tile(
        draw_obbs(scale_to_uint8_sum(img_stack), gt_obbs, []),
        f"Frames={len(frame_ids)} T_frame={args.t_frame} GTtraj={len(trajectory_masks)}",
    )

    top = np.hstack([panel0, panel1, panel2])
    bottom = np.hstack([panel3, panel4, panel5])
    return np.vstack([top, bottom])


def main() -> None:
    args = parse_args()
    split_dir = args.dataset_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split dir: {split_dir}")
    frame_meta = load_metadata_by_frame(args.dataset_root)
    if not frame_meta:
        raise FileNotFoundError(f"Missing metadata.jsonl under {args.dataset_root}")
    scale_xy = load_dataset_scale(args.dataset_root)

    want_cpu = args.device.lower().startswith("cpu")
    device = torch.device("cpu" if want_cpu or not torch.cuda.is_available() else args.device)
    model = build_model(args, device)

    seq_dirs = sorted(p.name for p in (split_dir / "images").iterdir() if p.is_dir())
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence dirs found in {split_dir / 'images'}")
    rng = random.Random(args.seed)
    chosen = seq_dirs if len(seq_dirs) <= args.count else rng.sample(seq_dirs, args.count)

    out_dir = args.out_dir / f"{args.model}_{args.split}_stackproj"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seq_id in chosen:
        panel = build_projection_panel(model, args, seq_id, split_dir, frame_meta, scale_xy, device)
        dst = out_dir / f"{seq_id}.png"
        cv2.imwrite(str(dst), panel)
        print(dst)


if __name__ == "__main__":
    main()
