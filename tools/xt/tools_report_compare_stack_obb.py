from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"
REPORTS_ROOT = RUNS_ROOT / "obb" / "reports"
COMPARE_RESULT_ROOT = REPO_ROOT / "compares" / "MSAMNet-master" / "result"
BUILD_STACK_SCRIPT = REPO_ROOT / "tools" / "xt" / "tools_build_msod_stack_obb.py"
EVAL_COMPARE_SCRIPT = REPO_ROOT / "tools" / "xt" / "tools_eval_compare_stack_obb.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fair stacked-OBB comparison between sequence segmentation baselines and the XT OBB model."
    )
    parser.add_argument("--seq-dataset-root", type=Path, default=REPO_ROOT / "dataset" / "xt_seq_msod_x")
    parser.add_argument("--dataset-name", type=str, default="MSOD_RAWBG")
    parser.add_argument("--stack-dataset-root", type=Path, default=REPO_ROOT / "dataset" / "obb_xt_seq_msod_stack_rawbg")
    parser.add_argument("--yolo-stack-dataset-root", type=Path, default=REPO_ROOT / "dataset" / "obb_xt_seq_msod_stack_rawbg_rgb")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--models", nargs="+", default=["CSAUNet", "DNANet", "DnTNet", "MSAMNet"])
    parser.add_argument("--ckpt-metric", type=str, default="mIoU")
    parser.add_argument("--t-frame", type=int, default=3)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--suffix", type=str, default=".tif")
    parser.add_argument("--fusionblock", type=str, default="AAFE")
    parser.add_argument("--compare-device", type=str, default="cuda:0")
    parser.add_argument("--pred-thr", type=float, default=0.0)
    parser.add_argument("--stack-mode", choices=["max", "sum"], default="max")
    parser.add_argument("--component-min-pixels", type=int, default=4)
    parser.add_argument("--component-dilate", type=int, default=1)
    parser.add_argument("--pred-agg-mode", choices=["components", "gt_partition"], default="components")
    parser.add_argument("--score-reduction", choices=["max", "mean"], default="max")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--save-labels", action="store_true")
    parser.add_argument("--save-stacks", action="store_true")
    parser.add_argument("--rebuild-stack-dataset", action="store_true")
    parser.add_argument(
        "--our-best-pt",
        type=Path,
        default=REPO_ROOT / "runs" / "obb" / "cmp_xt_sc_endpoint_v1_gpu_full" / "weights" / "best.pt",
    )
    parser.add_argument("--yolo-device", type=str, default="0")
    parser.add_argument("--yolo-imgsz", type=int, default=512)
    parser.add_argument("--yolo-batch", type=int, default=8)
    parser.add_argument("--compare-save-root", type=Path, default=RUNS_ROOT / "xt_compare_stack_obb_fair")
    parser.add_argument("--yolo-save-root", type=Path, default=RUNS_ROOT / "xt_compare_stack_obb_fair")
    return parser.parse_args()


def ensure_stack_dataset(args: argparse.Namespace) -> Path:
    data_yaml = args.stack_dataset_root / "data.yaml"
    if data_yaml.exists() and not args.rebuild_stack_dataset:
        return data_yaml

    seq_dataset_dir = args.seq_dataset_root / args.dataset_name
    if not seq_dataset_dir.exists():
        raise FileNotFoundError(f"Missing sequence dataset: {seq_dataset_dir}")

    cmd = [
        sys.executable,
        str(BUILD_STACK_SCRIPT),
        "--dataset-root",
        str(seq_dataset_dir),
        "--out-root",
        str(args.stack_dataset_root),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing stacked dataset yaml after build: {data_yaml}")
    return data_yaml


def dataset_is_single_channel(dataset_root: Path, split: str) -> bool:
    image_dir = dataset_root / "images" / split
    first_image = next(iter(sorted(image_dir.glob("*"))), None)
    if first_image is None:
        raise FileNotFoundError(f"No images found under {image_dir}")
    with Image.open(first_image) as img:
        if img.mode in {"RGB", "RGBA"}:
            return False
        return len(img.getbands()) == 1


def write_data_yaml(dataset_root: Path) -> None:
    text = "\n".join(
        [
            f"path: {dataset_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: line",
            "",
        ]
    )
    (dataset_root / "data.yaml").write_text(text, encoding="utf-8")


def ensure_rgb_stack_dataset(args: argparse.Namespace, source_data_yaml: Path) -> Path:
    source_root = source_data_yaml.parent
    if not dataset_is_single_channel(source_root, args.split):
        return source_data_yaml

    target_root = args.yolo_stack_dataset_root
    target_data_yaml = target_root / "data.yaml"
    if target_data_yaml.exists() and not args.rebuild_stack_dataset:
        return target_data_yaml

    if target_root.exists():
        shutil.rmtree(target_root)

    for split in ["train", "val", "test"]:
        src_img_dir = source_root / "images" / split
        if not src_img_dir.exists():
            continue
        dst_img_dir = target_root / "images" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        for src_path in sorted(src_img_dir.glob("*")):
            with Image.open(src_path) as img:
                img.convert("RGB").save(dst_img_dir / src_path.name)

    (target_root / "labels").mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        src_lbl_dir = source_root / "labels" / split
        if not src_lbl_dir.exists():
            continue
        dst_lbl_dir = target_root / "labels" / split
        if dst_lbl_dir.exists() or dst_lbl_dir.is_symlink():
            dst_lbl_dir.unlink()
        dst_lbl_dir.symlink_to(src_lbl_dir.resolve(), target_is_directory=True)

    for name in ["metadata.json", "summary.json"]:
        src_meta = source_root / name
        if src_meta.exists():
            shutil.copy2(src_meta, target_root / name)
    write_data_yaml(target_root)
    return target_data_yaml


def find_compare_weights(model: str, dataset_name: str, ckpt_metric: str) -> Path:
    pattern = f"{dataset_name}_{model}_*_wDS/{ckpt_metric}_best_{model}_{dataset_name}_epoch.pth.tar"
    matches = list(COMPARE_RESULT_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint matched pattern: {pattern}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def run_compare_eval(args: argparse.Namespace, model: str, weights: Path) -> dict:
    save_root = args.compare_save_root / args.dataset_name.lower()
    cmd = [
        sys.executable,
        str(EVAL_COMPARE_SCRIPT),
        "--model",
        model,
        "--weights",
        str(weights),
        "--dataset-root",
        str(args.seq_dataset_root),
        "--dataset-name",
        args.dataset_name,
        "--split",
        args.split,
        "--t-frame",
        str(args.t_frame),
        "--input-size",
        str(args.input_size),
        "--suffix",
        args.suffix,
        "--fusionblock",
        args.fusionblock,
        "--device",
        args.compare_device,
        "--pred-thr",
        str(args.pred_thr),
        "--stack-mode",
        args.stack_mode,
        "--component-min-pixels",
        str(args.component_min_pixels),
        "--component-dilate",
        str(args.component_dilate),
        "--pred-agg-mode",
        args.pred_agg_mode,
        "--score-reduction",
        args.score_reduction,
        "--save-dir",
        str(save_root),
    ]
    if args.max_sequences > 0:
        cmd.extend(["--max-sequences", str(args.max_sequences)])
    if args.save_labels:
        cmd.append("--save-labels")
    if args.save_stacks:
        cmd.append("--save-stacks")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    summary_path = save_root / f"{model}_{args.split}_stackobb" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing compare summary: {summary_path}")
    row = json.loads(summary_path.read_text(encoding="utf-8"))
    row["metric_source"] = "tools_eval_compare_stack_obb.py"
    row["checkpoint_selection"] = args.ckpt_metric
    return row


def run_yolo_eval(args: argparse.Namespace, data_yaml: Path) -> dict:
    best_pt = args.our_best_pt.expanduser().resolve()
    if not best_pt.exists():
        raise FileNotFoundError(f"Missing YOLO checkpoint: {best_pt}")

    model = YOLO(str(best_pt))
    val = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.yolo_imgsz,
        batch=args.yolo_batch,
        device=args.yolo_device,
        project=str(args.yolo_save_root / args.dataset_name.lower()),
        name=f"ours_{args.split}_stackobb_val",
        exist_ok=True,
        verbose=False,
    )
    return {
        "model": "cmp_xt_sc_endpoint_v1",
        "weights": str(best_pt),
        "dataset_root": str(data_yaml.parent),
        "split": args.split,
        "metric_source": "ultralytics.YOLO.val",
        "map50": float(val.box.map50),
        "map50_95": float(val.box.map),
        "precision": float(val.box.mp),
        "recall": float(val.box.mr),
        "pred_definition": "direct_yolo_obb_predictions",
        "gt_definition": "stacked_trajectory_obb_dataset_labels",
    }


def build_report_rows(compare_rows: dict[str, dict], ours_row: dict) -> list[dict]:
    rows = []
    for model, row in compare_rows.items():
        rows.append(
            {
                "tag": model,
                "family": "compare_segmentation",
                "weights": row["weights"],
                "map50": row["map50"],
                "map50_95": row["map50_95"],
                "precision": row["precision"],
                "recall": row["recall"],
                "metric_source": row["metric_source"],
                "pred_definition": row["pred_definition"],
            }
        )
    rows.append(
        {
            "tag": ours_row["model"],
            "family": "yolo_obb",
            "weights": ours_row["weights"],
            "map50": ours_row["map50"],
            "map50_95": ours_row["map50_95"],
            "precision": ours_row["precision"],
            "recall": ours_row["recall"],
            "metric_source": ours_row["metric_source"],
            "pred_definition": ours_row["pred_definition"],
        }
    )
    rows.sort(key=lambda row: row["map50_95"], reverse=True)
    return rows


def write_reports(args: argparse.Namespace, summary: dict) -> tuple[Path, Path]:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"cmp_{args.dataset_name.lower()}_{args.split}_stack_obb"
    out_json = REPORTS_ROOT / f"{stem}_summary.json"
    out_md = REPORTS_ROOT / f"{stem}_report.md"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = summary["rows"]
    lines = []
    lines.append(f"# {args.dataset_name} {args.split} Stacked OBB Comparison")
    lines.append("")
    lines.append(f"- Sequence dataset: `{summary['sequence_dataset']}`")
    lines.append(f"- Stacked OBB dataset: `{summary['stack_dataset']}`")
    lines.append(f"- Compare pred aggregation: `{summary['pred_agg_mode']}`")
    lines.append("")
    lines.append("| Method | Family | mAP50 | mAP50-95 | Precision | Recall |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['tag']} | {row['family']} | {row['map50']:.4f} | {row['map50_95']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- Compare-model GT is aggregated from sequence metadata (`line_obb` / `pt_obb`).")
    lines.append(f"- Compare-model prediction OBBs use `{summary['pred_agg_mode']}` without GT-assisted partitioning by default.")
    lines.append("- YOLO result is a direct `val` run on the same stacked OBB test set.")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_json, out_md


def main() -> None:
    args = parse_args()
    compare_data_yaml = ensure_stack_dataset(args)
    yolo_data_yaml = ensure_rgb_stack_dataset(args, compare_data_yaml)

    compare_rows: dict[str, dict] = {}
    for model in args.models:
        weights = find_compare_weights(model, args.dataset_name, args.ckpt_metric)
        compare_rows[model] = run_compare_eval(args, model, weights)

    ours_row = run_yolo_eval(args, yolo_data_yaml)
    rows = build_report_rows(compare_rows, ours_row)
    summary = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "sequence_dataset": str(args.seq_dataset_root / args.dataset_name),
        "stack_dataset": str(compare_data_yaml.parent),
        "stack_dataset_yolo": str(yolo_data_yaml.parent),
        "pred_agg_mode": args.pred_agg_mode,
        "component_min_pixels": args.component_min_pixels,
        "component_dilate": args.component_dilate,
        "compare_models": compare_rows,
        "ours": ours_row,
        "rows": rows,
    }
    out_json, out_md = write_reports(args, summary)
    print(json.dumps({"report_json": str(out_json), "report_md": str(out_md), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
