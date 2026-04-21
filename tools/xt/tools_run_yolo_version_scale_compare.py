from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path("/home/chris/ultralytics")
RUNS = ROOT / "runs" / "obb"
REPORTS = RUNS / "reports"
RUNS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


EXPERIMENTS = [
    # Version line (all n-scale)
    {"tag": "v8n", "model": "yolov8n-obb.yaml", "family": "version_n"},
    {"tag": "v11n", "model": "yolo11n-obb.yaml", "family": "version_n"},
    {"tag": "v12n", "model": "yolo12n-obb.yaml", "family": "version_n"},
    {"tag": "v26n", "model": "yolo26n-obb.yaml", "family": "version_n"},
    # Scale line (YOLO11)
    {"tag": "v11s", "model": "yolo11s-obb.yaml", "family": "scale_11"},
    {"tag": "v11m", "model": "yolo11m-obb.yaml", "family": "scale_11"},
    {"tag": "v11x", "model": "yolo11x-obb.yaml", "family": "scale_11"},
]

# Use local weights when available to avoid offline download failures and
# keep initialization closer to prior experiments.
LOCAL_PRETRAINED = {
    "v8n": ROOT / "yolov8n-obb.pt",
    "v11n": ROOT / "yolo11n-obb.pt",
    "v11s": ROOT / "yolo11s-obb.pt",
    "v11m": ROOT / "yolo11m-obb.pt",
    "v11x": ROOT / "yolo11x-obb.pt",
    "v26n": ROOT / "yolo26n-obb.pt",
}


def _param_count(model_yaml: str) -> int:
    m = YOLO(model_yaml)
    return int(sum(p.numel() for p in m.model.parameters()))


def _best_metrics_from_csv(csv_path: Path) -> dict:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"empty results.csv: {csv_path}")
    key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda x: float(x[key]))
    last = rows[-1]
    return {
        "best_epoch": int(float(best["epoch"])),
        "best_map50": float(best["metrics/mAP50(B)"]),
        "best_map50_95": float(best[key]),
        "best_p": float(best["metrics/precision(B)"]),
        "best_r": float(best["metrics/recall(B)"]),
        "last_epoch": int(float(last["epoch"])),
        "train_time_seconds_last_epoch": float(last["time"]),
    }


def _last_epoch_from_csv(csv_path: Path) -> int:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows:
        return -1
    return int(float(rows[-1]["epoch"]))


def eval_model(best_pt: Path, data_yaml: str, imgsz: int, batch: int, device: int | str) -> dict:
    m = YOLO(str(best_pt))
    v = m.val(data=data_yaml, imgsz=imgsz, batch=batch, device=device)
    speed = getattr(v, "speed", {}) or {}
    infer_ms = float(speed.get("inference", 0.0))
    preprocess_ms = float(speed.get("preprocess", 0.0))
    postprocess_ms = float(speed.get("postprocess", 0.0))
    return {
        "map50": float(v.box.map50),
        "map50_95": float(v.box.map),
        "mp": float(v.box.mp),
        "mr": float(v.box.mr),
        "infer_ms_img": infer_ms,
        "pre_ms_img": preprocess_ms,
        "post_ms_img": postprocess_ms,
        "fps_infer": (1000.0 / infer_ms) if infer_ms > 1e-9 else 0.0,
    }


def make_md(title: str, rows: list[dict]) -> str:
    s = []
    s.append(f"# {title}")
    s.append("")
    s.append("| Tag | Model | Params | mAP50 | mAP50-95 | Precision | Recall | Infer ms/img | FPS | Best Epoch |")
    s.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        infer_ms = float(r.get("infer_ms_img", 0.0) or 0.0)
        fps = float(r.get("fps_infer", 0.0) or 0.0)
        s.append(
            f"| {r['tag']} | {r['model']} | {r['params']:,} | "
            f"{r['map50']:.4f} | {r['map50_95']:.4f} | {r['mp']:.4f} | {r['mr']:.4f} | "
            f"{infer_ms:.2f} | {fps:.1f} | {r['best_epoch']} |"
        )
    s.append("")
    s.append("## Ranking (by mAP50-95)")
    for i, r in enumerate(sorted(rows, key=lambda x: x["map50_95"], reverse=True), 1):
        s.append(f"{i}. {r['tag']} ({r['map50_95']:.4f})")
    return "\n".join(s) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suffix", default="screen30")
    parser.add_argument("--data", default="dataset/obb_xt_clean/data_xt_sc.yaml")
    parser.add_argument("--angle", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--tags",
        default="",
        help="comma-separated tags to run, e.g. v8n,v11n,v12n. empty means run all.",
    )
    args = parser.parse_args()

    wanted = {x.strip() for x in args.tags.split(",") if x.strip()}
    exp_list = [e for e in EXPERIMENTS if not wanted or e["tag"] in wanted]
    if not exp_list:
        raise ValueError(f"no experiments selected by --tags={args.tags!r}")

    common = dict(
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        close_mosaic=10,
        fraction=1.0,
        project=str(RUNS),
        save=True,
        val=True,
        verbose=True,
        angle=2.0,
    )
    common["angle"] = args.angle

    all_rows: list[dict] = []
    data_tag = Path(args.data).stem.replace("data_", "")
    for exp in exp_list:
        tag = exp["tag"]
        model_yaml = exp["model"]
        family = exp["family"]
        run_name = f"cmp_{tag}_{data_tag}_{args.suffix}"
        run_dir = RUNS / run_name
        best_pt = run_dir / "weights" / "best.pt"
        last_pt = run_dir / "weights" / "last.pt"
        results_csv = run_dir / "results.csv"
        local_pretrained = LOCAL_PRETRAINED.get(tag)
        pretrained_arg = True
        if local_pretrained and local_pretrained.exists():
            pretrained_arg = str(local_pretrained)

        print(f"\n===== [{tag}] model={model_yaml} family={family} =====")
        is_complete = False
        if best_pt.exists() and results_csv.exists():
            last_epoch = _last_epoch_from_csv(results_csv)
            is_complete = last_epoch >= args.epochs

        if not args.force and is_complete:
            print(f"reuse existing run: {run_dir}")
        else:
            model = YOLO(model_yaml)
            train_args = dict(common)
            train_args.update(
                model=model_yaml,
                data=args.data,
                name=run_name,
                exist_ok=True,
                pretrained=pretrained_arg,
            )
            if results_csv.exists() and last_pt.exists():
                last_epoch = _last_epoch_from_csv(results_csv)
                if last_epoch < args.epochs:
                    # Resume incomplete run from last checkpoint.
                    train_args["model"] = str(last_pt)
                    train_args["resume"] = True
                    train_args["pretrained"] = False
            model.train(**train_args)

        if not best_pt.exists() or not results_csv.exists():
            raise FileNotFoundError(f"missing artifacts for run {run_name}: {best_pt} / {results_csv}")

        val_metrics = eval_model(best_pt, args.data, args.imgsz, args.batch, args.device)
        csv_metrics = _best_metrics_from_csv(results_csv)
        params = _param_count(model_yaml)

        row = {
            "tag": tag,
            "family": family,
            "model": model_yaml,
            "run_name": run_name,
            "best_pt": str(best_pt),
            "data": args.data,
            "pretrained_init": str(pretrained_arg),
            "params": params,
            **val_metrics,
            **csv_metrics,
        }
        all_rows.append(row)
        print(
            f"[{tag}] map50={row['map50']:.4f}, map50_95={row['map50_95']:.4f}, "
            f"P={row['mp']:.4f}, R={row['mr']:.4f}, best_epoch={row['best_epoch']}, params={row['params']}"
        )

    out_json = REPORTS / f"cmp_yolo_version_scale_{args.suffix}_summary.json"
    out_csv = REPORTS / f"cmp_yolo_version_scale_{args.suffix}_summary.csv"
    out_md_all = REPORTS / f"cmp_yolo_version_scale_{args.suffix}_report.md"
    out_md_v = REPORTS / f"cmp_yolo_version_n_{args.suffix}_report.md"
    out_md_s = REPORTS / f"cmp_yolo_scale_11_{args.suffix}_report.md"

    out_json.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    fieldnames = [
        "tag",
        "family",
        "model",
        "run_name",
        "data",
        "best_pt",
        "pretrained_init",
        "params",
        "map50",
        "map50_95",
        "mp",
        "mr",
        "infer_ms_img",
        "pre_ms_img",
        "post_ms_img",
        "fps_infer",
        "best_epoch",
        "best_map50",
        "best_map50_95",
        "best_p",
        "best_r",
        "last_epoch",
        "train_time_seconds_last_epoch",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    out_md_all.write_text(make_md(f"YOLO Version+Scale Compare ({args.suffix})", all_rows), encoding="utf-8")

    version_rows = [r for r in all_rows if r["tag"] in {"v8n", "v11n", "v26n"}]
    scale_rows = [r for r in all_rows if r["tag"] in {"v11n", "v11s", "v11m", "v11x"}]
    out_md_v.write_text(make_md(f"YOLO Version Line Compare ({args.suffix})", version_rows), encoding="utf-8")
    out_md_s.write_text(make_md(f"YOLO11 Scale Line Compare ({args.suffix})", scale_rows), encoding="utf-8")

    print("\nSaved:")
    print(out_json)
    print(out_csv)
    print(out_md_all)
    print(out_md_v)
    print(out_md_s)


if __name__ == "__main__":
    main()
