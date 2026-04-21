from pathlib import Path
import json

from ultralytics import YOLO


ROOT = Path("/home/chris/ultralytics")
RUNS = ROOT / "runs" / "obb"
REPORTS = RUNS / "reports"
RUNS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


COMMON = dict(
    model="yolo11n-obb.pt",
    imgsz=1024,
    epochs=100,
    batch=8,
    device=0,
    workers=8,
    seed=0,
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
)


def eval_model(tag: str, best_pt: Path, data_yaml: str) -> dict:
    model = YOLO(str(best_pt))
    val = model.val(data=data_yaml, imgsz=COMMON["imgsz"], batch=COMMON["batch"], device=COMMON["device"])
    return {
        "tag": tag,
        "best_pt": str(best_pt),
        "data": data_yaml,
        "map50": float(val.box.map50),
        "map50_95": float(val.box.map),
        "mp": float(val.box.mp),
        "mr": float(val.box.mr),
    }


def main() -> None:
    data_yaml = "dataset/obb_xt_clean/data_xt_sc.yaml"
    run_name = "cmp_xt_sc_endpoint_v1_gpu_full"

    x_sc_pt = RUNS / "cmp_xt_sc_gpu_full" / "weights" / "best.pt"
    angle_v1_pt = RUNS / "cmp_xt_sc_angleloss_v1_gpu_full" / "weights" / "best.pt"

    print("===== TRAIN endpoint-loss v1 =====")
    model = YOLO(COMMON["model"])
    train_args = dict(COMMON)
    train_args.update(data=data_yaml, name=run_name, exist_ok=True, angle=2.0)
    model.train(**train_args)
    endpoint_v1_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"

    print("===== VAL all =====")
    res_x_sc = eval_model("x_sc_baseline", x_sc_pt, data_yaml)
    res_angle_v1 = eval_model("x_sc_angleloss_v1", angle_v1_pt, data_yaml)
    res_endpoint_v1 = eval_model("x_sc_endpoint_v1", endpoint_v1_pt, data_yaml)
    rows = [res_x_sc, res_angle_v1, res_endpoint_v1]
    summary = {r["tag"]: r for r in rows}

    out_json = REPORTS / "cmp_xt_sc_endpoint_v1_summary_full.json"
    out_md = REPORTS / "cmp_xt_sc_endpoint_v1_report_full.md"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    by_map = sorted(rows, key=lambda r: r["map50_95"], reverse=True)
    base = res_angle_v1
    lines = []
    lines.append("# XT_sc Endpoint-Loss v1 Full Report")
    lines.append("")
    lines.append("| Setting | mAP50 | mAP50-95 | Precision | Recall |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['map50']:.4f} | {r['map50_95']:.4f} | {r['mp']:.4f} | {r['mr']:.4f} |"
        )
    lines.append("")
    lines.append("## Ranking (by mAP50-95)")
    for i, r in enumerate(by_map, 1):
        lines.append(f"{i}. {r['tag']} ({r['map50_95']:.4f})")
    lines.append("")
    lines.append(
        "- Delta(endpoint_v1 - angleloss_v1): "
        f"mAP50 {res_endpoint_v1['map50'] - base['map50']:+.4f}, "
        f"mAP50-95 {res_endpoint_v1['map50_95'] - base['map50_95']:+.4f}, "
        f"P {res_endpoint_v1['mp'] - base['mp']:+.4f}, "
        f"R {res_endpoint_v1['mr'] - base['mr']:+.4f}"
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("saved", out_json)
    print("saved", out_md)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
