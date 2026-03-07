from pathlib import Path
import json

from ultralytics import YOLO


ROOT = Path("/home/chris/ultralytics")
RUNS = ROOT / "runs" / "obb"
RUNS.mkdir(parents=True, exist_ok=True)


common = dict(
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
    m = YOLO(str(best_pt))
    v = m.val(data=data_yaml, imgsz=common["imgsz"], batch=common["batch"], device=common["device"])
    return {
        "tag": tag,
        "best_pt": str(best_pt),
        "data": data_yaml,
        "map50": float(v.box.map50),
        "map50_95": float(v.box.map),
        "mp": float(v.box.mp),
        "mr": float(v.box.mr),
    }


def main() -> None:
    run_name = "cmp_xt_sc_gpu_full"
    data_sc = "dataset/obb_xt_clean/data_xt_sc.yaml"

    print("===== TRAIN xt_sc =====")
    m_sc = YOLO(common["model"])
    args = dict(common)
    args.update(data=data_sc, name=run_name, exist_ok=True)
    m_sc.train(**args)
    sc_best = Path(m_sc.trainer.save_dir) / "weights" / "best.pt"

    print("===== VAL all models =====")
    x_only_best = RUNS / "cmp_x_only_gpu_full" / "weights" / "best.pt"
    t_only_best = RUNS / "cmp_t_only_gpu_full_final" / "weights" / "best.pt"
    xt_best = RUNS / "cmp_x_t_gpu_full_final" / "weights" / "best.pt"

    res_x = eval_model("x_only", x_only_best, "dataset/obb_xt_clean/data_x.yaml")
    res_t = eval_model("t_only", t_only_best, "dataset/obb_xt_clean/data_t.yaml")
    res_xt = eval_model("x_t", xt_best, "dataset/obb_xt_clean/data_xt.yaml")
    res_sc = eval_model("x_sc", sc_best, data_sc)

    summary = {"x_only": res_x, "t_only": res_t, "x_t": res_xt, "x_sc": res_sc}

    out_json = RUNS / "cmp_x_t_xt_sc_summary_full.json"
    out_md = RUNS / "cmp_x_t_xt_sc_report_full.md"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = [res_x, res_t, res_xt, res_sc]
    rows_sorted = sorted(rows, key=lambda r: r["map50_95"], reverse=True)
    base = res_x

    lines = []
    lines.append("# X vs T vs X+T vs X+SinCos(T) OBB Full Report")
    lines.append("")
    lines.append("## Metrics")
    lines.append("| Setting | mAP50 | mAP50-95 | Precision | Recall |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['map50']:.4f} | {r['map50_95']:.4f} | {r['mp']:.4f} | {r['mr']:.4f} |"
        )

    lines.append("")
    lines.append("## Ranking (by mAP50-95)")
    for i, r in enumerate(rows_sorted, 1):
        lines.append(f"{i}. {r['tag']} ({r['map50_95']:.4f})")

    lines.append("")
    lines.append("## Delta vs X-only")
    for r in [res_t, res_xt, res_sc]:
        lines.append(
            f"- {r['tag']} mAP50: {r['map50'] - base['map50']:+.4f}, "
            f"mAP50-95: {r['map50_95'] - base['map50_95']:+.4f}, "
            f"P: {r['mp'] - base['mp']:+.4f}, R: {r['mr'] - base['mr']:+.4f}"
        )

    lines.append("")
    lines.append("## Conclusion")
    lines.append(f"- Best setting: {rows_sorted[0]['tag']} (mAP50-95={rows_sorted[0]['map50_95']:.4f}).")
    lines.append("- If x_sc is best, sin/cos time encoding is validated for full-run training.")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("saved", out_json)
    print("saved", out_md)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
