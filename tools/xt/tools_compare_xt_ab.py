from pathlib import Path
import json
import pandas as pd
from ultralytics import YOLO
import multiprocessing.dummy as mp_dummy
import ultralytics.data.dataset as uds

# Sandbox-safe fallback: avoid SemLock in multiprocessing.pool.ThreadPool
uds.ThreadPool = mp_dummy.Pool
uds.NUM_THREADS = 1

ROOT = Path('/home/chris/ultralytics')
RUNS = ROOT / 'runs' / 'obb'
RUNS.mkdir(parents=True, exist_ok=True)

# CPU-only quick A/B (same hyper-params, only data differs)
common = dict(
    model='yolo11n-obb.pt',
    imgsz=1024,
    epochs=100,
    batch=8,
    device=0,
    workers=8,
    seed=0,
    deterministic=True,
    pretrained=True,
    optimizer='auto',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=3.0,
    translate=0.04,
    scale=0.12,
    shear=0.0,
    perspective=0.0,
    fliplr=0.3,
    flipud=0.0,
    mosaic=0.1,
    mixup=0.0,
    cutmix=0.0,
    close_mosaic=4,
    fraction=1.0,
    project=str(RUNS),
    save=True,
    val=True,
    verbose=True,
)

exps = [
    ('x_only', 'dataset/obb_xt_clean/data_x.yaml', 'cmp_x_only_gpu_full'),
    ('x_t', 'dataset/obb_xt_clean/data_xt.yaml', 'cmp_x_t_gpu_full'),
]

summary = []
for tag, data_yaml, run_name in exps:
    model = YOLO(common['model'])
    args = dict(common)
    args['data'] = data_yaml
    args['name'] = run_name
    print(f'\n===== TRAIN {tag} =====')
    model.train(**args)

    run_dir = RUNS / run_name
    best_pt = run_dir / 'weights' / 'best.pt'
    print(f'\n===== VAL {tag} ({best_pt}) =====')
    m2 = YOLO(str(best_pt))
    val_res = m2.val(data=data_yaml, imgsz=common['imgsz'], batch=common['batch'], device=common['device'])

    metrics = {
        'tag': tag,
        'run_name': run_name,
        'data': data_yaml,
        'best_pt': str(best_pt),
        'map50': float(val_res.box.map50),
        'map50_95': float(val_res.box.map),
        'mp': float(val_res.box.mp),
        'mr': float(val_res.box.mr),
    }

    csv_path = run_dir / 'results.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        last = df.iloc[-1].to_dict()
        metrics['last_epoch'] = int(last.get('epoch', -1))
    summary.append(metrics)

out_json = RUNS / 'cmp_xt_summary.json'
out_md = RUNS / 'cmp_xt_summary.md'
out_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

if len(summary) == 2:
    a, b = summary
    lines = []
    lines.append('# X vs X+T (Quick A/B)')
    lines.append('')
    lines.append('| Setting | mAP50 | mAP50-95 | Precision | Recall |')
    lines.append('|---|---:|---:|---:|---:|')
    for r in summary:
        lines.append(f"| {r['tag']} | {r['map50']:.4f} | {r['map50_95']:.4f} | {r['mp']:.4f} | {r['mr']:.4f} |")
    lines.append('')
    lines.append('## Delta (x_t - x_only)')
    lines.append('')
    lines.append(f"- mAP50: {b['map50'] - a['map50']:+.4f}")
    lines.append(f"- mAP50-95: {b['map50_95'] - a['map50_95']:+.4f}")
    lines.append(f"- Precision: {b['mp'] - a['mp']:+.4f}")
    lines.append(f"- Recall: {b['mr'] - a['mr']:+.4f}")
    out_md.write_text('\n'.join(lines), encoding='utf-8')

print('\nSaved summary:')
print(out_json)
print(out_md)
print(json.dumps(summary, indent=2))
