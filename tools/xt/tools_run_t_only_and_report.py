from pathlib import Path
import json
from ultralytics import YOLO
import multiprocessing.dummy as mp_dummy
import ultralytics.data.dataset as uds

# keep compatible with restrictive environments
uds.ThreadPool = mp_dummy.Pool
uds.NUM_THREADS = 1

ROOT = Path('/home/chris/ultralytics')
RUNS = ROOT / 'runs' / 'obb'
RUNS.mkdir(parents=True, exist_ok=True)

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

run_name = 'cmp_t_only_gpu_full_final'
data_t = 'dataset/obb_xt_clean/data_t.yaml'

print('===== TRAIN t_only =====')
mt = YOLO(common['model'])
args = dict(common)
args.update(data=data_t, name=run_name, exist_ok=True)
mt.train(**args)
run_dir = Path(mt.trainer.save_dir)
best_t = run_dir / 'weights' / 'best.pt'

print('===== VAL all models =====')

def eval_model(tag, best_pt, data_yaml):
    m = YOLO(str(best_pt))
    v = m.val(data=data_yaml, imgsz=common['imgsz'], batch=common['batch'], device=common['device'])
    return {
        'tag': tag,
        'best_pt': str(best_pt),
        'data': data_yaml,
        'map50': float(v.box.map50),
        'map50_95': float(v.box.map),
        'mp': float(v.box.mp),
        'mr': float(v.box.mr),
    }

x_only_best = RUNS / 'cmp_x_only_gpu_full' / 'weights' / 'best.pt'
xt_best = RUNS / 'cmp_x_t_gpu_full_final' / 'weights' / 'best.pt'

res_x = eval_model('x_only', x_only_best, 'dataset/obb_xt_clean/data_x.yaml')
res_t = eval_model('t_only', best_t, data_t)
res_xt = eval_model('x_t', xt_best, 'dataset/obb_xt_clean/data_xt.yaml')

summary = {'x_only': res_x, 't_only': res_t, 'x_t': res_xt}

out_json = RUNS / 'cmp_x_t_xt_summary.json'
out_md = RUNS / 'cmp_x_t_xt_report.md'
out_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

rows = [res_x, res_t, res_xt]
rows_sorted = sorted(rows, key=lambda r: r['map50_95'], reverse=True)

lines = []
lines.append('# X vs T vs X+T OBB Report')
lines.append('')
lines.append('## Metrics')
lines.append('| Setting | mAP50 | mAP50-95 | Precision | Recall |')
lines.append('|---|---:|---:|---:|---:|')
for r in rows:
    lines.append(f"| {r['tag']} | {r['map50']:.4f} | {r['map50_95']:.4f} | {r['mp']:.4f} | {r['mr']:.4f} |")
lines.append('')
lines.append('## Ranking (by mAP50-95)')
for i, r in enumerate(rows_sorted, 1):
    lines.append(f"{i}. {r['tag']} ({r['map50_95']:.4f})")
lines.append('')
lines.append('## Delta vs X-only')
for r in [res_t, res_xt]:
    lines.append(f"- {r['tag']} mAP50: {r['map50'] - res_x['map50']:+.4f}, mAP50-95: {r['map50_95'] - res_x['map50_95']:+.4f}, P: {r['mp'] - res_x['mp']:+.4f}, R: {r['mr'] - res_x['mr']:+.4f}")
lines.append('')
if rows_sorted[0]['tag'] == 'x_t':
    lines.append('## Conclusion')
    lines.append('- X+T is best under current setup; temporal projection adds measurable value beyond X-only and T-only.')
elif rows_sorted[0]['tag'] == 't_only':
    lines.append('## Conclusion')
    lines.append('- T-only unexpectedly dominates; revisit X-channel preprocessing and fusion strategy.')
else:
    lines.append('## Conclusion')
    lines.append('- X-only remains strongest; current T generation may still have domain mismatch.')

out_md.write_text('\n'.join(lines), encoding='utf-8')

print('saved', out_json)
print('saved', out_md)
print(json.dumps(summary, indent=2))
