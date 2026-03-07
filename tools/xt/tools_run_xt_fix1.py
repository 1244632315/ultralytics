from pathlib import Path
import json
from ultralytics import YOLO
import multiprocessing.dummy as mp_dummy
import ultralytics.data.dataset as uds

# sandbox-safe fallback still compatible outside sandbox
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

run_name = 'cmp_x_t_gpu_full_final'
data_yaml = 'dataset/obb_xt_clean/data_xt.yaml'
model = YOLO(common['model'])
args = dict(common)
args['data'] = data_yaml
args['name'] = run_name
args['exist_ok'] = True
print(f'===== TRAIN x_t fixed labels path =====')
model.train(**args)
run_dir = Path(model.trainer.save_dir)
best_pt = run_dir / 'weights' / 'best.pt'
print(f'===== VAL x_t ({best_pt}) =====')
m2 = YOLO(str(best_pt))
val_res = m2.val(data=data_yaml, imgsz=common['imgsz'], batch=common['batch'], device=common['device'])

xt = {
    'tag': 'x_t_fix1',
    'run_name': run_name,
    'data': data_yaml,
    'best_pt': str(best_pt),
    'map50': float(val_res.box.map50),
    'map50_95': float(val_res.box.map),
    'mp': float(val_res.box.mp),
    'mr': float(val_res.box.mr),
}

# compare against completed x_only run
x_only_best = RUNS / 'cmp_x_only_gpu_full' / 'weights' / 'best.pt'
mx = YOLO(str(x_only_best))
vx = mx.val(data='dataset/obb_xt_clean/data_x.yaml', imgsz=common['imgsz'], batch=common['batch'], device=common['device'])
xo = {
    'tag': 'x_only',
    'run_name': 'cmp_x_only_gpu_full',
    'data': 'dataset/obb_xt_clean/data_x.yaml',
    'best_pt': str(x_only_best),
    'map50': float(vx.box.map50),
    'map50_95': float(vx.box.map),
    'mp': float(vx.box.mp),
    'mr': float(vx.box.mr),
}

summary = {'x_only': xo, 'x_t_fix1': xt}
out_json = RUNS / 'cmp_xt_summary_fix1.json'
out_md = RUNS / 'cmp_xt_report_fix1.md'
out_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

lines = []
lines.append('# OBB XT A/B Report (Fix1)')
lines.append('')
lines.append('## Context')
lines.append('- Initial x_t run failed because `images_xt/*` maps to `labels_xt/*` in YOLO path convention, but only `labels/*` existed.')
lines.append('- Fix applied: created `dataset/obb_xt_clean/labels_xt/{train,val}` symlinks to `labels/{train,val}`.')
lines.append('')
lines.append('## Metrics')
lines.append('| Setting | mAP50 | mAP50-95 | Precision | Recall |')
lines.append('|---|---:|---:|---:|---:|')
for r in [xo, xt]:
    lines.append(f"| {r['tag']} | {r['map50']:.4f} | {r['map50_95']:.4f} | {r['mp']:.4f} | {r['mr']:.4f} |")
lines.append('')
lines.append('## Delta (x_t_fix1 - x_only)')
lines.append(f"- mAP50: {xt['map50'] - xo['map50']:+.4f}")
lines.append(f"- mAP50-95: {xt['map50_95'] - xo['map50_95']:+.4f}")
lines.append(f"- Precision: {xt['mp'] - xo['mp']:+.4f}")
lines.append(f"- Recall: {xt['mr'] - xo['mr']:+.4f}")

if xt['map50_95'] > xo['map50_95']:
    lines.append('')
    lines.append('## Conclusion')
    lines.append('- Adding temporal projection (X+T) is effective under the current setup.')
else:
    lines.append('')
    lines.append('## Conclusion')
    lines.append('- Adding temporal projection did not improve over X-only under the current setup.')
    lines.append('- Possible causes: T-domain mismatch, too strong/too weak uncertainty settings, or augmentation-policy mismatch.')

out_md.write_text('\n'.join(lines), encoding='utf-8')
print('saved', out_json)
print('saved', out_md)
print(json.dumps(summary, indent=2))
