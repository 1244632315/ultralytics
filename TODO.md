# XT Ablation TODO

## Baseline (done)
- [x] `x_only` / `t_only` / `x_t` completed on `dataset/obb_xt_clean`
- [x] Summary saved: `runs/obb/reports/cmp_x_t_xt_summary.json`

## Stage 1: X/T Independent Normalization (now)
- [x] Build `xt_norm1` dataset:
  - source `X`: `dataset/obb_xt_clean/images/{train,val}`
  - source `T`: `dataset/obb_xt_clean/images/{train_t,val_t}`
  - per-channel robust stretch: `p1-p99` clip + map to `[0,255]`
  - output layout: `dataset/obb_xt_clean/images/{train_xt_norm1,val_xt_norm1}`
- [x] Add `data_xt_norm1.yaml` for training/validation
- [x] Run short screening train (30 epochs) against `xt` baseline
  - result: `xt_norm1` did not improve over matched `xt_raw` in 30-epoch screen
- [ ] If improved, run full train (100 epochs) and compare with `x/t/xt`

## Stage 2: XT Fusion Variants
- [ ] Compare channel layouts:
  - `A`: `[X_norm, T_norm, X_norm]`
  - `B`: `[X_norm, T_norm, edge(X_norm)]`
  - `C`: `[X_norm, T_norm, 0]`

## Stage 2.5: Time Encoding Variant
- [x] `XT_sc` screening run (30 epochs), channels: `[X, sin(pi*tau), cos(pi*tau)]`, `tau=T/255`
- [x] Compare against matched `xt_raw_screen30` and `xt_norm1_screen30`
- [x] Result: `XT_sc` clearly improved in 30-epoch screen (`mAP50-95 +0.0587` vs raw)
- [x] Promote `XT_sc` to full run (100 epochs) for final confirmation
  - run dir: `runs/obb/cmp_xt_sc_gpu_full`
  - summary: `runs/obb/reports/cmp_x_t_xt_sc_summary_full.json`
  - report: `runs/obb/reports/cmp_x_t_xt_sc_report_full.md`
  - full-run result: `x_sc` is best (`mAP50-95=0.8356`)

## Stage 3: Background-T Intensity Control
- [ ] Keep full-frame time encoding, but compress background contrast
- [ ] Keep target trajectory T contrast unchanged

## Stage 4: Augmentation Tuning for Weak Small Targets
- [ ] Reduce geometry distortion (`mosaic/translate/scale`)
- [ ] Keep lightweight flips, disable aggressive transforms

## Stage 5: Loss/Head Optimization for Sparse OBB
- [x] `angle_loss` elongated-target weighting
  - change: do not suppress high-aspect-ratio targets in angle loss
  - plan: baseline `angle=1.0` vs boost `angle=2.0` full-run compare
  - run: normal foreground run on GPU (2026-03-08)
  - script: `tools/xt/tools_run_angle_loss_v1.py`
  - report: `runs/obb/reports/cmp_xt_sc_angleloss_v1_report_full.md`
  - summary: `runs/obb/reports/cmp_xt_sc_angleloss_v1_summary_full.json`
  - result: baseline `x_sc` mAP50-95 `0.8356` -> `x_sc_angleloss_v1` `0.8391` (`+0.0035`)
- [ ] Add endpoint auxiliary loss on centerline (`(cx,cy,w,h,theta) -> 2 endpoints`)
- [ ] If needed, increase angle head capacity/receptive field

## Report Deliverables
- [x] Per-run metrics table (`mAP50`, `mAP50-95`, `P`, `R`)
- [ ] Same-val-set comparison plots
- [x] Final recommendation for production training recipe
  - recommend `x_sc`: `[X, sin(pi*tau), cos(pi*tau)]`
