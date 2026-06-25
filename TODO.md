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

## Stage 1.1: Unified-T Rule (UT1) Rebuild + Retrain
- [x] Rebuild dataset with unified T composition logic
  - script: `tools/xt/tools_build_obb_xt_clean_ut1.py`
  - output root: `dataset/obb_xt_clean_ut1`
  - split stats: `train=892`, `val=223`, `total=1115`
  - sanity checks:
    - `images/val_t` unique levels per image (sampled): min/med/max `6/22/29`
    - `xt_sc` image-label pairing: train `892/892`, val `223/223`
- [x] Full training (100 epochs, GPU) on UT1 `xt_sc`
  - script: `tools/xt/tools_train_xt_sc_ut1.py`
  - run dir: `runs/obb/cmp_xt_sc_endpoint_ut1_gpu_full`
  - best epoch (`results.csv`): `99`
  - best val: `mAP50=0.9902`, `mAP50-95=0.8375`, `P=0.9795`, `R=0.9643`
- [x] Same-set comparison (`dataset/obb_xt_clean_ut1/data_xt_sc.yaml`)
  - old model: `runs/obb/cmp_xt_sc_endpoint_v1_gpu_full/weights/best.pt`
    - `mAP50=0.9885`, `mAP50-95=0.8472`, `P=0.9797`, `R=0.9739`
  - UT1 model: `runs/obb/cmp_xt_sc_endpoint_ut1_gpu_full/weights/best.pt`
    - `mAP50=0.9901`, `mAP50-95=0.8409`, `P=0.9773`, `R=0.9643`
  - delta (UT1 - old): `mAP50 +0.0016`, `mAP50-95 -0.0063`, `P -0.0024`, `R -0.0096`
  - report: `runs/obb/reports/cmp_xt_sc_ut1_report_full.md`
  - summary: `runs/obb/reports/cmp_xt_sc_ut1_summary_full.json`

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
- [x] Add endpoint auxiliary loss on centerline (`(cx,cy,w,h,theta) -> 2 endpoints`)
  - implementation: `ultralytics/utils/loss.py` (`calculate_endpoint_loss`, swap-invariant endpoint matching)
  - script: `tools/xt/tools_run_endpoint_loss_v1.py`
  - run dir: `runs/obb/cmp_xt_sc_endpoint_v1_gpu_full`
  - report: `runs/obb/reports/cmp_xt_sc_endpoint_v1_report_full.md`
  - summary: `runs/obb/reports/cmp_xt_sc_endpoint_v1_summary_full.json`
  - result: `x_sc_angleloss_v1` `0.8391` -> `x_sc_endpoint_v1` `0.8455` (`+0.0064`)
- [ ] If needed, increase angle head capacity/receptive field

## Stage 6: Trajectory-Level Comparison Protocol
- [x] Freeze trajectory-level comparison goal and scope
  - protocol doc: `tools/xt/TRAJECTORY_EVAL_PROTOCOL.md`
  - main question: trajectory detected or not, and whether detection is complete enough
- [ ] Build unified evaluator for detector and segmentation baselines
  - target outputs: `TDR`, `CTR`, `mean_TC`, `EHR`, `Frag`, `FTPS`
  - v1 script added: `tools/xt/tools_eval_trajectory_protocol.py`
  - current status: supports `frame_support` summaries and `box_match_proxy` fallback
  - reuse current adapters from:
    - `tools/xt/tools_eval_compare_stack_obb.py`
    - `tools/xt/tools_eval_kbs_compare_bbox.py`
    - `tools/xt/tools_eval_kbs_ours_seqmask.py`
- [x] Define initial pilot sequence subset for protocol calibration
  - target size: `5-10` representative KBS sequences
  - include: single target, multi-target, weak target, cluttered background
  - initial pilot ids: `059 027 006 077 083`
- [ ] Run Group B first: fair cross-family trajectory comparison
  - methods: `CSAUNet`, `DNANet`, `DnTNet`, `MSAMNet`, `x_sc_endpoint_v1`
  - primary table: trajectory-level metrics only
  - current pilot outputs:
    - detector reconstruct summary: `runs/kbs_eval/kbs_detector_bbox_pilot5_reconstruct_cpu/ours_xt_sc/summary.json`
    - protocol report: `runs/trajectory_eval/protocol_pilot5_kbs_bbox/report.md`
  - current full outputs:
    - ours reconstruct full: `runs/kbs_eval/kbs_detector_bbox_reconstruct_full_cpu/ours_xt_sc/summary.json`
    - DNANet ft10 split bbox: `runs/kbs_eval/kbs_dnanet_ft10_split_bbox/summary.json`
    - MSAMNet ft10 split bbox: `runs/kbs_eval/kbs_msamnet_ft10_split_bbox/summary.json`
    - merged trajectory report: `runs/trajectory_eval/protocol_kbs_full_reconstruct_and_compare/report.md`
- [ ] Then rerun Group A and Group C under the same protocol
  - Group A: `x_only/t_only/x_t/x_sc/x_sc_endpoint_v1`
  - Group C: same-source vs real-domain transfer comparison

## Report Deliverables
- [x] Per-run metrics table (`mAP50`, `mAP50-95`, `P`, `R`)
- [ ] Same-val-set comparison plots
- [x] Final recommendation for production training recipe
  - recommend `x_sc`: `[X, sin(pi*tau), cos(pi*tau)]`
