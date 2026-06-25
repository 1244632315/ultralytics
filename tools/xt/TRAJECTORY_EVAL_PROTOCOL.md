# Trajectory-Level Evaluation Protocol

## Goal

This protocol replaces pixel-first or box-first comparison with a unified trajectory-level comparison.

Core questions:

- Was each ground-truth trajectory detected?
- Was the trajectory detected completely enough?
- Was the trajectory broken into fragments?
- How many extra false trajectories were introduced?

This is the main comparison protocol for:

- detector-style methods such as YOLO-OBB
- segmentation-style methods such as `CSAUNet`, `DNANet`, `DnTNet`, `MSAMNet`

## Scope

Primary datasets:

- simulation same-source comparison:
  - `dataset/xt_seq_msod_x/MSOD_SIM_JOINT`
  - `dataset/xt_seq_msod_x/MSOD_RAWBG`
- real-domain comparison:
  - `dataset/KBS_dataset/mosaic+`

Primary model groups:

- ours:
  - `x_only`
  - `t_only`
  - `x_t`
  - `x_sc`
  - `x_sc_endpoint_v1`
- compare segmentation baselines:
  - `CSAUNet`
  - `DNANet`
  - `DnTNet`
  - `MSAMNet`

## Unified Evaluation Object

All methods must be converted into the same intermediate object:

- `sequence_id`
- `pred_track_id`
- `frame_ids`
- `per_frame_mask`
- `union_mask`
- optional `bbox/obb`

Comparison is performed on trajectory instances, not directly on raw masks or raw OBBs.

## Prediction Conversion Rules

### Segmentation baselines

Pipeline:

1. per-frame prediction mask
2. connected components per frame
3. cross-frame association into track proposals
4. union mask and per-frame support for each predicted trajectory

Current reusable code:

- `tools/xt/tools_eval_kbs_compare_seg.py`
- `tools/xt/tools_eval_kbs_compare_seqproj.py`
- `tools/xt/tools_eval_kbs_compare_bbox.py`
- `tools/xt/tools_eval_compare_stack_obb.py`

### Ours detector

Pipeline:

1. OBB prediction per sequence or per frame
2. rasterize predicted OBBs to masks
3. convert to trajectory support representation
4. evaluate with the same GT trajectory definition

Current reusable code:

- `tools/xt/tools_eval_kbs_ours_seqmask.py`
- `tools/xt/tools_eval_kbs_detector_bbox.py`

## Ground-Truth Trajectory Definition

For each GT trajectory instance:

- retain its frame support
- retain per-frame binary support mask
- retain sequence-level union mask
- optionally retain centerline or sampled skeleton points

Phase 1 uses frame-support and mask-support metrics only.

Phase 2 may add centerline sampling if phase-1 metrics are not discriminative enough.

## Primary Metrics

### 1. Track Completeness (`TC`)

For one GT trajectory:

- compute the fraction of GT-supported frames or GT-supported mask area covered by the matched prediction

Recommended phase-1 implementation:

- frame-weighted mask coverage over all GT frames

Range:

- `0.0` means fully missed
- `1.0` means fully covered

### 2. Track Detection Rate (`TDR`)

Definition:

- fraction of GT trajectories with `TC >= 0.5`

### 3. Complete Track Rate (`CTR`)

Definition:

- fraction of GT trajectories with `TC >= 0.8`

### 4. Endpoint Hit Rate (`EHR`)

Definition:

- both trajectory start and end neighborhoods are covered by prediction

Phase-1 implementation:

- endpoint region is built from first and last GT support areas with a small dilation radius

### 5. Fragmentation (`Frag`)

Definition:

- number of predicted trajectory pieces assigned to one GT trajectory

Recommended report forms:

- `mean_frag_per_gt`
- `num_gt_with_frag_gt_1`

### 6. False Tracks per Sequence (`FTPS`)

Definition:

- unmatched predicted trajectories per sequence

Recommended report forms:

- `FTPS`
- `false_tracks_per_frame`

## Match Rule

Matching is GT-centric.

For each GT trajectory:

1. collect all predicted trajectories with non-trivial overlap
2. choose the best prediction by highest trajectory coverage score
3. compute `TC`, `EHR`, and fragmentation against that GT

Suggested initial overlap gate:

- candidate if frame overlap exists and union-mask overlap is non-zero

Suggested initial thresholds:

- detected: `TC >= 0.5`
- complete: `TC >= 0.8`

Predictions with no GT match are counted as false tracks.

## Auxiliary Metrics

These can remain in appendix only:

- pixel `Precision/Recall/IoU/Dice`
- OBB `mAP50/mAP50-95`
- raw sequence projection IoU

They are not the main conclusion for cross-family comparison.

## Experiment Groups

## Group A: Internal Ours Validation

Purpose:

- justify the current representation and loss design

Models:

- `x_only`
- `t_only`
- `x_t`
- `x_sc`
- `x_sc_endpoint_v1`

Main result:

- trajectory-level table

Appendix:

- OBB validation table

## Group B: Fair Cross-Family Comparison

Purpose:

- compare segmentation and detection methods under the same trajectory task

Models:

- `CSAUNet`
- `DNANet`
- `DnTNet`
- `MSAMNet`
- `x_sc_endpoint_v1`

Main result table columns:

- `Method`
- `TDR`
- `CTR`
- `mean_TC`
- `EHR`
- `mean_frag_per_gt`
- `FTPS`

## Group C: Domain Transfer Comparison

Purpose:

- separate same-source capability from real-domain robustness

Subgroups:

- same-source:
  - `MSOD_SIM_JOINT`
  - `MSOD_RAWBG`
- real-domain:
  - `KBS test`

Rule:

- do not mix same-source and real-domain conclusions in one ranking table

## Execution Order

## Phase 0: Freeze Protocol

Deliverables:

- this protocol document
- threshold set for `TC`, `CTR`, `EHR`

## Phase 1: Pilot Set

Use `5-10` representative sequences.

Required coverage:

- single clean trajectory
- multiple nearby trajectories
- weak or broken trajectory
- strong clutter or star background

Goals:

- verify metric stability
- verify qualitative alignment with human judgment
- tune thresholds only once here

## Phase 2: Unified Evaluator

Implement one evaluator that:

- loads method outputs
- converts them into trajectory instances
- computes `TDR`, `CTR`, `TC`, `EHR`, `Frag`, `FTPS`
- exports per-sequence and summary JSON

Suggested output:

- `runs/trajectory_eval/<run_name>/summary.json`
- `runs/trajectory_eval/<run_name>/per_sequence/*.json`
- `runs/trajectory_eval/<run_name>/vis/*.png`

Current status:

- v1 script added: `tools/xt/tools_eval_trajectory_protocol.py`
- current implementation supports:
  - frame-support evaluation when `pred_tracks` exist
  - explicit `box_match_proxy` fallback when only bbox-match summaries exist

## Phase 3: Group B First

Run the cross-family comparison first because it addresses the current criticism most directly.

Priority:

1. `x_sc_endpoint_v1`
2. `DnTNet`
3. `DNANet`
4. `CSAUNet`
5. `MSAMNet`

## Phase 4: Group A and Group C

After the protocol is stable:

- rerun internal ours comparison with trajectory metrics
- run same-source vs real-domain transfer comparison

## Deliverables

Minimum report package:

- one main markdown summary
- one machine-readable summary JSON
- one per-sequence table
- one qualitative figure set

Main table should report trajectory metrics only.

Appendix can report:

- OBB metrics
- pixel metrics
- runtime

## Immediate Next Tasks

1. add a unified trajectory evaluator script under `tools/xt/`
2. define pilot sequence ids for `KBS`
3. reuse current `bbox/seqmask/stack_obb` outputs as input adapters
4. generate the first pilot summary before any full rerun
