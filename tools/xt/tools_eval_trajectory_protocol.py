from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trajectory-level metrics from existing bbox-style track summary JSON files."
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Named input in the form method=/abs/or/rel/path/to/summary.json",
    )
    parser.add_argument("--det-tc", type=float, default=0.5, help="TC threshold for detected trajectory.")
    parser.add_argument("--complete-tc", type=float, default=0.8, help="TC threshold for complete trajectory.")
    parser.add_argument("--frag-iou", type=float, default=0.1, help="Minimum bbox IoU to count as a fragment candidate.")
    parser.add_argument(
        "--frag-min-frame-overlap",
        type=int,
        default=1,
        help="Minimum shared frames between GT and prediction to count as a fragment candidate.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs") / "trajectory_eval",
        help="Output directory for protocol summaries.",
    )
    parser.add_argument("--run-name", type=str, default="protocol_frame_support_v1")
    parser.add_argument("--seq-ids", nargs="*", default=None, help="Optional sequence ids to evaluate, e.g. 059 027 006.")
    return parser.parse_args()


def parse_named_input(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.stem, path
    name, path = raw.split("=", 1)
    return name.strip(), Path(path.strip())


def load_json(path: Path) -> dict:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def box_iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def box_gt_coverage_xyxy(gt_box: list[float], pred_box: list[float]) -> float:
    gx1, gy1, gx2, gy2 = gt_box
    px1, py1, px2, py2 = pred_box
    ix1 = max(gx1, px1)
    iy1 = max(gy1, py1)
    ix2 = min(gx2, px2)
    iy2 = min(gy2, py2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    return float(inter / gt_area) if gt_area > 0 else 0.0


def frame_set(track: dict) -> set[int]:
    return {int(x) for x in track.get("frames", [])}


def frame_completeness(gt_track: dict, pred_track: dict | None) -> float:
    if pred_track is None:
        return 0.0
    gt_frames = frame_set(gt_track)
    pred_frames = frame_set(pred_track)
    if not gt_frames:
        return 0.0
    return float(len(gt_frames & pred_frames) / max(1, len(gt_frames)))


def endpoint_hit(gt_track: dict, pred_track: dict | None) -> bool:
    if pred_track is None:
        return False
    gt_frames = sorted(frame_set(gt_track))
    pred_frames = frame_set(pred_track)
    if not gt_frames:
        return False
    return gt_frames[0] in pred_frames and gt_frames[-1] in pred_frames


def choose_match(
    gt_idx: int,
    gt_track: dict,
    gt_box: list[float],
    pred_tracks: list[dict],
    pred_boxes: list[list[float]],
    matches: list[dict],
) -> tuple[int | None, dict | None, float]:
    for row in matches:
        if int(row["gt_idx"]) == gt_idx:
            pred_idx = int(row["pred_idx"])
            pred_track = pred_tracks[pred_idx]
            return pred_idx, pred_track, frame_completeness(gt_track, pred_track)

    best_idx = None
    best_track = None
    best_score = -1.0
    for pred_idx, (pred_track, pred_box) in enumerate(zip(pred_tracks, pred_boxes)):
        tc = frame_completeness(gt_track, pred_track)
        if tc <= 0:
            continue
        score = tc + 0.05 * box_iou_xyxy(gt_box, pred_box)
        if score > best_score:
            best_score = score
            best_idx = pred_idx
            best_track = pred_track
    if best_idx is None:
        return None, None, 0.0
    return best_idx, best_track, frame_completeness(gt_track, best_track)


def choose_match_box_only(
    gt_idx: int,
    gt_box: list[float],
    pred_boxes: list[list[float]],
    matches: list[dict],
) -> tuple[int | None, float]:
    for row in matches:
        if int(row["gt_idx"]) == gt_idx:
            pred_idx = int(row["pred_idx"])
            return pred_idx, box_gt_coverage_xyxy(gt_box, pred_boxes[pred_idx])

    best_idx = None
    best_cov = 0.0
    for pred_idx, pred_box in enumerate(pred_boxes):
        cov = box_gt_coverage_xyxy(gt_box, pred_box)
        if cov > best_cov:
            best_cov = cov
            best_idx = pred_idx
    if best_idx is None or best_cov <= 0:
        return None, 0.0
    return best_idx, float(best_cov)


def fragment_count(
    gt_track: dict,
    gt_box: list[float],
    pred_tracks: list[dict],
    pred_boxes: list[list[float]],
    min_iou: float,
    min_frame_overlap: int,
) -> int:
    gt_frames = frame_set(gt_track)
    count = 0
    for pred_track, pred_box in zip(pred_tracks, pred_boxes):
        shared = len(gt_frames & frame_set(pred_track))
        if shared < min_frame_overlap:
            continue
        if box_iou_xyxy(gt_box, pred_box) < min_iou:
            continue
        count += 1
    return count


def fragment_count_box_only(
    gt_box: list[float],
    pred_boxes: list[list[float]],
    min_iou: float,
) -> int:
    count = 0
    for pred_box in pred_boxes:
        if box_iou_xyxy(gt_box, pred_box) >= min_iou:
            count += 1
    return count


def evaluate_summary(
    name: str,
    data: dict,
    det_tc: float,
    complete_tc: float,
    frag_iou: float,
    frag_min_frame_overlap: int,
    seq_filter: set[str] | None,
) -> dict:
    per_sequence = data.get("per_sequence", {})
    total_gt = 0
    total_pred = 0
    total_frames = 0
    detected = 0
    complete = 0
    endpoint_hits = 0
    total_tc = 0.0
    total_frag = 0
    num_gt_with_frag_gt_1 = 0
    total_false_tracks = 0
    tp = fp = fn = 0
    seq_rows = {}
    num_sequences = 0
    support_mode_counts = {"frame_support": 0, "box_match_proxy": 0}

    for seq_id, row in per_sequence.items():
        if seq_filter is not None and str(seq_id) not in seq_filter:
            continue
        num_sequences += 1
        gt_tracks = row.get("gt_tracks", [])
        pred_tracks = row.get("pred_tracks", [])
        gt_boxes = row.get("gt_boxes_xyxy", [])
        pred_boxes = row.get("pred_boxes_xyxy", [])
        matches = row.get("matches", [])
        num_frames = int(row.get("num_frames", 0))
        seq_false_tracks = max(0, len(pred_tracks) - len({int(m["pred_idx"]) for m in matches}))
        pred_track_mode = str(row.get("pred_track_mode", "")).strip().lower()
        support_mode = "frame_support" if (pred_track_mode == "reconstruct" or pred_tracks) else "box_match_proxy"
        if support_mode == "box_match_proxy":
            seq_false_tracks = max(0, len(pred_boxes) - len({int(m["pred_idx"]) for m in matches}))
        support_mode_counts[support_mode] += 1

        total_gt += len(gt_tracks)
        total_pred += len(pred_tracks) if pred_tracks else len(pred_boxes)
        total_frames += num_frames
        total_false_tracks += seq_false_tracks
        tp += int(row.get("tp", 0))
        fp += int(row.get("fp", 0))
        fn += int(row.get("fn", 0))

        gt_rows = []
        seq_detected = 0
        seq_complete = 0
        seq_endpoint_hits = 0
        seq_total_tc = 0.0
        seq_total_frag = 0

        for gt_idx, (gt_track, gt_box) in enumerate(zip(gt_tracks, gt_boxes)):
            if support_mode == "frame_support":
                pred_idx, pred_track, tc = choose_match(gt_idx, gt_track, gt_box, pred_tracks, pred_boxes, matches)
                ehr = endpoint_hit(gt_track, pred_track)
                frag = fragment_count(
                    gt_track,
                    gt_box,
                    pred_tracks,
                    pred_boxes,
                    min_iou=frag_iou,
                    min_frame_overlap=frag_min_frame_overlap,
                )
                pred_frames = [] if pred_track is None else list(pred_track.get("frames", []))
            else:
                pred_idx, tc = choose_match_box_only(gt_idx, gt_box, pred_boxes, matches)
                ehr = False
                frag = fragment_count_box_only(gt_box, pred_boxes, min_iou=frag_iou)
                pred_frames = []
            seq_total_tc += tc
            seq_total_frag += frag
            total_tc += tc
            total_frag += frag
            if tc >= det_tc:
                detected += 1
                seq_detected += 1
            if tc >= complete_tc:
                complete += 1
                seq_complete += 1
            if ehr:
                endpoint_hits += 1
                seq_endpoint_hits += 1
            if frag > 1:
                num_gt_with_frag_gt_1 += 1
            gt_rows.append(
                {
                    "gt_track_id": int(gt_track.get("track_id", gt_idx)),
                    "matched_pred_idx": pred_idx,
                    "tc": float(tc),
                    "detected": bool(tc >= det_tc),
                    "complete": bool(tc >= complete_tc),
                    "endpoint_hit": bool(ehr),
                    "fragment_count": int(frag),
                    "gt_frames": list(gt_track.get("frames", [])),
                    "pred_frames": pred_frames,
                }
            )

        seq_num_gt = len(gt_tracks)
        seq_rows[seq_id] = {
            "support_mode": support_mode,
            "num_frames": num_frames,
            "num_gt_tracks": seq_num_gt,
            "num_pred_tracks": len(pred_tracks) if pred_tracks else len(pred_boxes),
            "false_tracks": seq_false_tracks,
            "tdr": float(seq_detected / seq_num_gt) if seq_num_gt else 1.0,
            "ctr": float(seq_complete / seq_num_gt) if seq_num_gt else 1.0,
            "mean_tc": float(seq_total_tc / seq_num_gt) if seq_num_gt else 1.0,
            "ehr": float(seq_endpoint_hits / seq_num_gt) if seq_num_gt else 1.0,
            "mean_frag_per_gt": float(seq_total_frag / seq_num_gt) if seq_num_gt else 0.0,
            "gt_rows": gt_rows,
        }

    metrics = {
        "num_sequences": num_sequences,
        "num_frames": total_frames,
        "num_gt_tracks": total_gt,
        "num_pred_tracks": total_pred,
        "tp_box_match": tp,
        "fp_box_match": fp,
        "fn_box_match": fn,
        "support_mode_counts": support_mode_counts,
        "tdr": float(detected / total_gt) if total_gt else 1.0,
        "ctr": float(complete / total_gt) if total_gt else 1.0,
        "mean_tc": float(total_tc / total_gt) if total_gt else 1.0,
        "ehr": float(endpoint_hits / total_gt) if total_gt else 1.0,
        "mean_frag_per_gt": float(total_frag / total_gt) if total_gt else 0.0,
        "num_gt_with_frag_gt_1": int(num_gt_with_frag_gt_1),
        "ftps": float(total_false_tracks / max(1, num_sequences)),
        "false_tracks_per_frame": float(total_false_tracks / max(1, total_frames)),
        "track_precision": float(detected / max(1, detected + total_false_tracks)),
    }
    return {
        "method": name,
        "source_summary": str(data.get("summary_path", "")),
        "protocol_version": "trajectory_frame_support_v1",
        "thresholds": {
            "det_tc": float(det_tc),
            "complete_tc": float(complete_tc),
            "frag_iou": float(frag_iou),
            "frag_min_frame_overlap": int(frag_min_frame_overlap),
        },
        "metrics": metrics,
        "per_sequence": seq_rows,
    }


def make_markdown(rows: list[dict]) -> str:
    lines = [
        "# Trajectory Protocol Frame-Support Summary",
        "",
        "| Method | TDR | CTR | mean_TC | EHR | mean_Frag | FTPS | Track Precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['method']} | {m['tdr']:.4f} | {m['ctr']:.4f} | {m['mean_tc']:.4f} | "
            f"{m['ehr']:.4f} | {m['mean_frag_per_gt']:.4f} | {m['ftps']:.4f} | {m['track_precision']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Protocol notes:",
            "",
            "- This is the phase-1 frame-support version of the trajectory protocol.",
            "- `TC` is computed from GT/predicted frame support overlap.",
            "- `EHR` checks whether both GT endpoints are covered in frame support.",
            "- `Frag` counts prediction tracks with shared frames and minimum bbox IoU support.",
            "- If a summary has no `pred_tracks`, the script falls back to `box_match_proxy` for that method.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out_root = args.out_root / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for raw in args.inputs:
        name, path = parse_named_input(raw)
        data = load_json(path)
        result = evaluate_summary(
            name=name,
            data=data,
            det_tc=float(args.det_tc),
            complete_tc=float(args.complete_tc),
            frag_iou=float(args.frag_iou),
            frag_min_frame_overlap=int(args.frag_min_frame_overlap),
            seq_filter=None if not args.seq_ids else {str(x) for x in args.seq_ids},
        )
        result["source_summary"] = str(path.expanduser().resolve())
        rows.append(result)

    rows.sort(key=lambda row: row["metrics"]["tdr"], reverse=True)
    summary = {
        "protocol_version": "trajectory_frame_support_v1",
        "num_methods": len(rows),
        "methods": rows,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = out_root / "report.md"
    report_path.write_text(make_markdown(rows), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"saved {summary_path}")
    print(f"saved {report_path}")


if __name__ == "__main__":
    main()
