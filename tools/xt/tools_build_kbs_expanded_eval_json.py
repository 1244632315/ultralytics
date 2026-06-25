from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an expanded KBS evaluation json by excluding sequences used in finetuning."
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "train.json",
    )
    parser.add_argument(
        "--test-json",
        type=Path,
        default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "test.json",
    )
    parser.add_argument(
        "--exclude-summary",
        type=Path,
        action="append",
        default=None,
        help="Summary json that contains seq_ids/selected_seq_ids/train_seq_ids/val_seq_ids to exclude.",
    )
    parser.add_argument(
        "--exclude-seq-id",
        nargs="*",
        default=None,
        help="Optional explicit sequence ids such as 050 061 015.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "test_expanded_excluding_ft10.json",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "test_expanded_excluding_ft10_summary.json",
    )
    return parser.parse_args()


def load_ids(path: Path) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [f"{int(x):03d}" for x in rows]


def collect_excluded_ids(summary_path: Path) -> set[str]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    excluded: set[str] = set()
    for key in ["seq_ids", "selected_seq_ids", "train_seq_ids", "val_seq_ids"]:
        for row in data.get(key, []):
            excluded.add(f"{int(row):03d}")
    return excluded


def main() -> None:
    args = parse_args()

    train_ids = set(load_ids(args.train_json))
    test_ids = set(load_ids(args.test_json))
    all_ids = train_ids | test_ids

    exclude_summaries = [p.expanduser().resolve() for p in (args.exclude_summary or [])]
    excluded_ids: set[str] = set()
    for summary_path in exclude_summaries:
        excluded_ids |= collect_excluded_ids(summary_path)
    for row in args.exclude_seq_id or []:
        excluded_ids.add(f"{int(row):03d}")

    expanded_ids = sorted(all_ids - excluded_ids, key=int)
    expanded_numeric = [int(x) for x in expanded_ids]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(expanded_numeric, indent=2), encoding="utf-8")

    summary = {
        "train_json": str(args.train_json.expanduser().resolve()),
        "test_json": str(args.test_json.expanduser().resolve()),
        "exclude_summaries": [str(p) for p in exclude_summaries],
        "explicit_exclude_seq_ids": sorted(args.exclude_seq_id or [], key=int),
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "all_kbs_count": len(all_ids),
        "excluded_count": len(excluded_ids),
        "expanded_eval_count": len(expanded_ids),
        "excluded_seq_ids": sorted(excluded_ids, key=int),
        "expanded_eval_seq_ids": expanded_ids,
        "out_json": str(args.out_json.expanduser().resolve()),
    }
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
