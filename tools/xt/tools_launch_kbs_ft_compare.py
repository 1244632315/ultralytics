from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / "tools" / "xt"
COMPARE_RESULT_ROOT = REPO_ROOT / "compares" / "MSAMNet-master" / "result"

COMPARE_CHECKPOINTS = {
    "CSAUNet": "MSOD_RAWBG_CSAUNet_17_04_2026_11_18_01_wDS/mIoU_best_CSAUNet_MSOD_RAWBG_epoch.pth.tar",
    "DNANet": "MSOD_RAWBG_DNANet_17_04_2026_14_32_33_wDS/mIoU_best_DNANet_MSOD_RAWBG_epoch.pth.tar",
    "DnTNet": "MSOD_RAWBG_DnTNet_18_04_2026_19_38_29_wDS/mIoU_best_DnTNet_MSOD_RAWBG_epoch.pth.tar",
    "MSAMNet": "MSOD_RAWBG_MSAMNet_18_04_2026_08_35_33_wDS/mIoU_best_MSAMNet_MSOD_RAWBG_epoch.pth.tar",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KBS finetune dataset and launch compare-model finetuning jobs.")
    parser.add_argument("--train-json", type=Path, default=REPO_ROOT / "dataset" / "KBS_dataset" / "mosaic+" / "json" / "train.json")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--seq-ids", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-seq-count", type=int, default=0)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--step-size", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("--suffix", type=str, default=".png")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--t-frame", type=int, default=3)
    parser.add_argument("--dataset-tag", type=str, default="")
    parser.add_argument("--models", nargs="+", choices=["CSAUNet", "DNANet", "DnTNet", "MSAMNet"], default=["CSAUNet", "DNANet", "DnTNet", "MSAMNet"])
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def choose_seq_ids(args: argparse.Namespace) -> list[str]:
    if args.seq_ids:
        return [f"{int(x):03d}" for x in args.seq_ids]
    ids = json.loads(args.train_json.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    picked = rng.sample(ids, args.sample_count)
    return [f"{int(x):03d}" for x in picked]


def run(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def main() -> None:
    args = parse_args()
    seq_ids = choose_seq_ids(args)
    sample_count = len(seq_ids)
    dataset_tag = args.dataset_tag.strip() or f"KBS_FT{sample_count}_REAL_P99"
    compare_out_root = REPO_ROOT / "dataset" / "xt_seq_msod_x" / dataset_tag
    ours_out_root = REPO_ROOT / "dataset" / f"obb_kbs_ft{sample_count}_notebook_aligned"
    log_root = REPO_ROOT / "runs" / f"kbs_ft{sample_count}_logs"

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    summary = {
        "seq_ids": seq_ids,
        "sample_count": sample_count,
        "seed": args.seed,
        "dataset_tag": dataset_tag,
        "compare_out_root": str(compare_out_root),
        "ours_out_root": str(ours_out_root),
        "device": args.device,
        "epochs": args.epochs,
        "lr": args.lr,
        "step_size": args.step_size,
        "val_ratio": args.val_ratio,
        "val_seq_count": args.val_seq_count,
        "models": args.models,
    }
    log_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        build_cmd = [
            sys.executable,
            str(TOOLS / "tools_build_kbs_ft3_datasets.py"),
            "--seed",
            str(args.seed),
            "--sample-count",
            str(sample_count),
            "--val-ratio",
            str(args.val_ratio),
            "--val-seq-count",
            str(args.val_seq_count),
            "--seq-ids",
            *seq_ids,
            "--compare-out-root",
            str(compare_out_root),
            "--ours-out-root",
            str(ours_out_root),
        ]
        run(build_cmd, log_root / "00_build_dataset.log", env=env)

    smoke_cmd = [
        sys.executable,
        str(TOOLS / "tools_run_compare_msod_x.py"),
        "--smoke-dataset-only",
        "--smoke-split",
        "train",
        "--dataset-name",
        dataset_tag,
        "--dataset-root",
        str(REPO_ROOT / "dataset" / "xt_seq_msod_x"),
        "--suffix",
        args.suffix,
        "--input-size",
        str(args.input_size),
        "--t-frame",
        str(args.t_frame),
        "--test-batch-size",
        "2",
    ]
    run(smoke_cmd, log_root / "01_compare_smoke.log", env=env)

    for model_name in args.models:
        ckpt_rel = COMPARE_CHECKPOINTS[model_name]
        train_cmd = [
            sys.executable,
            str(TOOLS / "tools_run_compare_msod_x.py"),
            "--mode",
            "train",
            "--model",
            model_name,
            "--dataset-root",
            str(REPO_ROOT / "dataset" / "xt_seq_msod_x"),
            "--dataset-name",
            dataset_tag,
            "--gpus",
            args.device,
            "--epochs",
            str(args.epochs),
            "--train-batch-size",
            str(args.train_batch_size),
            "--test-batch-size",
            str(args.test_batch_size),
            "--workers",
            str(args.workers),
            "--t-frame",
            str(args.t_frame),
            "--input-size",
            str(args.input_size),
            "--suffix",
            args.suffix,
            "--lr",
            str(args.lr),
            "--optimizer",
            "Adagrad",
            "--scheduler",
            "StepLR",
            "--step-size",
            str(args.step_size),
            "--resume",
            ckpt_rel,
        ]
        run(train_cmd, log_root / f"10_train_{model_name}.log", env=env)

    (log_root / "launch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
