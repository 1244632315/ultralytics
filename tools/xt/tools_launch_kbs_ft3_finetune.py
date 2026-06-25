from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / "tools" / "xt"
RUN_LOG_ROOT = REPO_ROOT / "runs" / "kbs_ft3_logs"

COMPARE_CHECKPOINTS = {
    "CSAUNet": "MSOD_RAWBG_CSAUNet_17_04_2026_11_18_01_wDS/mIoU_best_CSAUNet_MSOD_RAWBG_epoch.pth.tar",
    "DNANet": "MSOD_RAWBG_DNANet_17_04_2026_14_32_33_wDS/mIoU_best_DNANet_MSOD_RAWBG_epoch.pth.tar",
    "DnTNet": "MSOD_RAWBG_DnTNet_18_04_2026_19_38_29_wDS/mIoU_best_DnTNet_MSOD_RAWBG_epoch.pth.tar",
    "MSAMNet": "MSOD_RAWBG_MSAMNet_18_04_2026_08_35_33_wDS/mIoU_best_MSAMNet_MSOD_RAWBG_epoch.pth.tar",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 3-sequence KBS finetune datasets and launch all finetune jobs.")
    parser.add_argument("--seq-ids", nargs="*", default=["050", "061", "015"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--compare-epochs", type=int, default=30)
    parser.add_argument("--ours-epochs", type=int, default=30)
    parser.add_argument("--compare-lr", type=float, default=0.001)
    parser.add_argument("--compare-step-size", type=int, default=15)
    parser.add_argument("--ours-batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


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
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    RUN_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "seq_ids": [f"{int(x):03d}" for x in args.seq_ids],
        "seed": args.seed,
        "device": args.device,
        "compare_epochs": args.compare_epochs,
        "ours_epochs": args.ours_epochs,
    }

    if not args.skip_build:
        build_cmd = [
            sys.executable,
            str(TOOLS / "tools_build_kbs_ft3_datasets.py"),
            "--seed",
            str(args.seed),
            "--seq-ids",
            *summary["seq_ids"],
        ]
        run(build_cmd, RUN_LOG_ROOT / "00_build_dataset.log", env=env)

    smoke_cmd = [
        sys.executable,
        str(TOOLS / "tools_run_compare_msod_x.py"),
        "--smoke-dataset-only",
        "--smoke-split",
        "train",
        "--dataset-name",
        "KBS_FT3_REAL_P99",
        "--dataset-root",
        str(REPO_ROOT / "dataset" / "xt_seq_msod_x"),
        "--suffix",
        ".png",
        "--input-size",
        "512",
        "--t-frame",
        "3",
        "--test-batch-size",
        "2",
    ]
    run(smoke_cmd, RUN_LOG_ROOT / "01_compare_smoke.log", env=env)

    for model_name, ckpt_rel in COMPARE_CHECKPOINTS.items():
        log_name = f"10_train_{model_name}.log"
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
            "KBS_FT3_REAL_P99",
            "--gpus",
            args.device,
            "--epochs",
            str(args.compare_epochs),
            "--train-batch-size",
            "4",
            "--test-batch-size",
            "4",
            "--workers",
            str(args.workers),
            "--t-frame",
            "3",
            "--input-size",
            "512",
            "--suffix",
            ".png",
            "--lr",
            str(args.compare_lr),
            "--optimizer",
            "Adagrad",
            "--scheduler",
            "StepLR",
            "--step-size",
            str(args.compare_step_size),
            "--resume",
            ckpt_rel,
        ]
        run(train_cmd, RUN_LOG_ROOT / log_name, env=env)

    ours_cmd = [
        sys.executable,
        str(TOOLS / "tools_train_kbs_ft3_ours.py"),
        "--epochs",
        str(args.ours_epochs),
        "--batch",
        str(args.ours_batch),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
    ]
    run(ours_cmd, RUN_LOG_ROOT / "20_train_ours.log", env=env)

    (RUN_LOG_ROOT / "launch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
