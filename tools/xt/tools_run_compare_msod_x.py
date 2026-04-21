from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = REPO_ROOT / "compares" / "MSAMNet-master"
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compare-model train/test on x-only MSOD dataset.")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--model", choices=["CSAUNet", "DNANet", "DnTNet", "MSAMNet"], default="MSAMNet")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset" / "xt_seq_msod_x")
    parser.add_argument("--dataset-name", type=str, default="MSOD_RAWBG")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--t-frame", type=int, default=3)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--suffix", type=str, default=".tif")
    parser.add_argument("--fusionblock", type=str, default="AAFE")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--st-model", type=str, default="")
    parser.add_argument("--smoke-dataset-only", action="store_true")
    parser.add_argument("--smoke-split", choices=["train", "val", "test"], default="train")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    script = "train.py" if args.mode == "train" else "test.py"
    cmd = [
        sys.executable,
        script,
        "--model",
        args.model,
        "--root",
        str(args.dataset_root),
        "--dataset",
        args.dataset_name,
        "--gpus",
        args.gpus,
        "--workers",
        str(args.workers),
        "--T_frame",
        str(args.t_frame),
        "--input_size",
        str(args.input_size),
        "--test_batch_size",
        str(args.test_batch_size),
        "--suffix",
        args.suffix,
    ]
    if args.mode == "train":
        cmd.extend(
            [
                "--epochs",
                str(args.epochs),
                "--train_batch_size",
                str(args.train_batch_size),
            ]
        )
    else:
        if not args.model_dir or not args.st_model:
            raise ValueError("--model-dir and --st-model are required for test mode.")
        cmd.extend(["--model_dir", args.model_dir, "--st_model", args.st_model])
    if args.model == "MSAMNet":
        cmd.extend(["--fusionblock", args.fusionblock])
    return cmd


def run_dataset_smoke(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader

    from model.dataloader import MSODataset

    dataset_dir = args.dataset_root / args.dataset_name
    split_file = dataset_dir / f"{args.smoke_split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    with split_file.open("r", encoding="utf-8") as f:
        img_ids = [line.strip() for line in f if line.strip()]
    if not img_ids:
        raise RuntimeError(f"No sample ids found in {split_file}")

    dataset = MSODataset(
        dataset_dir=str(dataset_dir),
        mode=args.smoke_split,
        img_ids=img_ids[: max(args.test_batch_size, 2)],
        input_size=args.input_size,
        num_frame=args.t_frame,
        suffix=args.suffix,
    )
    loader = DataLoader(dataset=dataset, batch_size=min(args.test_batch_size, len(dataset)), num_workers=0, drop_last=False)
    data, labels = next(iter(loader))
    print(
        {
            "dataset_dir": str(dataset_dir),
            "split": args.smoke_split,
            "items_checked": len(dataset),
            "data_shape": tuple(data.shape),
            "label_shape": tuple(labels.shape),
            "data_min": float(data.min()),
            "data_max": float(data.max()),
            "label_sum": float(labels.sum()),
        }
    )


def main() -> None:
    args = parse_args()
    if args.smoke_dataset_only:
        run_dataset_smoke(args)
        return
    cmd = build_command(args)
    print("cwd:", COMPARE_ROOT)
    print("cmd:", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    subprocess.run(cmd, cwd=COMPARE_ROOT, check=True, env=env)


if __name__ == "__main__":
    main()
