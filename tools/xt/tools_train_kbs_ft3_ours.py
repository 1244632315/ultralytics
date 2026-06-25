from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "runs" / "obb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune XT_sc endpoint detector on 3-sequence KBS real dataset.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=RUNS / "cmp_xt_sc_endpoint_v1_gpu_full" / "weights" / "best.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "dataset" / "obb_kbs_ft3_notebook_aligned" / "data_xt_sc.yaml",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="cmp_xt_sc_endpoint_v1_kbs_ft3_seed0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights.expanduser().resolve()))
    model.train(
        data=str(args.data.expanduser().resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        momentum=0.937,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.05,
        scale=0.1,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        close_mosaic=0,
        project=str(RUNS),
        name=args.name,
        exist_ok=True,
        save=True,
        val=True,
        verbose=True,
        angle=2.0,
    )


if __name__ == "__main__":
    main()
