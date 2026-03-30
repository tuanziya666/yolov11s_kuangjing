import argparse
import os
import warnings
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("PIN_MEMORY", "False")

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ["ULTRALYTICS_IOU_LOSS"] = "nwd"
SETTINGS.update(wandb=False, raytune=False)

warnings.filterwarnings("ignore")


DATA_YAML = Path("ultralytics/cfg/datasets/coalmine4.yaml")
MODEL_WEIGHTS = "yolo11s.pt"
DEFAULT_DEVICE = "0"
DEFAULT_NWD_TAU = 12.8
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO11s baseline model with NWD bbox regression loss.")
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset yaml path.")
    parser.add_argument("--model", default=MODEL_WEIGHTS, help="Model weights or yaml.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="runs", help="Output project directory.")
    parser.add_argument("--name", default="yolo11s_nwd", help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=24, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible training.")
    parser.add_argument("--nwd-tau", type=float, default=DEFAULT_NWD_TAU, help="Normalization constant for NWD.")
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Disable deterministic training behavior.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from the provided model path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["ULTRALYTICS_NWD_TAU"] = str(args.nwd_tau)
    model = YOLO(args.model)

    train_kwargs = {
        "data": args.data,
        "task": "detect",
        "project": args.project,
        "name": args.name,
        "device": args.device,
        "seed": args.seed,
        "deterministic": not args.non_deterministic,
    }
    for key in ("epochs", "batch", "imgsz", "workers"):
        value = getattr(args, key)
        if value is not None:
            train_kwargs[key] = value

    if args.resume:
        train_kwargs["resume"] = True

    model.train(**train_kwargs)
