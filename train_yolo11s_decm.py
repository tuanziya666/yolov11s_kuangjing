import argparse
import os
import warnings
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("PIN_MEMORY", "False")
    os.environ.setdefault("PYTORCH_JIT", "0")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

os.environ.setdefault("WANDB_MODE", "disabled")
SETTINGS.update(wandb=False, raytune=False)

warnings.filterwarnings("ignore")


DATA_YAML = Path("ultralytics/cfg/datasets/coalmine4.yaml")
MODEL_CFG = Path("ultralytics/cfg/models/11/yolo11s-DECM-LDC.yaml")
MODEL_WEIGHTS = "yolo11s.pt"
DEFAULT_DEVICE = "0"
DEFAULT_SEED = 42
TRAIN_DEFAULTS = {"epochs": 100, "batch": 8, "imgsz": 640, "workers": 0}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO11s DECM variant.")
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset yaml path.")
    parser.add_argument(
        "--model",
        default=str(MODEL_CFG),
        help="Model yaml or checkpoint path. Point this to DECM-L / LD / LDC yaml or a checkpoint for resume.",
    )
    parser.add_argument(
        "--weights",
        default=MODEL_WEIGHTS,
        help="Pretrained weights to initialize from when model is a yaml file.",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="runs", help="Output project directory.")
    parser.add_argument("--name", default="yolo11s_decm", help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=None, help="Input image size.")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible training.")
    parser.add_argument("--non-deterministic", action="store_true", help="Disable deterministic training behavior.")
    parser.add_argument("--resume", action="store_true", help="Resume from the provided --model checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = YOLO(args.model)
    if not args.resume and str(args.model).lower().endswith((".yaml", ".yml")):
        model = model.load(args.weights)

    train_kwargs = {
        "data": args.data,
        "task": "detect",
        "project": args.project,
        "name": args.name,
        "device": args.device,
        "seed": args.seed,
        "deterministic": not args.non_deterministic,
    }

    if args.resume:
        train_kwargs["resume"] = True
    else:
        for key in ("epochs", "batch", "imgsz", "workers"):
            value = getattr(args, key)
            train_kwargs[key] = TRAIN_DEFAULTS[key] if value is None else value

    model.train(**train_kwargs)
