import argparse
import os
import warnings
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

os.environ.setdefault("WANDB_MODE", "disabled")
SETTINGS.update(wandb=False, raytune=False)

warnings.filterwarnings("ignore")


DATA_YAML = Path("ultralytics/cfg/datasets/coalmine4.yaml")
MODEL_CFG = Path("ultralytics/cfg/models/11/yolo11s-p2-lska-neck.yaml")
PRETRAINED_WEIGHTS = "yolo11s.pt"
DEFAULT_DEVICE = "0"
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO11s + P2 + neck-LSKA model.")
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset yaml path.")
    parser.add_argument("--model", default=str(MODEL_CFG), help="Model yaml or checkpoint path.")
    parser.add_argument(
        "--weights",
        default=PRETRAINED_WEIGHTS,
        help="Pretrained weights to transfer when --model is a yaml. Set empty to train from scratch.",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="runs", help="Output project directory.")
    parser.add_argument("--name", default="yolo11s_p2_lska_neck", help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=24, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible training.")
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Disable deterministic training behavior.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from the latest checkpoint or from the provided checkpoint path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = Path(args.model)
    model = YOLO(args.model)

    if not args.resume and model_path.suffix in {".yaml", ".yml"} and args.weights:
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
    for key in ("epochs", "batch", "imgsz", "workers"):
        value = getattr(args, key)
        if value is not None:
            train_kwargs[key] = value

    if args.resume:
        train_kwargs["resume"] = True if args.resume == "auto" else args.resume

    model.train(**train_kwargs)
