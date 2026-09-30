r"""Fine-tune an RF-DETR detection model on a Roboflow COCO dataset.

All defaults are defined in TrainingConfig. Command-line arguments override only
the values explicitly supplied. Without --checkpoint-filepath, the model starts
from its official pretrained COCO weights, cached under MODELS_DIRPATH.
A custom checkpoint supplies initial weights; it does not resume optimizer state.

The dataset root must contain train/ and valid/, each with images and
_annotations.coco.json. RF-DETR reads these through dataset_file="roboflow" and
adjusts the detection head to the dataset's classes. Validation uses valid/;
the test split is not evaluated during this run (run_test=False).

By default, each run saves config.json under RUNS_DIRPATH / run_name / NOW.
The default run_name is rf_detr_<model>. Use --run-name to name an experiment
while keeping the automatic timestamp. An explicit --output-dirpath overrides
the full path; no timestamp is appended. RF-DETR saves its training logs and
checkpoints in the same directory. Validation uses eval_max_dets=100 and enables
per-class metrics. TensorBoard logging is disabled.

Examples (run from the repository root with dependencies already installed;
--no-sync uses the existing environment without installing or updating dependencies):

    # Fine-tune small on Open Images person with defaults from TrainingConfig.
    uv run --no-sync scripts/training/train_rf_detr.py

    # Name an experiment; artifacts go to runs/<run-name>/<date_and_time>/.
    uv run --no-sync scripts/training/train_rf_detr.py \
        --run-name open_images_small_finetuning

    # Select another model and override training settings.
    uv run --no-sync scripts/training/train_rf_detr.py \
        --model-name medium --epochs 30 --batch-size 2 --grad-accum-steps 8 \
        --lr 0.0001 --num-workers 8

    # Start from compatible weights with custom dataset and output paths.
    uv run --no-sync scripts/training/train_rf_detr.py \
        --model-name small --checkpoint-filepath /data/models/rf-detr-small.pth \
        --dataset-dirpath /data/open-images-person --output-dirpath /data/runs/person

    # Show available arguments and their defaults.
    uv run --no-sync scripts/training/train_rf_detr.py --help
"""

import argparse
from pathlib import Path
from typing import get_args

from rf_detr_train.models.rf_detr import build_model
from rf_detr_train.models.types import RFDETRModelName
from rf_detr_train.training.schemas import TrainingConfig
from rf_detr_train.utils.files import create_directory, save_json


def parse_args() -> argparse.Namespace:
    """Parse explicit overrides; read configuration defaults only to display help.

    :return: Parsed command-line arguments with omitted options left out.
    """
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        description="Fine-tune an RF-DETR detection model on a Roboflow COCO dataset.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model-name",
        choices=get_args(RFDETRModelName),
        help=f"RF-DETR model variant to fine-tune. (default: {defaults.model_name})",
    )
    parser.add_argument(
        "--run-name",
        help="Experiment name for automatic result paths. By default, derived from the selected model.",
    )
    parser.add_argument(
        "--checkpoint-filepath",
        type=Path,
        help=f"Initial weights; omit to use official COCO weights. (default: {defaults.checkpoint_filepath})",
    )
    parser.add_argument(
        "--device",
        help=f"Device used for training, for example cuda or cpu. (default: {defaults.device})",
    )
    parser.add_argument(
        "--dataset-dirpath",
        type=Path,
        help=f"Roboflow COCO root containing train/ and valid/. (default: {defaults.dataset_dirpath})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help=f"Number of training epochs. (default: {defaults.epochs})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=f"Number of images per batch on each device. (default: {defaults.batch_size})",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        help=f"Number of batches accumulated per optimizer step. (default: {defaults.grad_accum_steps})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help=f"Learning rate. (default: {defaults.lr})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help=f"Data loader worker count; zero loads data in the main process. (default: {defaults.num_workers})",
    )
    parser.add_argument(
        "--output-dirpath",
        type=Path,
        help="Exact run directory; overrides run_name and the automatic timestamp in the path.",
    )
    return parser.parse_args()


def main() -> None:
    """Save the run configuration and fine-tune the configured model."""
    args = parse_args()
    config = TrainingConfig(**vars(args))

    create_directory(config.output_dirpath, parents=True, exist_ok=True)
    save_json(filepath=config.output_dirpath / "config.json", data=config.model_dump(mode="json"))

    model = build_model(
        model_name=config.model_name,
        device=config.device,
        checkpoint_filepath=config.checkpoint_filepath,
    )
    model.train(
        dataset_dir=str(config.dataset_dirpath),
        dataset_file="roboflow",
        output_dir=str(config.output_dirpath),
        device=config.device,
        epochs=config.epochs,
        batch_size=config.batch_size,
        grad_accum_steps=config.grad_accum_steps,
        lr=config.lr,
        num_workers=config.num_workers,
        eval_max_dets=100,
        log_per_class_metrics=True,
        tensorboard=False,
        run_test=False,
    )


if __name__ == "__main__":
    main()
