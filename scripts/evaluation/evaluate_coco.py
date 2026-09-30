r"""Evaluate an RF-DETR detection model on native COCO and save the result.

All defaults are defined in EvaluationConfig. Command-line arguments override only
the values explicitly supplied. Without --checkpoint-filepath, the model uses its
official pretrained COCO weights. A custom checkpoint must match the selected
variant and native COCO category IDs.

The dataset root must contain val2017/ and annotations/instances_val2017.json.
Native COCO evaluates this validation set for both split="val" and split="test".
RF-DETR loads the images and annotations through dataset_file="coco".

By default, each run writes results.json containing config and metrics under
RUNS_DIRPATH / "coco" / "rf_detr_<model>_native_coco" / NOW. Use --output-dirpath
to choose another directory. The metrics are exactly those returned by model.evaluate().

Examples (run from the repository root with dependencies already installed;
--no-sync uses the existing environment without installing or updating dependencies):

    # Evaluate with defaults from EvaluationConfig.
    uv run --no-sync scripts/evaluation/evaluate_coco.py

    # Select another model and override evaluation settings.
    uv run --no-sync scripts/evaluation/evaluate_coco.py \
        --model-name medium --batch-size 4 --num-workers 8

    # Use a compatible checkpoint, custom paths and aggregate metrics only.
    uv run --no-sync scripts/evaluation/evaluate_coco.py \
        --model-name small --checkpoint-filepath /data/models/rf-detr-small.pth \
        --dataset-dirpath /data/coco-2017 --output-dirpath /data/results/small \
        --no-log-per-class-metrics

    # Show available arguments and their defaults.
    uv run --no-sync scripts/evaluation/evaluate_coco.py --help
"""

import argparse
from pathlib import Path
from typing import get_args

from rf_detr_train.evaluation.schemas import EvaluationConfig, EvaluationResult
from rf_detr_train.models.rf_detr import build_model
from rf_detr_train.models.types import RFDETRModelName
from rf_detr_train.utils.files import create_directory, save_json


def parse_args() -> argparse.Namespace:
    """Parse explicit overrides; read configuration defaults only to display help.

    :return: Parsed command-line arguments with omitted options left out.
    """
    defaults = EvaluationConfig()
    parser = argparse.ArgumentParser(
        description="Evaluate an RF-DETR detection model on native COCO.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model-name",
        choices=get_args(RFDETRModelName),
        help=f"RF-DETR model variant to evaluate. (default: {defaults.model_name})",
    )
    parser.add_argument(
        "--checkpoint-filepath",
        type=Path,
        help=f"Checkpoint path; omit to use official pretrained weights. (default: {defaults.checkpoint_filepath})",
    )
    parser.add_argument(
        "--device",
        help=f"Device used for evaluation, for example cuda or cpu. (default: {defaults.device})",
    )
    parser.add_argument(
        "--dataset-dirpath",
        type=Path,
        help=f"Native COCO root containing val2017/ and annotations/. (default: {defaults.dataset_dirpath})",
    )
    parser.add_argument(
        "--split",
        choices=get_args(EvaluationConfig.model_fields["split"].annotation),
        help=f"Evaluation mode; native COCO uses validation data in both modes. (default: {defaults.split})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=f"Number of images per evaluation batch. (default: {defaults.batch_size})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help=f"Data loader worker count; zero loads data in the main process. (default: {defaults.num_workers})",
    )
    parser.add_argument(
        "--eval-max-dets",
        type=int,
        help=f"Maximum detections per image and category used by COCO metrics. (default: {defaults.eval_max_dets})",
    )
    parser.add_argument(
        "--log-per-class-metrics",
        action=argparse.BooleanOptionalAction,
        help=f"Report per-class metrics. (default: {defaults.log_per_class_metrics})",
    )
    parser.add_argument(
        "--output-dirpath",
        type=Path,
        help="Results directory. By default, EvaluationConfig derives it from the selected model and run time.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate the configured model and save its configuration and metrics."""
    args = parse_args()
    config = EvaluationConfig(**vars(args))

    create_directory(config.output_dirpath, parents=True, exist_ok=True)

    model = build_model(
        model_name=config.model_name,
        device=config.device,
        checkpoint_filepath=config.checkpoint_filepath,
    )
    metrics = model.evaluate(
        dataset_dir=str(config.dataset_dirpath),
        dataset_file="coco",
        split=config.split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        eval_max_dets=config.eval_max_dets,
        log_per_class_metrics=config.log_per_class_metrics,
    )

    result = EvaluationResult(config=config, metrics=metrics)
    save_json(filepath=config.output_dirpath / "results.json", data=result.model_dump(mode="json"))


if __name__ == "__main__":
    main()
