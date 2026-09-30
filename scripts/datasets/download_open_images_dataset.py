r"""Download Open Images V7 detection data through the FiftyOne dataset zoo.

Defaults:
    Dataset directory: FIFTYONE_DATASETS_DIRPATH (datasets/fiftyone/).
    Split: test.
    Classes: no class filter.
    Sample limit: none; download the entire selected split.

FiftyOne stores each split under <dataset-dirpath>/open-images-v7/<split>/,
including images and the detection annotations managed by its Open Images downloader.
Supported splits are train, validation and test. This script downloads FiftyOne's
layout; it does not export COCO JSON or prepare RF-DETR's Roboflow directory layout.

Use --classes with Open Images class names, such as Person or "Human body".
Quote names containing spaces; pass multiple names as separate arguments.
Sample selection follows FiftyOne's download rules. Use --max-samples for a limited
download or --all-samples to explicitly remove the sample limit. These two options
are mutually exclusive. FiftyOne manages already downloaded data.

Examples (run from the repository root with dependencies already installed;
--no-sync uses the existing environment without installing or updating dependencies):

    # Download the full test split using the default dataset directory.
    uv run --no-sync scripts/datasets/download_open_images_dataset.py

    # Download up to 100 test images for a quick experiment.
    uv run --no-sync scripts/datasets/download_open_images_dataset.py --max-samples 100

    # Request a subset using exact class names, including a name with spaces.
    uv run --no-sync scripts/datasets/download_open_images_dataset.py \
        --split test --classes Person "Human body" --max-samples 100

    # Download the full training split.
    uv run --no-sync scripts/datasets/download_open_images_dataset.py --split train --all-samples

    # Override the FiftyOne dataset root (open-images-v7/ is created below it).
    uv run --no-sync scripts/datasets/download_open_images_dataset.py \
        --dataset-dirpath /data/fiftyone --split validation --max-samples 100

    # Show all command-line options and defaults.
    uv run --no-sync scripts/datasets/download_open_images_dataset.py --help
"""

import argparse
from pathlib import Path
from typing import Sequence

from rf_detr_train.consts import FIFTYONE_DATASETS_DIRPATH
from rf_detr_train.datasets import DatasetSplit
from rf_detr_train.datasets.fiftyone import download_open_images_dataset
from rf_detr_train.loggers import logger

DEFAULT_DATASET_DIRPATH = FIFTYONE_DATASETS_DIRPATH
DEFAULT_SPLIT: DatasetSplit = "test"
DEFAULT_CLASSES_OF_INTEREST: Sequence[str] | None = None  # ["Person"], ["Person", "Man", "Woman", "Girl", "Boy"]
DEFAULT_MAX_SAMPLES: int | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download the Open Images dataset through FiftyOne.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dirpath",
        type=Path,
        default=DEFAULT_DATASET_DIRPATH,
        help="Directory where FiftyOne stores downloaded datasets.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default=DEFAULT_SPLIT,
        help="Dataset split to download.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES_OF_INTEREST,
        help="Optional class names to download.",
    )
    samples_group = parser.add_mutually_exclusive_group()
    samples_group.add_argument(
        "--max-samples",
        type=int,
        dest="max_samples",
        help="Maximum number of samples to download.",
    )
    samples_group.add_argument(
        "--all-samples",
        action="store_const",
        const=None,
        dest="max_samples",
        help="Download all available samples.",
    )
    parser.set_defaults(max_samples=DEFAULT_MAX_SAMPLES)
    return parser.parse_args()


def main() -> None:
    """Download the requested Open Images split and log its sample count and location."""
    args = parse_args()
    logger.info(
        f"Downloading dataset `Open Images` (split: {args.split}, #samples: {args.max_samples}) "
        f"to path: '{args.dataset_dirpath}'"
    )
    open_images_dataset, open_images_path = download_open_images_dataset(
        dataset_dirpath=args.dataset_dirpath,
        split=args.split,
        classes=args.classes,
        max_samples=args.max_samples,
    )
    logger.info(
        f"Dataset `Open Images` downloaded: {open_images_dataset.num_samples} samples at path: '{open_images_path}'"
    )


if __name__ == "__main__":
    main()
