r"""Download COCO 2017 detection data through the FiftyOne dataset zoo.

Defaults:
    Dataset directory: FIFTYONE_DATASETS_DIRPATH (datasets/fiftyone/).
    Split: validation.
    Classes: no class filter.
    Sample limit: none; download the entire selected split.

FiftyOne stores each split under <dataset-dirpath>/coco-2017/<split>/.
Labelled train/validation splits contain data/ with images and labels.json with
COCO annotations. COCO test images have no public detection ground truth.
This script downloads FiftyOne's layout; conversion to RF-DETR layouts is separate.

Use --classes to request images containing the given COCO class names, such as
person or car. This selects images; it does not remap category IDs or guarantee
that their annotations contain only the requested classes. Use --max-samples
for a limited download or --all-samples to explicitly remove the sample limit.
These two options are mutually exclusive. FiftyOne manages already downloaded data.

Examples (run from the repository root with dependencies already installed;
--no-sync uses the existing environment without installing or updating dependencies):

    # Download the full validation split using the default dataset directory.
    uv run --no-sync scripts/datasets/download_coco_dataset.py

    # Download up to 100 validation images for a quick experiment.
    uv run --no-sync scripts/datasets/download_coco_dataset.py --max-samples 100

    # Request a validation subset containing person/car annotations.
    uv run --no-sync scripts/datasets/download_coco_dataset.py \
        --split validation --classes person car --max-samples 100

    # Download the full training split.
    uv run --no-sync scripts/datasets/download_coco_dataset.py --split train --all-samples

    # Override the FiftyOne dataset root (coco-2017/ is created below it).
    uv run --no-sync scripts/datasets/download_coco_dataset.py \
        --dataset-dirpath /data/fiftyone --split validation --max-samples 100

    # Show all command-line options and defaults.
    uv run --no-sync scripts/datasets/download_coco_dataset.py --help
"""

import argparse
from pathlib import Path
from typing import Sequence

from rf_detr_train.consts import FIFTYONE_DATASETS_DIRPATH
from rf_detr_train.datasets import DatasetSplit
from rf_detr_train.datasets.fiftyone import download_coco_dataset
from rf_detr_train.loggers import logger

DEFAULT_DATASET_DIRPATH = FIFTYONE_DATASETS_DIRPATH
DEFAULT_SPLIT: DatasetSplit = "validation"
DEFAULT_CLASSES_OF_INTEREST: Sequence[str] | None = None  # ["person"]
DEFAULT_MAX_SAMPLES: int | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download the COCO dataset through FiftyOne.",
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
    """Download the requested COCO split and log its sample count and location."""
    args = parse_args()
    logger.info(
        f"Downloading dataset `COCO` (split: {args.split}, #samples: {args.max_samples}) "
        f"to path: '{args.dataset_dirpath}'"
    )
    coco_dataset, coco_path = download_coco_dataset(
        dataset_dirpath=args.dataset_dirpath,
        split=args.split,
        classes=args.classes,
        max_samples=args.max_samples,
    )
    logger.info(f"Dataset `COCO` downloaded: {coco_dataset.num_samples} samples at path: '{coco_path}'")


if __name__ == "__main__":
    main()
