r"""Convert a downloaded FiftyOne COCO 2017 split to the native COCO layout.

Input under --source-dirpath (default: FIFTYONE_DATASETS_DIRPATH / "coco-2017"):
    <split>/data/<image file_name>
    <split>/labels.json

Output under --output-dirpath (default: NATIVE_COCO_DATASETS_DIRPATH / "coco-2017"):
    train2017/<image file_name>
    val2017/<image file_name>
    annotations/instances_train2017.json
    annotations/instances_val2017.json

Each invocation converts only the selected split: train -> train2017 or validation -> val2017.
The default split is validation. The JSON is copied unchanged, including category IDs,
image IDs, annotation IDs, bounding boxes, masks and metadata. Image subdirectories
from file_name are preserved. This layout is read by RF-DETR with dataset_file="coco".

Images are relative symlinks by default, so the downloaded source images must remain
available. Use --image-mode copy to create independent image copies. An existing output
split is never overwritten; the other split can be added to the same output root.
COCO test is unsupported because its detection annotations are not publicly available.

Examples (run from the repository root with dependencies already installed;
--no-sync uses the existing environment without installing or updating dependencies):

    # Convert validation using the default paths and relative symlinks.
    uv run --no-sync scripts/datasets/convert_fiftyone_coco_to_native_coco.py

    # Add the downloaded training split to the same output dataset.
    uv run --no-sync scripts/datasets/convert_fiftyone_coco_to_native_coco.py --split train

    # Copy validation images into a separate, independent dataset.
    uv run --no-sync scripts/datasets/convert_fiftyone_coco_to_native_coco.py \
        --split validation --image-mode copy \
        --output-dirpath datasets/native_coco/coco-2017-copy

    # Override the input and output dataset roots.
    uv run --no-sync scripts/datasets/convert_fiftyone_coco_to_native_coco.py \
        --source-dirpath /data/fiftyone/coco-2017 \
        --output-dirpath /data/native_coco/coco-2017 \
        --split validation

    # Show all command-line options and defaults.
    uv run --no-sync scripts/datasets/convert_fiftyone_coco_to_native_coco.py --help
"""

import argparse
from pathlib import Path

from rf_detr_train.consts import FIFTYONE_DATASETS_DIRPATH, NATIVE_COCO_DATASETS_DIRPATH
from rf_detr_train.datasets import DatasetSplit, ImageMode
from rf_detr_train.datasets.coco_conversion import convert_fiftyone_coco
from rf_detr_train.loggers import logger

DEFAULT_SOURCE_DIRPATH = FIFTYONE_DATASETS_DIRPATH / "coco-2017"
DEFAULT_OUTPUT_DIRPATH = NATIVE_COCO_DATASETS_DIRPATH / "coco-2017"
DEFAULT_SPLIT: DatasetSplit = "validation"
DEFAULT_IMAGE_MODE: ImageMode = "symlink"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a downloaded FiftyOne COCO 2017 split to the native COCO directory layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Category IDs are preserved. No test split is generated: COCO test has no public detection annotations.",
    )
    parser.add_argument(
        "--source-dirpath",
        type=Path,
        default=DEFAULT_SOURCE_DIRPATH,
        help="FiftyOne COCO root containing <split>/data/ and <split>/labels.json.",
    )
    parser.add_argument(
        "--output-dirpath",
        type=Path,
        default=DEFAULT_OUTPUT_DIRPATH,
        help="Output root for train2017/, val2017/ and annotations/. Existing output splits are never overwritten.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default=DEFAULT_SPLIT,
        help="Source split to convert. Run again with the other split to add it to the same dataset.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy"),
        default=DEFAULT_IMAGE_MODE,
        help="Link to source images (source must remain available) or copy them for an independent dataset.",
    )
    return parser.parse_args()


def main() -> None:
    """Convert the selected split using the command-line options."""
    args = parse_args()
    try:
        convert_fiftyone_coco(
            source_dirpath=args.source_dirpath,
            output_dirpath=args.output_dirpath,
            split=args.split,
            layout="native",
            image_mode=args.image_mode,
        )
    except (OSError, ValueError) as exc:
        logger.error(str(exc))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
