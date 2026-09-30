r"""Convert downloaded Open Images V7 persons to COCO for RF-DETR.

Input under --source-dirpath (default: FIFTYONE_DATASETS_DIRPATH / "open-images-v7"):
    <split>/data/<image_id>.jpg
    <split>/labels/detections.csv
    <split>/metadata/classes.csv

Person, Man, Woman, Girl and Boy are merged into one category: person, ID 1.
The script considers all locally downloaded images in the selected split. It
excludes entire images with any IsGroupOf=1 (group) or IsGroupOf=-1 (unknown)
annotation of these five classes, and images without person boxes. Only images
whose person annotations all have IsGroupOf=0 are retained. Unknown-status images
are counted separately, excluding those already counted as containing groups.
Human body and other classes are not exported.
Identical boxes are deduplicated; occluded/truncated people and overlapping boxes
are retained. Original images are not resized or re-encoded. No data is downloaded.

Defaults: split=validation, layout=roboflow, image-mode=symlink.
One invocation converts one split. Relative symlinks require the source images
to remain available; --image-mode copy produces an independent copy.
Existing output splits are never overwritten.

Roboflow output (default: ROBOFLOW_DATASETS_DIRPATH / "open-images-v7-person"):
    train/<images> and train/_annotations.coco.json
    valid/<images> and valid/_annotations.coco.json
    test/<images> and test/_annotations.coco.json

Native output (default: NATIVE_COCO_DATASETS_DIRPATH / "open-images-v7-person"):
    train -> train2017/ and annotations/instances_train2017.json
    validation -> val2017/ and annotations/instances_val2017.json
    test -> val2017/ and annotations/instances_val2017.json in a SEPARATE dataset
        root, default: NATIVE_COCO_DATASETS_DIRPATH / "open-images-v7-person-test".
The native test view uses val2017 because RF-DETR's native COCO evaluation reads
that directory. It still contains the original Open Images test samples.

JSON category ID 1 is preserved in both layouts. Use native COCO to evaluate the
original COCO checkpoint (person output ID 1). The Roboflow loader maps person to
model index 0 for one-class fine-tuning and subsequent evaluation. Evaluating the
original COCO checkpoint directly with the Roboflow loader misaligns class IDs.
Metrics describe this filtered subset using the COCO protocol, not official OI mAP.

Examples (run from the repository root with dependencies already installed;
--no-sync uses the existing environment without installing or updating dependencies):

    # Prepare the three splits for one-class fine-tuning.
    uv run --no-sync scripts/datasets/convert_fiftyone_open_images_to_coco.py --split train
    uv run --no-sync scripts/datasets/convert_fiftyone_open_images_to_coco.py --split validation
    uv run --no-sync scripts/datasets/convert_fiftyone_open_images_to_coco.py --split test

    # Export the same test samples for evaluating the original COCO checkpoint.
    uv run --no-sync scripts/datasets/convert_fiftyone_open_images_to_coco.py --split test --layout native

    # Copy images instead of linking them and override the output directory.
    uv run --no-sync scripts/datasets/convert_fiftyone_open_images_to_coco.py \
        --split test --image-mode copy --output-dirpath /data/open-images-person-copy

    uv run --no-sync scripts/datasets/convert_fiftyone_open_images_to_coco.py --help
"""

import argparse
from pathlib import Path

from rf_detr_train.consts import FIFTYONE_DATASETS_DIRPATH, NATIVE_COCO_DATASETS_DIRPATH, ROBOFLOW_DATASETS_DIRPATH
from rf_detr_train.datasets import CocoLayout, DatasetSplit, ImageMode
from rf_detr_train.datasets.open_images_conversion import convert_fiftyone_open_images
from rf_detr_train.loggers import logger

DEFAULT_SOURCE_DIRPATH = FIFTYONE_DATASETS_DIRPATH / "open-images-v7"
DEFAULT_DATASET_NAME = "open-images-v7-person"
DEFAULT_SPLIT: DatasetSplit = "validation"
DEFAULT_LAYOUT: CocoLayout = "roboflow"
DEFAULT_IMAGE_MODE: ImageMode = "symlink"


def parse_args() -> argparse.Namespace:
    """Parse conversion options; derive the default output root from layout and split."""
    parser = argparse.ArgumentParser(
        description="Convert local Open Images persons to COCO, keeping only confirmed non-group person annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dirpath", type=Path, default=DEFAULT_SOURCE_DIRPATH, help="FiftyOne Open Images dataset root."
    )
    parser.add_argument(
        "--output-dirpath",
        type=Path,
        help="Output root; defaults to datasets/{roboflow,native_coco}/open-images-v7-person "
        "depending on layout. Native test uses open-images-v7-person-test instead.",
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default=DEFAULT_SPLIT, help="Downloaded split to convert."
    )
    parser.add_argument(
        "--layout", choices=("roboflow", "native"), default=DEFAULT_LAYOUT, help="Output directory layout."
    )
    parser.add_argument(
        "--image-mode", choices=("symlink", "copy"), default=DEFAULT_IMAGE_MODE, help="How to export image files."
    )
    args = parser.parse_args()
    if args.output_dirpath is None:
        root = ROBOFLOW_DATASETS_DIRPATH if args.layout == "roboflow" else NATIVE_COCO_DATASETS_DIRPATH
        name = DEFAULT_DATASET_NAME
        if args.layout == "native" and args.split == "test":
            name = f"{name}-test"
        args.output_dirpath = root / name
    return args


def main() -> None:
    """Convert the requested split and report invalid inputs without overwriting data."""
    args = parse_args()
    try:
        convert_fiftyone_open_images(
            source_dirpath=args.source_dirpath,
            output_dirpath=args.output_dirpath,
            split=args.split,
            layout=args.layout,
            image_mode=args.image_mode,
        )
    except (OSError, ValueError) as exc:
        logger.error(str(exc))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
