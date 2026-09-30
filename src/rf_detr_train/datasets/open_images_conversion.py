"""Convert downloaded Open Images person annotations to COCO detection data."""

import csv
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from rf_detr_train.datasets.types import CocoLayout, DatasetSplit, ImageMode
from rf_detr_train.loggers import logger
from rf_detr_train.utils.files import load_json, require_dirpath, save_json
from rf_detr_train.utils.progress import create_progress

PERSON_CLASSES = ("Person", "Man", "Woman", "Girl", "Boy")
PERSON_CATEGORY_ID = 1


def _load_person_boxes(
    split_dirpath: Path | str,
    image_ids: set[str],
) -> tuple[dict[str, set[tuple[float, float, float, float]]], set[str], set[str]]:
    """Read person boxes and identify images with group-of or unknown group-of annotations."""
    split_dirpath = Path(split_dirpath)
    with (split_dirpath / "metadata" / "classes.csv").open(newline="", encoding="utf-8") as file:
        classes = dict(csv.reader(file))
    missing = set(PERSON_CLASSES) - set(classes.values())
    if missing:
        raise ValueError(f"Missing Open Images class definitions: {sorted(missing)}")
    person_ids = {class_id for class_id, name in classes.items() if name in PERSON_CLASSES}
    boxes: dict[str, set[tuple[float, float, float, float]]] = {}
    group_images: set[str] = set()
    unknown_group_images: set[str] = set()
    annotations_filepath = split_dirpath / "labels" / "detections.csv"
    logger.info(f"Reading Open Images annotations from: '{annotations_filepath}'")
    with annotations_filepath.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing detection CSV columns: {sorted(required - set(reader.fieldnames or []))}")
        for line_number, row in enumerate(reader, start=2):
            image_id = row["ImageID"]
            if image_id not in image_ids or row["LabelName"] not in person_ids:
                continue
            if row["IsGroupOf"] == "1":
                group_images.add(image_id)
                continue
            if row["IsGroupOf"] == "-1":
                unknown_group_images.add(image_id)
                continue
            if row["IsGroupOf"] != "0":
                raise ValueError(
                    f"Invalid IsGroupOf {row['IsGroupOf']!r} at CSV line {line_number} (image {image_id})."
                )
            try:
                xmin, ymin, xmax, ymax = (float(row[key]) for key in ("XMin", "YMin", "XMax", "YMax"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid bounding box at CSV line {line_number}.") from exc
            if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
                raise ValueError(f"Invalid normalized bounding box at CSV line {line_number}.")
            # Collapsing person subclasses must not duplicate an identical box.
            boxes.setdefault(image_id, set()).add((xmin, ymin, xmax, ymax))
    return boxes, group_images, unknown_group_images


def convert_fiftyone_open_images(
    source_dirpath: Path | str,
    output_dirpath: Path | str,
    split: DatasetSplit,
    layout: CocoLayout,
    image_mode: ImageMode = "symlink",
) -> Path:
    """Convert one local Open Images split to a single COCO ``person`` category.

    Person, Man, Woman, Girl and Boy map to category ID 1. Images containing any
    group-of (1) or unknown group-of (-1) annotation of these classes are excluded
    entirely, as are images without person annotations. Only confirmed non-group
    (0) person annotations are retained. Other classes do not affect filtering.
    Occluded and truncated individual people are retained. Only identical boxes
    are deduplicated after merging classes; overlapping boxes remain separate.

    :param source_dirpath: FiftyOne Open Images root containing the source splits.
    :param output_dirpath: Output dataset root, separate from the source tree.
        Native test data must use a separate root from native validation data.
    :param split: Source split: train, validation or test. All local images are considered.
    :param layout: Roboflow train/valid/test, or native train2017/val2017.
        Both native validation and test use val2017 for RF-DETR evaluation.
    :param image_mode: Create relative symlinks or copy original image files.
    :return: Path to the generated COCO annotation file.
    :raises FileExistsError: If the output split already exists.
    :raises ValueError: If annotations/options are invalid or no images remain.
    """
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unsupported source split: {split!r}")
    if layout not in ("roboflow", "native"):
        raise ValueError(f"Unsupported COCO layout: {layout!r}")
    if image_mode not in ("symlink", "copy"):
        raise ValueError(f"Unsupported image mode: {image_mode!r}")
    source_dirpath = Path(source_dirpath).expanduser().resolve()
    output_dirpath = Path(output_dirpath).expanduser().resolve()
    if output_dirpath.is_relative_to(source_dirpath) or source_dirpath.is_relative_to(output_dirpath):
        raise ValueError("Source and output directories must be separate, without nesting.")

    split_dirpath = source_dirpath / split
    source_images = require_dirpath(split_dirpath / "data", description="Open Images images")
    if layout == "roboflow":
        output_split = {"train": "train", "validation": "valid", "test": "test"}[split]
        images_destination = output_dirpath / output_split
        annotations_destination = images_destination / "_annotations.coco.json"
        existing_annotations = output_dirpath.glob("*/_annotations.coco.json")
    else:
        output_split = "train2017" if split == "train" else "val2017"
        images_destination = output_dirpath / output_split
        annotations_destination = output_dirpath / "annotations" / f"instances_{output_split}.json"
        existing_annotations = (output_dirpath / "annotations").glob("instances_*.json")
    for path in (images_destination, annotations_destination):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}. Choose another output directory.")
    categories = [{"id": PERSON_CATEGORY_ID, "name": "person"}]
    for filepath in existing_annotations:
        if load_json(filepath).get("categories") != categories:
            raise ValueError(f"Existing split has different category definitions: {filepath}")

    image_paths: dict[str, Path] = {}
    for filepath in sorted(source_images.iterdir()):
        if filepath.suffix.lower() not in (".jpg", ".jpeg", ".png") or not filepath.is_file():
            continue
        if filepath.stem in image_paths:
            raise ValueError(f"Duplicate Open Images image ID: {filepath.stem}")
        image_paths[filepath.stem] = filepath
    boxes, group_images, unknown_group_images = _load_person_boxes(split_dirpath, set(image_paths))
    # Count images with both flags only once, under confirmed groups.
    unknown_group_images -= group_images
    excluded_images = group_images | unknown_group_images
    selected_ids = sorted(set(boxes) - excluded_images)
    without_people = len(set(image_paths) - set(boxes) - excluded_images)
    logger.info(
        f"Open Images {split}: {len(image_paths)} local images, {len(group_images)} excluded with person groups, "
        f"{len(unknown_group_images)} additionally excluded with unknown person group status, "
        f"{without_people} excluded without person boxes, {len(selected_ids)} selected"
    )
    if not selected_ids:
        raise ValueError("No images with individual person annotations remain after filtering.")

    data: dict[str, Any] = {
        "info": {
            "description": "Open Images V7 persons with confirmed non-group annotations",
            "source_split": split,
            "source_classes": list(PERSON_CLASSES),
            "input_images": len(image_paths),
            "excluded_group_images": len(group_images),
            "excluded_unknown_group_images": len(unknown_group_images),
            "excluded_without_person_images": without_people,
        },
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    output_dirpath.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{split}-", dir=output_dirpath) as staging_dir:
        staging_path = Path(staging_dir)
        staging_images = staging_path / "images"
        staging_images.mkdir()
        with create_progress(unit="images", show_metrics=False) as progress:
            for image_id, source_id in enumerate(progress.track(selected_ids, description=f"Converting {split}"), 1):
                source_image = image_paths[source_id]
                with Image.open(source_image) as image:
                    width, height = image.size
                data["images"].append(
                    {"id": image_id, "file_name": source_image.name, "width": width, "height": height}
                )
                for xmin, ymin, xmax, ymax in sorted(boxes[source_id]):
                    bbox = [xmin * width, ymin * height, (xmax - xmin) * width, (ymax - ymin) * height]
                    data["annotations"].append(
                        {
                            "id": len(data["annotations"]) + 1,
                            "image_id": image_id,
                            "category_id": PERSON_CATEGORY_ID,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                        }
                    )
                if image_mode == "copy":
                    shutil.copy2(source_image, staging_images / source_image.name)
                else:
                    # Compute the relative link from its final location, outside staging.
                    (staging_images / source_image.name).symlink_to(os.path.relpath(source_image, images_destination))

        if layout == "roboflow":
            save_json(staging_images / "_annotations.coco.json", data)
            staging_images.rename(images_destination)
        else:
            staged_annotations = staging_path / "labels.json"
            save_json(staged_annotations, data)
            annotations_destination.parent.mkdir(parents=True, exist_ok=True)
            staging_images.rename(images_destination)
            staged_annotations.rename(annotations_destination)
    logger.info(
        f"Saved {len(data['images'])} images and {len(data['annotations'])} person boxes "
        f"at: '{annotations_destination}'"
    )
    return annotations_destination
