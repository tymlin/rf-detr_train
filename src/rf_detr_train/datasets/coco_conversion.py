from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from rf_detr_train.datasets.types import DatasetSplit
from rf_detr_train.loggers import logger
from rf_detr_train.utils.files import load_json, require_dirpath
from rf_detr_train.utils.progress import create_progress

CocoLayout = Literal["roboflow", "native"]
ImageMode = Literal["symlink", "copy"]


def _validate_coco(data: dict[str, Any], images_dirpath: Path) -> list[Path]:
    """Validate annotation references and find every image before writing output."""
    if not isinstance(data, dict):
        raise ValueError("Expected a COCO JSON object.")

    ids: dict[str, set[int]] = {}
    for field in ("images", "categories", "annotations"):
        records = data.get(field)
        if not isinstance(records, list):
            raise ValueError(f"Expected a COCO '{field}' list in labels.json.")
        ids[field] = set()
        for record in records:
            if not isinstance(record, dict) or type(record.get("id")) is not int:
                raise ValueError(f"Each COCO '{field}' entry must have an integer ID.")
            if record["id"] in ids[field]:
                raise ValueError(f"Duplicate ID {record['id']} in COCO '{field}'.")
            ids[field].add(record["id"])

    if not data["images"] or not data["categories"]:
        raise ValueError("The split must contain images and category definitions.")
    for category in data["categories"]:
        if not isinstance(category.get("name"), str) or not category["name"]:
            raise ValueError(f"Category {category['id']} must have a name.")
    for annotation in data["annotations"]:
        if annotation.get("image_id") not in ids["images"]:
            raise ValueError(f"Annotation {annotation['id']} references an unknown image_id.")
        if annotation.get("category_id") not in ids["categories"]:
            raise ValueError(f"Annotation {annotation['id']} references an unknown category_id.")

    image_paths: list[Path] = []
    seen: set[Path] = set()
    for image in data["images"]:
        filename = image.get("file_name")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Image {image['id']} must have a file_name.")
        relative_path = Path(filename)
        if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.name:
            raise ValueError(f"Expected an image path relative to data/: {filename!r}")
        if relative_path == Path("_annotations.coco.json"):
            raise ValueError("An image filename conflicts with '_annotations.coco.json'.")
        if relative_path in seen:
            raise ValueError(f"Duplicate image filename: {filename!r}")
        if not (images_dirpath / relative_path).is_file():
            raise FileNotFoundError(f"Missing COCO image: {images_dirpath / relative_path}")
        seen.add(relative_path)
        image_paths.append(relative_path)
    return image_paths


def convert_fiftyone_coco(
    source_dirpath: str | Path,
    output_dirpath: str | Path,
    split: DatasetSplit,
    layout: CocoLayout,
    image_mode: ImageMode = "symlink",
) -> Path:
    """Convert one downloaded FiftyOne COCO 2017 split without changing its JSON.

    :param source_dirpath: COCO root containing ``train/`` or ``validation/``.
    :param output_dirpath: Root of the converted dataset.
    :param split: Labelled source split: ``train`` or ``validation``.
    :param layout: Output directory layout: ``roboflow`` or ``native``.
    :param image_mode: Create relative symlinks or independent image copies.
    :return: Path to the output annotation file.
    :raises FileExistsError: If this output split already exists; no files are overwritten.
    :raises ValueError: If the input annotations or requested options are invalid.
    """
    if split not in ("train", "validation"):
        raise ValueError("Only train/validation are supported; COCO test has no public detection annotations.")
    if layout not in ("roboflow", "native"):
        raise ValueError(f"Unsupported COCO layout: {layout!r}")
    if image_mode not in ("symlink", "copy"):
        raise ValueError(f"Unsupported image mode: {image_mode!r}")

    source_dirpath = Path(source_dirpath).expanduser().resolve()
    output_dirpath = Path(output_dirpath).expanduser().resolve()
    if output_dirpath.is_relative_to(source_dirpath) or source_dirpath.is_relative_to(output_dirpath):
        raise ValueError("Source and output directories must be separate, without nesting.")

    source_images = require_dirpath(source_dirpath / split / "data", description="FiftyOne images")
    source_annotations = source_dirpath / split / "labels.json"
    if layout == "roboflow":
        split_names = {"train": "train", "validation": "valid"}
        images_destination = output_dirpath / split_names[split]
        annotations_destination = images_destination / "_annotations.coco.json"
        other_split = "valid" if split == "train" else "train"
        other_annotations = output_dirpath / other_split / "_annotations.coco.json"
    else:
        split_names = {"train": "train2017", "validation": "val2017"}
        images_destination = output_dirpath / split_names[split]
        annotations_destination = output_dirpath / "annotations" / f"instances_{split_names[split]}.json"
        other_split = "val2017" if split == "train" else "train2017"
        other_annotations = output_dirpath / "annotations" / f"instances_{other_split}.json"

    for path in (images_destination, annotations_destination):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}. Choose another output directory.")

    data = load_json(source_annotations)
    image_paths = _validate_coco(data, source_images)
    if other_annotations.is_file():
        other_data = load_json(other_annotations)
        categories = {category["id"]: category["name"] for category in data["categories"]}
        other_categories = {category["id"]: category["name"] for category in other_data["categories"]}
        if categories != other_categories:
            raise ValueError(f"Category IDs/names differ from the existing split: {other_annotations}")

    logger.info(
        f"Converting COCO {split} to {layout}: {len(image_paths)} images, "
        f"{len(data['annotations'])} annotations, {len(data['categories'])} categories ({image_mode})"
    )
    output_dirpath.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{split}-", dir=output_dirpath) as staging_dir:
        staging_path = Path(staging_dir)
        staging_images = staging_path / "images"
        staging_images.mkdir()
        with create_progress(unit="images", show_metrics=False) as progress:
            for relative_path in progress.track(image_paths, description=f"Converting {split}"):
                source_image = source_images / relative_path
                staged_image = staging_images / relative_path
                staged_image.parent.mkdir(parents=True, exist_ok=True)
                if image_mode == "copy":
                    shutil.copy2(source_image, staged_image)
                else:
                    # Links are relative to their final directory, not the staging directory.
                    final_parent = (images_destination / relative_path).parent
                    staged_image.symlink_to(os.path.relpath(source_image, final_parent))

        if layout == "roboflow":
            shutil.copy2(source_annotations, staging_images / "_annotations.coco.json")
            staging_images.rename(images_destination)
        else:
            staged_annotations = staging_path / "labels.json"
            shutil.copy2(source_annotations, staged_annotations)
            annotations_destination.parent.mkdir(parents=True, exist_ok=True)
            staging_images.rename(images_destination)
            staged_annotations.rename(annotations_destination)

    logger.info(f"Converted annotations saved unchanged at: '{annotations_destination}'")
    return annotations_destination
