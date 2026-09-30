from typing import Literal

DatasetName = Literal["coco", "open_images"]
DatasetSplit = Literal["train", "validation", "test"]
RFDETRDatasetFile = Literal["coco", "roboflow"]
CocoLayout = Literal["roboflow", "native"]
ImageMode = Literal["symlink", "copy"]
