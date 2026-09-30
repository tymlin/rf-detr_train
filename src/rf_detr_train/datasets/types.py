from typing import Literal

DatasetName = Literal["coco", "open_images"]
DatasetSplit = Literal["train", "validation", "test"]
CocoLayout = Literal["roboflow", "native"]
ImageMode = Literal["symlink", "copy"]
