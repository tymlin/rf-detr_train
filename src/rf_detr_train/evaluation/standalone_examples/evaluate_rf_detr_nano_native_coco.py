from typing import Literal

from rfdetr import RFDETRNano

from rf_detr_train.consts import NATIVE_COCO_DATASETS_DIRPATH, RESULTS_DIRPATH
from rf_detr_train.utils.files import create_directory, save_json

DEVICE = "cuda"
BATCH_SIZE = 8
NUM_WORKERS = 4
SPLIT: Literal["test", "val"] = "val"
DATASET_DIRPATH = NATIVE_COCO_DATASETS_DIRPATH / "coco-2017"
OUTPUT_DIRPATH = RESULTS_DIRPATH / "coco" / "rf_detr_nano_native_coco"

create_directory(OUTPUT_DIRPATH, parents=True)

model = RFDETRNano(device=DEVICE)
metrics = model.evaluate(
    dataset_dir=str(DATASET_DIRPATH),
    dataset_file="coco",
    split=SPLIT,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    eval_max_dets=100,
    log_per_class_metrics=True,
)
save_json(filepath=OUTPUT_DIRPATH / "metrics.json", data=metrics)
