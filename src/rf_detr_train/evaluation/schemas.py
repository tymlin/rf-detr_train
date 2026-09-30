"""Configuration and results for native or Roboflow COCO evaluation."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rf_detr_train.consts import NATIVE_COCO_DATASETS_DIRPATH, NOW, RUNS_DIRPATH
from rf_detr_train.datasets.types import RFDETRDatasetFile
from rf_detr_train.models.types import RFDETRModelName


class EvaluationConfig(BaseModel):
    """Configure a model and a single COCO evaluation run with the selected loader."""

    model_config = ConfigDict(extra="forbid")

    model_name: RFDETRModelName = Field(default="small", description="RF-DETR detection model variant to evaluate.")
    run_name: str = Field(
        default_factory=lambda data: f"rf_detr_{data['model_name']}",
        min_length=1,
        description="Experiment name used in the automatic results directory; defaults to rf_detr_<model_name>.",
    )
    dataset_file: RFDETRDatasetFile = Field(
        default="coco",
        description="RF-DETR dataset loader: coco for native COCO layout, roboflow for Roboflow COCO layout.",
    )
    dataset_dirpath: Path = Field(
        default=NATIVE_COCO_DATASETS_DIRPATH / "coco-2017",
        description="Dataset root matching dataset_file; changing the loader does not change the default path.",
    )
    output_dirpath: Path = Field(
        default_factory=lambda data: RUNS_DIRPATH / data["run_name"] / NOW,
        description="Results directory; defaults to RUNS_DIRPATH / run_name / NOW. An explicit path overrides it.",
    )
    checkpoint_filepath: Path | None = Field(
        default=None,
        description="Checkpoint matching the variant and loader's class mapping; None uses official COCO weights.",
    )
    device: str = Field(default="cuda", description="Device used to run RF-DETR, for example cuda or cpu.")
    split: Literal["val", "test"] = Field(
        default="val",
        description="Roboflow: val reads valid/ and test reads test/. Native COCO: both read val2017/.",
    )
    batch_size: int = Field(default=8, gt=0, description="Number of images processed in each evaluation batch.")
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of data loader worker processes; zero loads data in the main process.",
    )
    eval_max_dets: int = Field(
        default=100,
        gt=0,
        description="Maximum detections per image and category used by COCO metrics; 100 is the standard COCO limit.",
    )
    log_per_class_metrics: bool = Field(
        default=True,
        description="Enable RF-DETR per-class metric reporting, including class AP in the returned metrics.",
    )


class EvaluationResult(BaseModel):
    """Store the run configuration and metrics returned by RF-DETR."""

    config: EvaluationConfig = Field(description="Configuration used for this evaluation run.")
    metrics: dict[str, float] = Field(description="Metric names and values returned by RF-DETR model.evaluate().")
