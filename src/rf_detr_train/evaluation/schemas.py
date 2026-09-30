"""Configuration and results for native COCO evaluation."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rf_detr_train.consts import NATIVE_COCO_DATASETS_DIRPATH, NOW, RUNS_DIRPATH
from rf_detr_train.models.types import RFDETRModelName


class EvaluationConfig(BaseModel):
    """Configure a model and a single native COCO evaluation run."""

    model_config = ConfigDict(extra="forbid")

    model_name: RFDETRModelName = Field(default="small", description="RF-DETR detection model variant to evaluate.")
    run_name: str = Field(
        default_factory=lambda data: f"rf_detr_{data['model_name']}",
        min_length=1,
        description="Experiment name used in the automatic results directory; defaults to rf_detr_<model_name>.",
    )
    dataset_dirpath: Path = Field(
        default=NATIVE_COCO_DATASETS_DIRPATH / "coco-2017",
        description="Native COCO dataset root containing val2017/ and annotations/.",
    )
    output_dirpath: Path = Field(
        default_factory=lambda data: RUNS_DIRPATH / data["run_name"] / NOW,
        description="Results directory; defaults to RUNS_DIRPATH / run_name / NOW. An explicit path overrides it.",
    )
    checkpoint_filepath: Path | None = Field(
        default=None,
        description="Checkpoint matching the selected variant; None uses its official pretrained weights.",
    )
    device: str = Field(default="cuda", description="Device used to run RF-DETR, for example cuda or cpu.")
    split: Literal["val", "test"] = Field(
        default="val",
        description="Evaluation mode passed to RF-DETR; native COCO uses its validation set for both val and test.",
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
