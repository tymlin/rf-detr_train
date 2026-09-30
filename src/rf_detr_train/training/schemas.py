"""Configuration for fine-tuning RF-DETR on a Roboflow COCO dataset."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rf_detr_train.consts import NOW, ROBOFLOW_DATASETS_DIRPATH, RUNS_DIRPATH
from rf_detr_train.models.types import RFDETRModelName


class TrainingConfig(BaseModel):
    """Configure a model and a single fine-tuning run."""

    model_config = ConfigDict(extra="forbid")

    model_name: RFDETRModelName = Field(default="small", description="RF-DETR detection model variant to fine-tune.")
    run_name: str = Field(
        default_factory=lambda data: f"rf_detr_{data['model_name']}",
        min_length=1,
        description="Experiment name used in the automatic results directory; defaults to rf_detr_<model_name>.",
    )
    dataset_dirpath: Path = Field(
        default=ROBOFLOW_DATASETS_DIRPATH / "open-images-v7-person",
        description="Roboflow COCO dataset root containing train/ and valid/ with images and _annotations.coco.json.",
    )
    output_dirpath: Path = Field(
        default_factory=lambda data: RUNS_DIRPATH / data["run_name"] / NOW,
        description="Run directory; defaults to RUNS_DIRPATH / run_name / NOW. An explicit path overrides it.",
    )
    checkpoint_filepath: Path | None = Field(
        default=None,
        description="Initial weights matching the selected variant; None uses its official pretrained COCO weights.",
    )
    device: str = Field(default="cuda", description="Device used for training, for example cuda or cpu.")
    epochs: int = Field(default=20, gt=0, description="Number of training epochs.")
    batch_size: int = Field(default=4, gt=0, description="Number of images per training batch on each device.")
    grad_accum_steps: int = Field(
        default=4,
        gt=0,
        description="Batches per optimizer step; effective batch per device is batch_size times this value.",
    )
    lr: float = Field(default=1e-4, gt=0, description="Learning rate passed to RF-DETR training.")
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of data loader worker processes; zero loads data in the main process.",
    )
