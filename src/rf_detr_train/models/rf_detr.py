"""Build an RF-DETR detection model with pretrained weights."""

import os
from pathlib import Path

from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.detr import RFDETR

from rf_detr_train.consts import MODELS_DIRPATH
from rf_detr_train.models.types import RFDETRModelName

MODEL_CLASSES: dict[RFDETRModelName, type[RFDETR]] = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}


def build_model(
    model_name: RFDETRModelName,
    device: str = "cuda",
    checkpoint_filepath: Path | str | None = None,
) -> RFDETR:
    """Build the selected variant using its official weights or a compatible checkpoint.

    :param model_name: Model variant: nano, small, medium or large.
    :param device: Device passed to RF-DETR, for example cuda or cpu.
    :param checkpoint_filepath: Optional checkpoint matching the selected variant.
        None uses the variant's default pretrained weights.
    :return: Initialized RF-DETR model.
    """
    # RF-DETR uses RF_HOME as its pretrained weights cache; store weights in the project's models directory.
    os.environ["RF_HOME"] = str(MODELS_DIRPATH)
    model_class = MODEL_CLASSES[model_name]
    if checkpoint_filepath is None:
        return model_class(device=device)

    return model_class(device=device, pretrain_weights=str(checkpoint_filepath))
