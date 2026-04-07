from pathlib import Path
from threading import Lock
from typing import List, Union

import numpy as np
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.base.classification import (
    ClassificationModel,
    ClassificationPrediction,
)

from inference_models.models.aie._aie_ultralytics_base import _AIEUltralyticsBase


class AIEForClassification(
    ClassificationModel[List[np.ndarray], list],
):
    """AIE YOLO classification model.

    Loads YOLO classify .pt or .nst weights trained with the AIE Training Toolkit.
    Uses ultralytics YOLO for inference.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "AIEForClassification":
        model_dir = Path(model_name_or_path)
        yolo_model = _AIEUltralyticsBase._load_ultralytics_model(
            model_dir, str(device)
        )
        names = yolo_model.names
        class_names = [names[i] for i in sorted(names.keys())]
        return cls(model=yolo_model, class_names_list=class_names, device=device)

    def __init__(
        self,
        model,
        class_names_list: List[str],
        device: torch.device,
    ):
        self._model = model
        self._class_names = class_names_list
        self._device = device
        self._lock = Lock()

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[np.ndarray]:
        return _AIEUltralyticsBase._images_to_numpy_list(images)

    def forward(
        self, pre_processed_images: List[np.ndarray], **kwargs
    ) -> list:
        with self._lock:
            results = self._model.predict(
                pre_processed_images,
                verbose=False,
            )
        return results

    def post_process(
        self, model_results: list, **kwargs
    ) -> ClassificationPrediction:
        class_ids = []
        confidences = []
        for result in model_results:
            probs = result.probs
            top1_id = probs.top1
            top1_conf = probs.data[top1_id].item()
            class_ids.append(top1_id)
            confidences.append(top1_conf)
        return ClassificationPrediction(
            class_id=torch.tensor(class_ids, dtype=torch.int, device=self._device),
            confidence=torch.tensor(confidences, device=self._device),
        )
