from pathlib import Path
from threading import Lock
from typing import List, Tuple, Union

import numpy as np
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.base.instance_segmentation import (
    InstanceDetections,
    InstanceSegmentationModel,
)

from inference_models.models.aie._aie_ultralytics_base import _AIEUltralyticsBase


class AIEForInstanceSegmentation(
    InstanceSegmentationModel[List[np.ndarray], List[Tuple[int, int]], list],
):
    """AIE YOLO instance segmentation model.

    Loads YOLO segment .pt or .nst weights trained with the AIE Training Toolkit.
    Uses ultralytics YOLO for inference.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "AIEForInstanceSegmentation":
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
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        np_images = _AIEUltralyticsBase._images_to_numpy_list(images)
        original_sizes = [(img.shape[0], img.shape[1]) for img in np_images]
        return np_images, original_sizes

    def forward(
        self, pre_processed_images: List[np.ndarray], **kwargs
    ) -> list:
        confidence = kwargs.get("confidence", 0.25)
        iou_threshold = kwargs.get("iou_threshold", 0.45)
        with self._lock:
            results = self._model.predict(
                pre_processed_images,
                conf=confidence,
                iou=iou_threshold,
                verbose=False,
            )
        return results

    def post_process(
        self,
        model_results: list,
        pre_processing_meta: List[Tuple[int, int]],
        **kwargs,
    ) -> List[InstanceDetections]:
        detections_list = []
        for result in model_results:
            boxes = result.boxes
            masks = result.masks

            if boxes is None or len(boxes) == 0:
                detections_list.append(
                    InstanceDetections(
                        xyxy=torch.zeros((0, 4), device=self._device),
                        class_id=torch.zeros((0,), dtype=torch.int, device=self._device),
                        confidence=torch.zeros((0,), device=self._device),
                        mask=torch.zeros((0, 0, 0), device=self._device),
                    )
                )
                continue

            mask_tensor = (
                masks.data.to(self._device)
                if masks is not None
                else torch.zeros(
                    (len(boxes), 0, 0), device=self._device
                )
            )

            detections_list.append(
                InstanceDetections(
                    xyxy=boxes.xyxy.to(self._device),
                    class_id=boxes.cls.int().to(self._device),
                    confidence=boxes.conf.to(self._device),
                    mask=mask_tensor,
                )
            )
        return detections_list
