from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Union

import numpy as np
import torch

from inference_models.models.base.types import PreprocessedInputs, RawPrediction


@dataclass
class AnomalyDetectionResult:
    anomaly_map: torch.Tensor  # (1, H, W) — pixel-level anomaly heatmap
    anomaly_score: torch.Tensor  # scalar — image-level anomaly score


class AnomalyDetectionModel(ABC, Generic[PreprocessedInputs, RawPrediction]):

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "AnomalyDetectionModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        pass

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[AnomalyDetectionResult]:
        pre_processed_images = self.pre_process(images, **kwargs)
        model_results = self.forward(pre_processed_images, **kwargs)
        return self.post_process(model_results, **kwargs)

    @abstractmethod
    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> PreprocessedInputs:
        pass

    @abstractmethod
    def forward(
        self, pre_processed_images: PreprocessedInputs, **kwargs
    ) -> RawPrediction:
        pass

    @abstractmethod
    def post_process(
        self, model_results: RawPrediction, **kwargs
    ) -> List[AnomalyDetectionResult]:
        pass

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[AnomalyDetectionResult]:
        return self.infer(images, **kwargs)
