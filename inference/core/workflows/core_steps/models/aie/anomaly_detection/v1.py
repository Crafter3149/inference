from typing import List, Literal, Optional, Type, Union

import numpy as np
from pydantic import ConfigDict, Field

from inference.core.entities.requests.inference import CVInferenceRequest
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class AIEAnomalyDetectionBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "AIE Anomaly Detection",
            "version": "v1",
            "short_description": "Run anomaly detection on an AIE EfficientAD model.",
            "long_description": (
                "Run anomaly detection using an AIE-trained EfficientAD model. "
                "Accepts a local model directory path and outputs an anomaly score "
                "and anomaly heatmap."
            ),
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["anomaly", "defect", "aie", "efficientad"],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-chart-network",
                "blockPriority": 0,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/aie_anomaly_detection_model@v1"]
    model_id: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        description="Local path to AIE anomaly detection model directory",
        examples=["D:/models/ad_model", "$inputs.model_id"],
    )
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="anomaly_map", kind=[IMAGE_KIND]),
            OutputDefinition(name="anomaly_score", kind=[FLOAT_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class AIEAnomalyDetectionBlockV1(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return AIEAnomalyDetectionBlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(images=images, model_id=model_id)
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            raise NotImplementedError(
                "AIE Anomaly Detection does not support remote execution."
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
    ) -> BlockResult:
        inference_images = [
            i.to_inference_format(numpy_preferred=True) for i in images
        ]
        request = CVInferenceRequest(
            api_key=self._api_key,
            model_id=model_id,
            image=inference_images,
            source="workflow-execution",
        )
        self._model_manager.add_model(
            model_id=model_id,
            api_key=self._api_key,
        )
        predictions = self._model_manager.infer_from_request_sync(
            model_id=model_id, request=request
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        results = []
        for image_data, prediction in zip(images, predictions):
            anomaly_map_raw = np.array(prediction.anomaly_map, dtype=np.float32)
            if anomaly_map_raw.ndim == 3:
                anomaly_map_raw = anomaly_map_raw[0]
            clipped = anomaly_map_raw.clip(0, 1)
            grayscale = (clipped * 255).astype(np.uint8)
            # IMAGE_KIND expects BGR 3-channel or grayscale numpy array
            # Convert grayscale to 3-channel BGR for compatibility
            anomaly_map_bgr = np.stack([grayscale, grayscale, grayscale], axis=-1)
            anomaly_map_image = WorkflowImageData(
                parent_metadata=image_data.parent_metadata,
                workflow_root_ancestor_metadata=image_data.workflow_root_ancestor_metadata,
                numpy_image=anomaly_map_bgr,
            )
            results.append(
                {
                    "anomaly_map": anomaly_map_image,
                    "anomaly_score": float(prediction.anomaly_score),
                }
            )
        return results
