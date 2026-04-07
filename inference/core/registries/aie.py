import json
import os
from typing import Optional

from inference.core.exceptions import ModelNotRecognisedError
from inference.core.models.base import Model
from inference.core.registries.roboflow import RoboflowModelRegistry


class AIEModelRegistry(RoboflowModelRegistry):
    """Extends RoboflowModelRegistry with local model directory detection.

    When model_id is a local directory containing model_config.json,
    reads (task_type, model_architecture) from the config and looks up
    the adapter class in registry_dict. Otherwise falls through to
    the standard Roboflow resolution path.
    """

    def get_model(
        self,
        model_id: str,
        api_key: str,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> Model:
        if os.path.isdir(model_id):
            config_path = os.path.join(model_id, "model_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                model_type = (config["task_type"], config["model_architecture"])
                if model_type not in self.registry_dict:
                    raise ModelNotRecognisedError(
                        f"Local model type {model_type} not found in registry. "
                        f"Check model_config.json in {model_id}"
                    )
                return self.registry_dict[model_type]
        return super().get_model(
            model_id,
            api_key,
            countinference=countinference,
            service_secret=service_secret,
        )
