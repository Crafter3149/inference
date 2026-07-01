import json
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from inference.core.managers.entities import ModelDescription


class ServerVersionInfo(BaseModel):
    """Server version information.

    Attributes:
        name (str): Server name.
        version (str): Server version.
        uuid (str): Server UUID.
    """

    name: str = Field(examples=["Roboflow Inference Server"])
    version: str = Field(examples=["0.0.1"])
    uuid: str = Field(examples=["9c18c6f4-2266-41fb-8a0f-c12ae28f6fbe"])


class ModelDescriptionEntity(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(
        description="Identifier of the model", examples=["some-project/3"]
    )
    task_type: str = Field(
        description="Type of the task that the model performs",
        examples=["classification"],
    )
    batch_size: Optional[int] = Field(
        None,
        description="Batch size accepted by the model (if registered).",
    )
    input_height: Optional[int] = Field(
        None,
        description="Image input height accepted by the model (if registered).",
    )
    input_width: Optional[int] = Field(
        None,
        description="Image input width accepted by the model (if registered).",
    )

    @classmethod
    def from_model_description(
        cls, model_description: ModelDescription
    ) -> "ModelDescriptionEntity":
        # Convert string batch_size (indicating dynamic batching) to None
        batch_size = model_description.batch_size
        try:
            batch_size = int(batch_size)
        except (TypeError, ValueError):
            batch_size = None
        return cls(
            model_id=model_description.model_id,
            task_type=model_description.task_type,
            batch_size=batch_size,
            input_height=model_description.input_height,
            input_width=model_description.input_width,
        )


class ModelsDescriptions(BaseModel):
    models: List[ModelDescriptionEntity] = Field(
        description="List of models that are loaded by model manager.",
    )

    @classmethod
    def from_models_descriptions(
        cls, models_descriptions: List[ModelDescription]
    ) -> "ModelsDescriptions":
        return cls(
            models=[
                ModelDescriptionEntity.from_model_description(
                    model_description=model_description
                )
                for model_description in models_descriptions
            ]
        )


class LocalModelEntry(BaseModel):
    """One model directory discovered on the server's local filesystem."""

    model_config = ConfigDict(protected_namespaces=())
    name: str = Field(description="Model directory name", examples=["lightset1"])
    model_id: str = Field(
        description="Absolute directory path; pass this as model_id when loading.",
        examples=["E:/models/lightset1"],
    )
    task_type: Optional[str] = Field(
        None, description="task_type read from the directory's model_config.json"
    )
    model_architecture: Optional[str] = Field(
        None,
        description="model_architecture read from the directory's model_config.json",
    )


class LocalModelsResponse(BaseModel):
    """Model directories available under the server-configured LOCAL_MODELS_DIR."""

    configured: bool = Field(
        description="Whether LOCAL_MODELS_DIR is set on the server."
    )
    root: Optional[str] = Field(
        None, description="The configured models root directory (if any)."
    )
    models: List[LocalModelEntry] = Field(
        default_factory=list,
        description="Model directories (each containing a model_config.json) under root.",
    )

    @classmethod
    def scan(cls, root: Optional[str]) -> "LocalModelsResponse":
        """Enumerate immediate sub-directories of ``root`` that contain a
        ``model_config.json`` marker file. ``root`` is server-configured (never
        client-supplied), so there is no path-traversal surface; only the model
        directory names + their absolute paths + parsed metadata are exposed.
        """
        if not root:
            return cls(configured=False, root=None, models=[])
        base = Path(root)
        entries: List[LocalModelEntry] = []
        if base.is_dir():
            for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
                if not child.is_dir():
                    continue
                cfg = child / "model_config.json"
                if not cfg.is_file():
                    continue  # whitelist: only dirs carrying the model marker file
                task_type = None
                architecture = None
                try:
                    data = json.loads(cfg.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        task_type = data.get("task_type")
                        architecture = data.get("model_architecture")
                except (OSError, ValueError):
                    pass  # malformed config -> still list the dir, metadata stays None
                entries.append(
                    LocalModelEntry(
                        name=child.name,
                        model_id=str(child),
                        task_type=task_type,
                        model_architecture=architecture,
                    )
                )
        return cls(configured=True, root=str(base), models=entries)
