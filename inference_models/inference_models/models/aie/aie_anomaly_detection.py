import json
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from torchvision import transforms

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.aie.decrypt import decrypt_nst
from inference_models.models.base.anomaly_detection import (
    AnomalyDetectionModel,
    AnomalyDetectionResult,
)

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = 256


class AIEForAnomalyDetection(
    AnomalyDetectionModel[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
):
    """AIE EfficientAD anomaly detection model.

    Loads EfficientAD .pt or .nst weights trained with the AIE Training Toolkit.
    Input: RGB images resized to (imagesize x imagesize).
    Output: anomaly heatmap + image-level anomaly score.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "AIEForAnomalyDetection":
        model_dir = Path(model_name_or_path)
        weights_pt = model_dir / "weights.pt"
        weights_nst = model_dir / "weights.nst"

        if weights_pt.exists():
            logger.info("Loading EfficientAD model from %s", weights_pt)
            model = _load_efficientad_from_file(weights_pt, device)
        elif weights_nst.exists():
            logger.info(
                "Loading EfficientAD model from encrypted %s", weights_nst
            )
            decrypted = decrypt_nst(str(weights_nst))
            if decrypted is None:
                raise RuntimeError(
                    f"Failed to decrypt {weights_nst}: invalid .nst header"
                )
            model = _load_efficientad_from_bytes(decrypted, device)
        else:
            raise FileNotFoundError(
                f"No weights file found in {model_dir}. "
                "Expected weights.pt or weights.nst"
            )

        image_size = DEFAULT_IMAGE_SIZE
        params_json = model_dir / "params.json"
        if params_json.exists():
            with open(params_json) as f:
                params = json.load(f)
            model_params = params.get("model_params", params)
            image_size = model_params.get("imagesize", DEFAULT_IMAGE_SIZE)

        return cls(model=model, image_size=image_size, device=device)

    def __init__(
        self,
        model: torch.nn.Module,
        image_size: int,
        device: torch.device,
    ):
        self._model = model
        self._image_size = image_size
        self._device = device
        self._transform = transforms.Compose(
            [
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
            ]
        )

    @property
    def class_names(self) -> List[str]:
        return ["good", "anomaly"]

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> torch.Tensor:
        from PIL import Image

        pil_images = _to_pil_list(images)
        tensors = [self._transform(img) for img in pil_images]
        batch = torch.stack(tensors).to(self._device)
        return batch

    def forward(
        self, pre_processed_images: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            combined_maps, images_score = self._model(pre_processed_images)
        return combined_maps, images_score

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> List[AnomalyDetectionResult]:
        combined_maps, images_score = model_results
        results = []
        for i in range(combined_maps.shape[0]):
            results.append(
                AnomalyDetectionResult(
                    anomaly_map=combined_maps[i],  # (1, H, W)
                    anomaly_score=images_score[i],  # scalar
                )
            )
        return results


def _load_efficientad_from_file(
    path: Path, device: torch.device
) -> torch.nn.Module:
    """Load EfficientAD model from a .pt file (pickle or TorchScript)."""
    try:
        import AIE  # noqa: F401
    except ImportError:
        pass  # TorchScript doesn't need AIE

    try:
        model = torch.load(str(path), map_location=device, weights_only=False)
        model.eval()
        return model
    except Exception:
        pass

    try:
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load EfficientAD model from {path}: {e}"
        ) from e


def _load_efficientad_from_bytes(
    data: bytes, device: torch.device
) -> torch.nn.Module:
    """Load EfficientAD model from decrypted bytes (pickle or TorchScript)."""
    try:
        import AIE  # noqa: F401
    except ImportError:
        pass  # TorchScript doesn't need AIE

    buf = BytesIO(data)
    try:
        model = torch.load(buf, map_location=device, weights_only=False)
        model.eval()
        return model
    except Exception:
        pass

    buf.seek(0)
    try:
        model = torch.jit.load(buf, map_location=device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load decrypted EfficientAD model as .pt or TorchScript: {e}"
        ) from e


def _to_pil_list(
    images: Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]],
) -> list:
    """Convert various image formats to a list of PIL Images (RGB)."""
    from PIL import Image

    def _single_to_pil(img) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)  # CHW → HWC
            arr = img.cpu().numpy()
        elif isinstance(img, np.ndarray):
            arr = img
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)

        if arr.ndim == 3 and arr.shape[2] == 3:
            # Assume BGR input (OpenCV convention), convert to RGB
            arr = arr[:, :, ::-1].copy()

        return Image.fromarray(arr)

    if isinstance(images, (torch.Tensor, np.ndarray)):
        if images.ndim == 4:
            return [_single_to_pil(images[i]) for i in range(images.shape[0])]
        elif images.ndim == 3:
            return [_single_to_pil(images)]
        else:
            raise ValueError(f"Expected 3D or 4D input, got {images.ndim}D")

    if isinstance(images, list):
        return [_single_to_pil(img) for img in images]

    raise TypeError(f"Unsupported images type: {type(images)}")
