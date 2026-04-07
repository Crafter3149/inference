import logging
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

from inference_models.models.aie.decrypt import decrypt_nst

logger = logging.getLogger(__name__)


class _AIEUltralyticsBase:
    """Shared .nst/.pt loading and ultralytics inference logic for AIE YOLO models."""

    @staticmethod
    def _load_ultralytics_model(model_dir: Path, device: str) -> "YOLO":
        """Load an ultralytics YOLO model from a model directory.

        Handles .pt and .nst (encrypted) weight files. For .nst files,
        decrypts and writes to a temp file before loading.

        Args:
            model_dir: Path to the model directory containing weights.pt or weights.nst.
            device: Device string for the model (e.g. "cpu", "cuda:0").

        Returns:
            An ultralytics YOLO model instance.

        Raises:
            ImportError: If the AIE or ultralytics package is not installed.
            FileNotFoundError: If no weights file is found.
            RuntimeError: If the model cannot be loaded.
        """
        try:
            import AIE  # noqa: F401
        except ImportError:
            raise ImportError(
                "AIE package is required to load AIE YOLO models. "
                "Install it with: pip install -e <path-to-AIE>"
            )
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package is required to load AIE YOLO models. "
                "Install it with: pip install ultralytics"
            )

        weights_pt = model_dir / "weights.pt"
        weights_nst = model_dir / "weights.nst"

        if weights_pt.exists():
            logger.info("Loading AIE YOLO model from %s", weights_pt)
            model = YOLO(str(weights_pt))
        elif weights_nst.exists():
            logger.info("Loading AIE YOLO model from encrypted %s", weights_nst)
            decrypted = decrypt_nst(str(weights_nst))
            if decrypted is None:
                raise RuntimeError(
                    f"Failed to decrypt {weights_nst}: invalid .nst header"
                )
            # Try loading as pickle first, then TorchScript
            try:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".pt", delete=False, dir=str(model_dir)
                )
                tmp.write(decrypted)
                tmp.close()
                model = YOLO(tmp.name)
            except Exception:
                # Fallback: try as TorchScript
                try:
                    torch.jit.load(BytesIO(decrypted))
                    tmp2 = tempfile.NamedTemporaryFile(
                        suffix=".torchscript", delete=False, dir=str(model_dir)
                    )
                    tmp2.write(decrypted)
                    tmp2.close()
                    model = YOLO(tmp2.name)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load decrypted .nst as YOLO .pt or TorchScript: {e}"
                    ) from e
        else:
            raise FileNotFoundError(
                f"No weights file found in {model_dir}. "
                "Expected weights.pt or weights.nst"
            )

        model.to(device)
        return model

    @staticmethod
    def _images_to_numpy_list(
        images: Union[
            torch.Tensor,
            np.ndarray,
            List[torch.Tensor],
            List[np.ndarray],
        ],
    ) -> List[np.ndarray]:
        """Convert various image input formats to List[np.ndarray] in BGR HWC uint8.

        Ultralytics expects BGR HWC numpy arrays or a list of them.

        Args:
            images: Input images in various formats.

        Returns:
            List of numpy arrays in BGR HWC uint8 format.
        """
        if isinstance(images, torch.Tensor):
            if images.ndim == 4:
                # (B, C, H, W) or (B, H, W, C)
                imgs = [images[i] for i in range(images.shape[0])]
            elif images.ndim == 3:
                imgs = [images]
            else:
                raise ValueError(
                    f"Expected 3D or 4D tensor, got {images.ndim}D"
                )
            return [_AIEUltralyticsBase._tensor_to_bgr_numpy(t) for t in imgs]

        if isinstance(images, np.ndarray):
            if images.ndim == 4:
                return [images[i] for i in range(images.shape[0])]
            elif images.ndim == 3:
                return [images]
            else:
                raise ValueError(
                    f"Expected 3D or 4D ndarray, got {images.ndim}D"
                )

        if isinstance(images, list):
            result = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    result.append(
                        _AIEUltralyticsBase._tensor_to_bgr_numpy(img)
                    )
                elif isinstance(img, np.ndarray):
                    result.append(img)
                else:
                    raise TypeError(
                        f"Unsupported image type in list: {type(img)}"
                    )
            return result

        raise TypeError(f"Unsupported images type: {type(images)}")

    @staticmethod
    def _tensor_to_bgr_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert a single image tensor to BGR HWC uint8 numpy array."""
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            # CHW → HWC
            arr = tensor.permute(1, 2, 0).cpu().numpy()
        elif tensor.ndim == 3:
            # Already HWC
            arr = tensor.cpu().numpy()
        else:
            raise ValueError(f"Expected 3D tensor, got shape {tensor.shape}")

        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)

        # If RGB (3 channels), convert to BGR for ultralytics
        if arr.shape[2] == 3:
            arr = arr[:, :, ::-1].copy()

        return arr
