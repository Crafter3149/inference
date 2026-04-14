"""
AIE Inference Server — native startup without Docker.

Embeds the server wiring logic from docker/config/gpu_http.py so that
``pip install inference-core`` (or the dev setup) provides a proper CLI
entry point to launch the inference server directly via uvicorn.

Usage::

    inference aie start                     # defaults: port 9001, host 0.0.0.0
    inference aie start --port 8080
    inference aie start --max-active-models 3
"""

from __future__ import annotations

import os

import typer
from typing_extensions import Annotated

aie_app = typer.Typer(help="AIE Inference Server — run locally without Docker.")

# Env var defaults for local AIE usage.
# These are set *before* any inference module is imported so that
# inference.core.env picks up the correct values at module load time.
_ENV_DEFAULTS: dict[str, str] = {
    "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES": "True",
    "ENABLE_STREAM_API": "False",
    "ACTIVE_LEARNING_ENABLED": "False",
    "WORKFLOWS_STEP_EXECUTION_MODE": "local",
    "ENABLE_BUILDER": "true",
    "ENABLE_DASHBOARD": "true",
    "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": "True",
}


def _apply_env_defaults() -> None:
    """Set required env vars if not already present."""
    for key, value in _ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


def _check_cuda() -> None:
    """Print CUDA availability info."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            typer.echo(f"CUDA: {name} (torch {torch.__version__})")
        else:
            typer.echo(
                f"[WARNING] CUDA not available (torch {torch.__version__}). "
                "Running on CPU."
            )
    except ImportError:
        typer.echo("[WARNING] PyTorch not installed. GPU models will fail.")


def _create_app():  # noqa: ANN201 — returns FastAPI but we avoid top-level import
    """Create the FastAPI application.

    Same wiring as ``docker/config/gpu_http.py`` but importable from the
    installed package (no ``--app-dir`` hack required).
    """
    import logging
    import time

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    from inference.core.env import MAX_ACTIVE_MODELS
    from inference.core.interfaces.http.http_api import HttpInterface
    from inference.core.managers.base import ModelManager
    from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
    from inference.core.registries.aie import AIEModelRegistry
    from inference.models.utils import ROBOFLOW_MODEL_TYPES

    _log = logging.getLogger("aie.requests")
    _LOG_PREFIXES = ("/infer/", "/model/", "/workflows/")

    class RequestLogMiddleware(BaseHTTPMiddleware):
        """Log inference / model / workflow requests at INFO level."""

        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            if not path.startswith(_LOG_PREFIXES):
                return await call_next(request)
            t0 = time.perf_counter()
            _log.info(f">> {request.method} {path}")
            response = await call_next(request)
            elapsed = (time.perf_counter() - t0) * 1000
            _log.info(f"<< {request.method} {path} [{response.status_code}] {elapsed:.0f}ms")
            return response

    model_registry = AIEModelRegistry(ROBOFLOW_MODEL_TYPES)
    model_manager = ModelManager(model_registry=model_registry)
    model_manager = WithFixedSizeCache(
        model_manager, max_size=MAX_ACTIVE_MODELS
    )
    model_manager.init_pingback()
    interface = HttpInterface(model_manager)
    app = interface.app
    app.add_middleware(RequestLogMiddleware)
    return app


@aie_app.command()
def start(
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Server port."),
    ] = 9001,
    host: Annotated[
        str,
        typer.Option("--host", help="Bind address."),
    ] = "0.0.0.0",
    max_active_models: Annotated[
        int,
        typer.Option(
            "--max-active-models",
            help="Max models kept in GPU memory (LRU eviction).",
        ),
    ] = 3,
    max_batch_size: Annotated[
        int,
        typer.Option(
            "--max-batch-size",
            help="Max images per inference batch (1 = safest for GPU).",
        ),
    ] = 1,
) -> None:
    """Start the AIE inference server (native, no Docker required)."""
    # 1. Env vars MUST be set before importing inference modules.
    os.environ["MAX_ACTIVE_MODELS"] = str(max_active_models)
    os.environ["MAX_BATCH_SIZE"] = str(max_batch_size)
    _apply_env_defaults()

    # 2. CUDA check (informational).
    _check_cuda()

    # 3. Create FastAPI app (imports inference modules here).
    fastapi_app = _create_app()

    typer.echo(f"\nStarting AIE inference server on {host}:{port}")
    typer.echo(f"  MAX_ACTIVE_MODELS={max_active_models}")
    typer.echo(f"  MAX_BATCH_SIZE={max_batch_size}")
    typer.echo("")

    # 4. Run uvicorn (blocking).
    import uvicorn

    uvicorn.run(fastapi_app, host=host, port=port, access_log=False)


if __name__ == "__main__":
    aie_app()
