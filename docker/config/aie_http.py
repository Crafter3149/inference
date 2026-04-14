"""AIE inference server -- Docker configuration.

Minimal app factory for Docker deployment. Mirrors gpu_http.py but
drops Active Learning and Stream API (not needed for AIE workloads).
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
        _log.info(">> %s %s", request.method, path)
        response = await call_next(request)
        elapsed = (time.perf_counter() - t0) * 1000
        _log.info("<< %s %s [%s] %.0fms", request.method, path, response.status_code, elapsed)
        return response


model_registry = AIEModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = ModelManager(model_registry=model_registry)
model_manager = WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)
model_manager.init_pingback()
interface = HttpInterface(model_manager)
app = interface.app
app.add_middleware(RequestLogMiddleware)
