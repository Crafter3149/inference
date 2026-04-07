# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Roboflow Inference is a set of Python packages for running computer vision models locally via HTTP API or CLI. Target Python version is 3.10 (minimum 3.8).

## Common Commands

### Setup
```bash
conda create -n inference-development python=3.10
conda activate inference-development
pip install -e .
pip install -e ".[sam]"           # optional: SAM model support
pip install -e ".[cloud-storage]" # optional: cloud storage support
```

### Code Quality
```bash
make style                  # format code (black + isort)
make check_code_quality     # lint check (black --check + isort --check + flake8)
```

### Testing
```bash
# Unit tests (by package)
pytest tests/inference/unit_tests/
pytest tests/inference_cli/unit_tests/
pytest tests/inference_sdk/unit_tests/
pytest tests/workflows/unit_tests/

# Single test file
pytest tests/inference/unit_tests/path/to/test_file.py

# Skip slow tests
pytest -m "not slow" tests/

# Integration tests (some require ROBOFLOW_API_KEY)
pytest tests/inference/models_predictions_tests/
pytest tests/workflows/integration_tests/
```

### Docker (run from repo root)
```bash
# Build CPU dev image (includes watchdog for auto-restart on code changes)
docker build -t roboflow/roboflow-inference-server-cpu:dev -f docker/dockerfiles/Dockerfile.onnx.cpu.dev .

# Run with volume mount for live code editing
docker run -p 9001:9001 -v ./inference:/app/inference roboflow/roboflow-inference-server-cpu:dev

# Build GPU image
docker build -t roboflow/roboflow-inference-server-gpu:dev -f docker/dockerfiles/Dockerfile.onnx.gpu .
```

## Architecture

### Package Structure
- **`inference/`** — Core library: model loading, HTTP API, workflows engine, streaming
- **`inference_cli/`** — CLI tools (`inference` command, entry point in `inference_cli/main.py`)
- **`inference_sdk/`** — Python SDK for interacting with a running inference server
- **`inference_models/`** — Standalone model package (separate `pyproject.toml`, uses `uv`)
- **`docker/`** — Dockerfiles for CPU/GPU/Jetson/Lambda/TensorRT builds
- **`tests/`** — Mirrors package structure: `unit_tests/`, `integration_tests/`, `models_predictions_tests/`

### Server Startup Flow (`docker/config/cpu_http.py`)
1. `RoboflowModelRegistry` is created with `ROBOFLOW_MODEL_TYPES` (model type → loader class mapping)
2. `ModelManager` is wrapped with `ActiveLearningManager` (unless Lambda/GCP) and `WithFixedSizeCache` (respects `MAX_ACTIVE_MODELS`)
3. `HttpInterface(model_manager)` creates the FastAPI app
4. Stream manager starts as a separate process if `ENABLE_STREAM_API=True`

### Key Subsystems
- **Model Registry** (`inference/core/registries/`): Maps model types to loader classes via `ROBOFLOW_MODEL_TYPES`
- **Model Manager** (`inference/core/managers/`): Handles model lifecycle with LRU cache eviction (`WithFixedSizeCache`)
- **HTTP Interface** (`inference/core/interfaces/http/http_api.py`): FastAPI app with routes for `/infer`, `/workflows`, model management
- **Workflows Engine** (`inference/core/workflows/execution_engine/`): Composable workflow blocks with dynamic batching and concurrent execution
- **Stream Management** (`inference/enterprise/stream_management/`): Enterprise feature for RTSP/camera/video processing (multi-process)
- **Environment Config** (`inference/core/env.py`): Central hub for all environment variable configuration

### Supported Models (40+ architectures)
YOLO family (v5/7/8/9/10/11/12/26, NAS, World), SAM/SAM2/SAM3, CLIP, Florence2, Grounding DINO, OWLv2, PaliGemma, Qwen2.5/3.5-VL, DocTR, EasyOCR, and more.

## Code Style

- **Formatter**: Black (88 char line width), excludes `__init__.py` and `node_modules`
- **Import sorting**: isort with `profile = black`
- **Linter**: flake8 — ignores D (docstrings), E203, E501, W503
- **Type hints**: Pydantic v2 for request/response models
- **Test markers**: `@pytest.mark.slow` for slow tests, `@pytest.mark.workflows` for workflow tests

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `ROBOFLOW_API_KEY` | `""` | Enables authenticated requests |
| `MODEL_CACHE_DIR` | `/tmp/cache` | Downloaded model storage |
| `PORT` | `9001` | HTTP API port |
| `MAX_ACTIVE_MODELS` | `1` | LRU cache size for loaded models |
| `NUM_WORKERS` | `1` | Server worker threads |
| `ENABLE_STREAM_API` | `True` | Enable stream management API |

## Published PyPI Packages

Built via `make create_wheels` using scripts in `.release/pypi/`:
`inference-core`, `inference-cpu`, `inference-gpu`, `inference` (meta), `inference-sdk`, `inference-cli`
