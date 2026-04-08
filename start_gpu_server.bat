@echo off
REM ============================================================
REM  GPU Inference Server Launcher
REM  Usage: start_gpu_server.bat [port]
REM  Uses .venv in this repo (built with uv, has CUDA torch)
REM ============================================================

setlocal

REM --- Activate .venv ---
set VENV_DIR=%~dp0.venv
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo ERROR: .venv not found at %VENV_DIR%
    echo Create it with:
    echo   cd %~dp0
    echo   uv venv --python 3.11 .venv
    echo   uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu124
    echo   uv pip install --python .venv\Scripts\python.exe -e .
    echo   uv pip install --python .venv\Scripts\python.exe -e inference_models
    exit /b 1
)
call "%VENV_DIR%\Scripts\activate.bat"

REM --- Config ---
set PORT=%1
if "%PORT%"=="" set PORT=9001
set APP_DIR=%~dp0docker\config

REM --- Environment Variables ---
set ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True
set ENABLE_BUILDER=true
set ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=True
set MAX_ACTIVE_MODELS=3
set WORKFLOWS_STEP_EXECUTION_MODE=local
set ENABLE_STREAM_API=False
set ACTIVE_LEARNING_ENABLED=False
set ENABLE_DASHBOARD=true

REM --- Preflight: check torch+CUDA ---
python -c "import torch; assert torch.cuda.is_available(), f'torch {torch.__version__} has NO CUDA'; print(f'torch {torch.__version__}  CUDA {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}')"
if errorlevel 1 (
    echo.
    echo [WARNING] torch CUDA not available. Server will run on CPU.
    echo To install CUDA torch in .venv:
    echo   uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu124
    echo.
    pause
)

REM --- Launch ---
echo.
echo Starting inference server on port %PORT% ...
echo   VENV:   %VENV_DIR%
echo   APP_DIR: %APP_DIR%
echo   MAX_ACTIVE_MODELS: %MAX_ACTIVE_MODELS%
echo   ENABLE_BUILDER: %ENABLE_BUILDER%
echo   WORKFLOWS_STEP_EXECUTION_MODE: %WORKFLOWS_STEP_EXECUTION_MODE%
echo.

uvicorn gpu_http:app --host 0.0.0.0 --port %PORT% --app-dir "%APP_DIR%"

endlocal
