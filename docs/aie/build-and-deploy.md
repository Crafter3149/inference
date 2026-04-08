# Build & Deploy 手冊

本手冊說明如何安裝、構建、部署 Inference Server（含 AIE 整合）。

---

## 1. 套件架構

AIE 整合的程式碼分佈在兩個現有套件中，不需要建立新套件：

```
inference_models (whl)
├── models/aie/                          # AIE 模型類別
│   ├── _aie_ultralytics_base.py         #   YOLO 共用載入邏輯
│   ├── aie_object_detection.py          #   AIEForObjectDetection
│   ├── aie_classification.py            #   AIEForClassification
│   ├── aie_instance_segmentation.py     #   AIEForInstanceSegmentation
│   ├── aie_anomaly_detection.py         #   AIEForAnomalyDetection
│   └── decrypt.py                       #   .nst 解密
├── models/base/anomaly_detection.py     #   AnomalyDetectionModel base class
└── models/auto_loaders/                 #   REGISTERED_MODELS 路由

inference-core / inference-cpu / inference-gpu (whl)
├── core/registries/aie.py               #   AIEModelRegistry
├── core/models/inference_models_adapters.py  #   AD adapter
├── core/entities/responses/inference.py #   AD response
├── core/workflows/core_steps/
│   ├── models/aie/anomaly_detection/v1.py  #   AD built-in block
│   └── loader.py                        #   block 註冊
├── models/utils.py                      #   ROBOFLOW_MODEL_TYPES 路由
└── docker/config/cpu_http.py            #   啟動腳本

aie (外部套件，使用者自行安裝)
└── AIE Training Toolkit                 #   .nst 解密所需的命名空間
```

---

## 2. 開發環境安裝（editable install）

開發時直接從原始碼安裝，修改即生效，不需重新打包：

```bash
# 1. 安裝 AIE Training Toolkit（提供 .nst 解密和 YOLO pickle 命名空間）
pip install -e D:\Project\TrainingTool\AIE

# 2. 安裝 inference server
pip install -e D:\Project\inferserver\inference

# 3. 安裝 inference_models
pip install -e D:\Project\inferserver\inference\inference_models

# 4.（可選）安裝業務邏輯 plugin
pip install -e /path/to/business_plugin
```

---

## 3. 啟動模式

Server 有兩種啟動模式。絕大多數場景使用**一般模式**即可。

### 3.1 一般模式（Normal Mode）

單一 uvicorn 進程，所有推論在同一進程內完成。

```
┌────────────────────────────────────┐
│           uvicorn process          │
│                                    │
│  AIEModelRegistry                  │
│       ↓                            │
│  ModelManager + WithFixedSizeCache │
│       ↓                            │
│  HttpInterface (FastAPI)           │
│       ↓                            │
│  model.infer_from_request()        │
└────────────────────────────────────┘
```

**啟動命令：**

```bash
# Windows
set ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True
set WORKFLOWS_PLUGINS=business_plugin
uvicorn cpu_http:app --host 0.0.0.0 --port 9001 --app-dir D:/Project/inferserver/inference/docker/config

# Linux / macOS
ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True \
WORKFLOWS_PLUGINS=business_plugin \
uvicorn cpu_http:app --host 0.0.0.0 --port 9001 --app-dir D:/Project/inferserver/inference/docker/config
```

**入口腳本**：`docker/config/cpu_http.py`（CPU）或 `docker/config/gpu_http.py`（GPU）

啟動流程：
1. 建立 `AIEModelRegistry(ROBOFLOW_MODEL_TYPES)` — 支援本地路徑 + Roboflow 遠端
2. 建立 `ModelManager` → 包裝 `WithFixedSizeCache`（LRU，大小由 `MAX_ACTIVE_MODELS` 控制）
3. 建立 `HttpInterface(model_manager)` → FastAPI app
4. 若 `ENABLE_STREAM_API=True`，另起子進程跑 Stream Manager

**推論路徑**：HTTP request → `ModelManager.model_infer_sync()` → `model.infer_from_request()` → 回傳結果

**適用場景**：
- Windows / Linux / macOS 皆可
- 開發、測試、小規模部署
- Workflow 執行
- AIE 模型推論（YOLO + EfficientAD）

### 3.2 Parallel 模式（Enterprise）

多進程 pipeline，將推論拆分為 preprocess → predict → postprocess 三階段，
透過 Redis + Celery 分散到不同進程，並在 predict 階段做 dynamic batching。

```
┌─────────────────────────────────────────────────────────────┐
│                    5 OS processes                            │
│                                                             │
│  ┌─────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ redis   │  │ celery -Q pre│  │ celery -Q post        │  │
│  │ server  │  │ (preprocess) │  │ (postprocess)         │  │
│  └────┬────┘  └──────┬───────┘  └───────────┬───────────┘  │
│       │              │ SharedMemory          │              │
│       │         ┌────▼──────────┐            │              │
│       │         │ infer.py      │            │              │
│       │         │ (predict +    │────────────┘              │
│       │         │  batching)    │                           │
│       │         └───────────────┘                           │
│  ┌────▼──────────────────────────┐                          │
│  │ gunicorn (HTTP API)           │                          │
│  │ DispatchModelManager          │                          │
│  │ → 不做推論，只分派到 Redis    │                          │
│  └───────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

**各進程的 model_manager：**

| 進程 | model_manager 類型 | 作用 |
|------|-------------------|------|
| gunicorn HTTP | `DispatchModelManager` | 不做推論，將 request 分派到 Redis queue |
| celery -Q pre | `StubLoaderManager` → `WithFixedSizeCache` | 執行 `model.preprocess()` |
| celery -Q post | `StubLoaderManager` → `WithFixedSizeCache` | 執行 `model.postprocess()` |
| infer.py | `ModelManager` → `WithFixedSizeCache` | 執行 `model.predict()`，含 dynamic batching |
| redis-server | — | message broker + pub/sub |

**推論路徑**：
```
HTTP request
  → DispatchModelManager.model_infer_sync()
    → checker.add_task() → celery preprocess.delay()
  → preprocess(): model.preprocess() → SharedMemory → redis.zadd("infer:{model_id}")
  → InferServer.infer_loop(): get_batch() → model.predict() → SharedMemory → postprocess.delay()
  → postprocess(): model.postprocess() → redis.publish("results")
  → ResultsChecker.loop() → event.set() → 回傳結果
```

**前置需求**：
- **Linux / Docker 限定**（依賴 SharedMemory + Redis + Celery）
- Redis server（container 內自動啟動）
- Celery + gunicorn（`requirements.parallel.txt`）
- 模型必須實作 `preprocess()` / `predict()` / `postprocess()` 三個方法

**適用場景**：
- 高併發生產環境（GPU 利用率最大化）
- 需要 dynamic batching 的場景
- 官方效能數據：物件偵測 +80%、實例分割 +121%、分類 +34%

### 3.3 AIE 模型與 Parallel 模式

> **重要**：AIE 模型目前僅支援一般模式。

Parallel 模式有以下限制，尚未為 AIE 模型適配：

| 限制 | 說明 |
|------|------|
| Registry 硬編碼 | `tasks.py`、`infer.py`、`parallel_http_config.py` 使用 `RoboflowModelRegistry`，非 `AIEModelRegistry` |
| `request_from_type()` | 僅支援 4 種 task type，不含 `anomaly-detection` |
| `response_from_type()` | 同上 |
| `get_model_type()` | 透過 Roboflow API 或 `GENERIC_MODELS` dict 解析，不支援本地目錄路徑 |

若未來需要 parallel 支援，需修改上述 4 處。一般模式下的單進程推論對 AIE 場景已足夠。

---

## 4. 環境變數

### 4.1 通用

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES` | `False` | 設為 `True` 允許本地路徑載入模型 |
| `WORKFLOWS_PLUGINS` | `""` | 要載入的 plugin，逗號分隔 |
| `PORT` | `9001` | HTTP API port |
| `MAX_ACTIVE_MODELS` | `1` | 模型 LRU cache 大小 |
| `NUM_WORKERS` | `1` | uvicorn / gunicorn worker 數量 |
| `USE_INFERENCE_MODELS` | `True` | 啟用 inference_models 套件 |
| `ENABLE_STREAM_API` | `True` | 啟用 Stream Management API（一般模式） |
| `ENABLE_BUILDER` | `False` | 啟用 Builder API（workflow 註冊/管理，見 Workflow 手冊 §6.2） |
| `MODEL_CACHE_DIR` | `/tmp/cache` | 模型與 workflow 儲存目錄 |

### 4.2 Parallel 模式專用

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `REDIS_HOST` | `localhost` | Redis server 地址 |
| `REDIS_PORT` | `6379` | Redis server port |
| `NUM_CELERY_WORKERS` | `1` | 每個 Celery queue 的 worker concurrency |
| `CELERY_LOG_LEVEL` | `WARNING` | Celery log level |
| `NUM_PARALLEL_TASKS` | `1000` | 最大同時進行的推論任務數 |
| `MAX_BATCH_SIZE` | `inf` | dynamic batching 最大 batch size（預設無上限，實際 cap 32） |
| `STUB_CACHE_SIZE` | `1` | Celery worker 的模型 cache 大小 |

---

## 5. 模型目錄管理

### 5.1 目錄結構

每個模型一個目錄，包含路由設定和權重檔：

```
model_dir/
├── model_config.json    # AutoModel 路由用（必須）
└── weights.pt           # 或 weights.nst（必須）
```

### 5.2 model_config.json 範例

**YOLO 物件偵測：**
```json
{
  "model_architecture": "aie",
  "task_type": "object-detection",
  "backend_type": "ultralytics"
}
```

**YOLO 分類：**
```json
{
  "model_architecture": "aie",
  "task_type": "classification",
  "backend_type": "ultralytics"
}
```

**YOLO 實例分割：**
```json
{
  "model_architecture": "aie",
  "task_type": "instance-segmentation",
  "backend_type": "ultralytics"
}
```

**EfficientAD 異常檢測：**
```json
{
  "model_architecture": "aie",
  "task_type": "anomaly-detection",
  "backend_type": "torch"
}
```

AD 模型可附帶 `params.json`（可選）：
```json
{
  "model_params": {
    "imagesize": 1024
  }
}
```
若未提供，預設 `imagesize=256`。

### 5.3 載入流程

```
訓練完成 → 產出模型目錄（weights + model_config.json）
         → 放到指定路徑（本地或 Docker 掛載）
         → POST /model/add {"model_id": "/models/locate_model"}
         → server 載入模型，可開始推論
         → 模型更新時，覆蓋權重檔，再次 /model/add 即可重新載入
```

`model_id` 可以是本地絕對路徑。server 用 `os.path.isdir()` 判斷是本地路徑還是 Roboflow 遠端 ID。

---

## 6. 構建 wheel（部署用）

非開發機器上用 whl 安裝，不需要原始碼：

```bash
# 在開發機上構建 whl
pip wheel --no-deps -w dist/ D:\Project\TrainingTool\AIE
pip wheel --no-deps -w dist/ D:\Project\inferserver\inference
pip wheel --no-deps -w dist/ D:\Project\inferserver\inference\inference_models
pip wheel --no-deps -w dist/ /path/to/business_plugin          # 可選

# 產出檔案（在 dist/ 目錄下）：
#   aie-0.24.0-py3-none-any.whl
#   inference-0.x.x-py3-none-any.whl
#   inference_models-0.x.x-py3-none-any.whl
#   business_plugin-0.1.0-py3-none-any.whl

# 在部署機上安裝
pip install dist/aie-*.whl
pip install dist/inference-*.whl
pip install dist/inference_models-*.whl
pip install dist/business_plugin-*.whl                          # 可選
```

---

## 7. Docker 部署（一般模式）

### 7.1 自建 Docker image

基於官方 GPU image 擴展。該 image 包含 PyTorch（CUDA 12.4）、ultralytics、ONNX Runtime，
AIE 的 YOLO 和 EfficientAD 所需的 runtime 都已涵蓋。

```dockerfile
FROM roboflow/roboflow-inference-server-gpu:latest

# Install AIE toolkit + inference（從本地 whl）
COPY dist/*.whl /tmp/wheels/
RUN pip3 install /tmp/wheels/*.whl && rm -rf /tmp/wheels

# （可選）Install business plugin
COPY business_plugin/ /tmp/business/
RUN pip3 install /tmp/business/ && rm -rf /tmp/business

ENV WORKFLOWS_PLUGINS=business_plugin
ENV ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True
```

### 7.2 構建與啟動

```bash
# 構建 image（需先完成 §6 的 whl 構建）
docker build -t my-inference-server:latest .

# 啟動（需要 NVIDIA Container Toolkit）
docker run --gpus all -p 9001:9001 \
  -v /srv/models:/models \
  -e MAX_ACTIVE_MODELS=3 \
  my-inference-server:latest
```

### 7.3 模型目錄掛載

在 host 上規劃一個固定的模型目錄，掛載到容器內：

```
Host 目錄結構（範例）：
/srv/models/
├── locate_model/                     # YOLO 物件偵測
│   ├── model_config.json
│   └── weights.pt
├── particle_model/                   # YOLO 粒子偵測
│   ├── model_config.json
│   └── weights.pt
└── ad_model/                         # EfficientAD 異常檢測
    ├── model_config.json
    ├── params.json
    └── weights.nst
```

```bash
docker run --gpus all -p 9001:9001 \
  -v /srv/models:/models \
  -e ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True \
  -e MAX_ACTIVE_MODELS=3 \
  my-inference-server:latest
```

```python
import requests

# 用容器內路徑載入模型
requests.post("http://localhost:9001/model/add", json={
    "model_id": "/models/locate_model"
})
requests.post("http://localhost:9001/model/add", json={
    "model_id": "/models/ad_model"
})
```

### 7.4 模型更新（迭代訓練）

訓練工具在 host 的 `/srv/models/` 下更新權重檔後，
對 server 重新呼叫 `/model/add` 即可載入新版本。不需要重啟容器或重建 image。

---

## 8. Docker 部署（Parallel 模式）

> **注意**：AIE 模型目前不支援 Parallel 模式（見 §3.3）。
> 本節說明 Parallel 模式的部署方式，適用於 Roboflow 原生模型的高併發場景。

### 8.1 官方 Parallel image

官方提供 `Dockerfile.onnx.gpu.parallel`，已包含 Redis 和 Celery：

```bash
# 從 repo 根目錄構建
docker build -t roboflow/inference-server-gpu-parallel:latest \
  -f docker/dockerfiles/Dockerfile.onnx.gpu.parallel .
```

該 image 的 ENTRYPOINT 是 `python3 entrypoint.py`，啟動時自動拉起 5 個進程。

### 8.2 啟動

```bash
docker run --gpus all -p 9001:9001 \
  -v /srv/models:/models \
  -e NUM_WORKERS=2 \
  -e NUM_CELERY_WORKERS=2 \
  -e MAX_ACTIVE_MODELS=3 \
  -e MAX_BATCH_SIZE=16 \
  roboflow/inference-server-gpu-parallel:latest
```

**關鍵差異**：
- Redis 在 container 內自動啟動，不需要外部 Redis
- `NUM_WORKERS` 控制 gunicorn HTTP worker 數
- `NUM_CELERY_WORKERS` 控制 pre/post queue 各自的 concurrency
- `MAX_BATCH_SIZE` 控制 dynamic batching 上限（預設 32）

### 8.3 一般模式 vs Parallel 模式對比

| 項目 | 一般模式 | Parallel 模式 |
|------|---------|--------------|
| 進程數 | 1（uvicorn） | 5（redis + 2 celery + infer + gunicorn） |
| 入口腳本 | `cpu_http.py` / `gpu_http.py` | `entrypoint.py` |
| 推論路徑 | 同步，單進程內完成 | pipeline，跨進程 SharedMemory |
| Dynamic batching | 無 | 有（infer.py 自動湊 batch） |
| Workflow 支援 | 完整 | 有限（透過 HTTP API 觸發） |
| 額外依賴 | 無 | Redis、Celery、gunicorn |
| 平台 | Windows / Linux / macOS | Linux / Docker 限定 |
| AIE 模型 | 支援 | **不支援** |
| 適用場景 | 開發 / 測試 / 一般部署 | 高併發生產環境 |

---

## 9. 驗證安裝

```python
# 確認所有 AIE 模組可匯入
from inference_models.models.aie.aie_object_detection import AIEForObjectDetection
from inference_models.models.aie.aie_classification import AIEForClassification
from inference_models.models.aie.aie_instance_segmentation import AIEForInstanceSegmentation
from inference_models.models.aie.aie_anomaly_detection import AIEForAnomalyDetection
from inference.core.registries.aie import AIEModelRegistry
from inference.core.workflows.core_steps.models.aie.anomaly_detection.v1 import AIEAnomalyDetectionBlockV1
print("All AIE modules imported successfully.")

# 確認 AIE Training Toolkit 可匯入（.nst 解密需要）
import AIE
print("AIE Training Toolkit available.")
```
