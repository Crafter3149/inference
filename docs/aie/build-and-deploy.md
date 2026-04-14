# AIE Inference Server — Build & Deploy

三種分發方式，依部署場景選擇。

## 分發方式總覽

| | Dev 開發環境 | Python whl 分發 | Docker 容器分發 |
|--|-------------|----------------|----------------|
| **適用場景** | 開發、除錯、改程式碼 | 已有 Python + CUDA 的機器 | 任何有 Docker + NVIDIA GPU 的 Linux |
| **前置需求** | Python 3.10, CUDA, Git | Python 3.10, CUDA | Docker, NVIDIA Container Toolkit |
| **模型更新** | 覆蓋檔案即可 | 覆蓋檔案即可 | volume mount，覆蓋即可 |
| **程式碼更新** | 即時生效 | 重新打包 whl | 重新 build image |
| **Image 大小** | — | ~500 MB (whl) | ~8-12 GB (含 CUDA + PyTorch) |
| **啟動命令** | `inference aie start` | `inference aie start` | `docker run --gpus all ...` |

---

## 共用前置：AIE Training Toolkit

三種方式都需要 AIE Training Toolkit（提供 `.nst` 解密 + YOLO pickle 命名空間）：

```bash
# 從源碼安裝（dev）
pip install -e D:\Project\TrainingTool\AIE

# 從 whl 安裝（分發）
pip install aie-0.24.0-py3-none-any.whl
```

Docker 方式中，AIE toolkit 已包含在 image 建置流程中，不需額外安裝。

---

## 方法 A：Dev 開發環境

修改程式碼後立即生效，適合開發和除錯。

### A.1 安裝

```bash
# 1. PyTorch（依 CUDA 版本選擇 index）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. AIE Training Toolkit
pip install -e D:\Project\TrainingTool\AIE

# 3. inference server（editable，改 .py 即時生效）
pip install -e D:\Project\inferserver\inference

# 4. inference_models（editable）
pip install -e D:\Project\inferserver\inference\inference_models

# 5.（可選）業務邏輯 plugin
pip install -e /path/to/business_plugin
```

### A.2 啟動

```bash
# 方式一：CLI（推薦）
inference aie start --port 9001 --max-active-models 3

# 方式二：uvicorn（可自訂參數）
set ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True
uvicorn gpu_http:app --host 0.0.0.0 --port 9001 --app-dir D:/Project/inferserver/inference/docker/config
```

### A.3 Dashboard 開發

Dashboard 需要單獨 build：

```bash
cd inference/landing/dashboard
npm install
npm run build          # 產出到 ../out/
```

build 後重啟 server，訪問 `http://localhost:9001/` 即可看到 dashboard。

---

## 方法 B：Python whl 分發

適用於已安裝 Python + CUDA 的部署機器。不需要原始碼。

### B.1 在開發機上打包

有兩種 whl 策略：

#### 策略一：inference-aie 單一 whl（推薦）

把 `inference` + `inference_models` + `inference_cli` + `inference_sdk` + dashboard 打包成一個 whl：

```bash
cd D:\Project\inferserver\inference

# 確保 dashboard 已 build
cd inference/landing/dashboard && npm run build && cd ../../..

# 打包
pip install wheel
python .release/pypi/inference.aie.setup.py bdist_wheel
# 產出：dist/inference_aie-1.2.0-py3-none-any.whl
```

分發清單（共 3 個檔案）：

```
dist/
├── aie-0.24.0-py3-none-any.whl              # AIE Training Toolkit
├── inference_aie-1.2.0-py3-none-any.whl     # inference server（含 dashboard）
└── install.bat                               # 安裝腳本（見下方）
```

#### 策略二：分離 whl

需要獨立更新 inference_models 時使用：

```bash
cd D:\Project\inferserver\inference

pip install wheel

# 打包 inference core/cli/sdk
python .release/pypi/inference.core.setup.py bdist_wheel
rm -rf build/*
python .release/pypi/inference.cli.setup.py bdist_wheel
rm -rf build/*
python .release/pypi/inference.sdk.setup.py bdist_wheel
rm -rf build/*

# 打包 inference_models
cd inference_models
pip wheel --no-deps -w ../dist/ .
cd ..

# 產出 4 個 whl
```

### B.2 在部署機上安裝

```bash
# 1. PyTorch（必須先裝，inference-aie 不含 torch）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. AIE Training Toolkit
pip install aie-0.24.0-py3-none-any.whl

# 3. inference server
pip install inference_aie-1.2.0-py3-none-any.whl

# 4.（可選）業務邏輯 plugin
pip install business_plugin-0.1.0-py3-none-any.whl
```

#### install.bat 範例

```bat
@echo off
echo === Installing AIE Inference Server ===

echo [1/3] Installing PyTorch (CUDA 12.4)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo [2/3] Installing AIE Training Toolkit...
pip install aie-0.24.0-py3-none-any.whl

echo [3/3] Installing inference server...
pip install inference_aie-1.2.0-py3-none-any.whl

echo.
echo === Done. Start server with: inference aie start ===
pause
```

### B.3 啟動

```bash
inference aie start --port 9001 --max-active-models 3
```

---

## 方法 C：Docker 容器分發

適用於任何有 Docker + NVIDIA GPU 的 Linux 機器。不需要 Python 環境。

### C.1 前置需求（部署機）

```bash
# Docker Engine
curl -fsSL https://get.docker.com | sh

# NVIDIA Container Toolkit（讓 Docker 存取 GPU）
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 驗證
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### C.2 在開發機上建置 image

```bash
cd D:\Project\inferserver\inference

# 建置（首次約 15-30 分鐘，後續有 cache 約 1-3 分鐘）
docker build -t aie-server:v1 -f docker/dockerfiles/Dockerfile.aie.torch.gpu .
```

`Dockerfile.aie.torch.gpu` 兩階段建置：
- **Builder**：CUDA 12.4 devel → PyTorch cu124 → inference_models deps（torch only，不含 onnx）→ core requirements
- **Runtime**：CUDA 12.4 runtime → 複製 Python → 複製源碼 → AIE ENV defaults

### C.3 匯出與傳輸

```bash
# 匯出（壓縮後約 4-6 GB）
docker save aie-server:v1 | gzip > aie-server-v1.tar.gz

# 傳到部署機（scp / USB / NAS）
scp aie-server-v1.tar.gz user@deploy-machine:/opt/aie/
```

### C.4 在部署機上載入並啟動

```bash
# 載入 image
docker load < aie-server-v1.tar.gz

# 啟動
docker run --gpus all -p 9001:9001 \
  -v /srv/models:/models \
  aie-server:v1
```

### C.5 自訂參數

透過環境變數覆蓋預設值：

```bash
docker run --gpus all -p 8080:8080 \
  -v /srv/models:/models \
  -e PORT=8080 \
  -e MAX_ACTIVE_MODELS=5 \
  -e MAX_BATCH_SIZE=4 \
  -e WORKFLOWS_PLUGINS=business_plugin \
  aie-server:v1
```

### C.6 更新流程

| 更新項目 | 操作 |
|---------|------|
| 模型權重 | 在 host 覆蓋 `/srv/models/` 下的檔案 → `POST /model/add` 重新載入 |
| 程式碼 | 在開發機重新 `docker build` → `docker save` → 傳輸 → `docker load` → 重啟容器 |
| 新增 plugin | 需修改 Dockerfile 加入 plugin → 重新 build |

---

## 模型目錄結構

三種分發方式共用同樣的模型目錄格式。

### 目錄結構

```
models/
├── locate_model/               # YOLO 物件偵測
│   ├── model_config.json
│   └── weights.pt
├── particle_model/             # YOLO 粒子偵測
│   ├── model_config.json
│   └── weights.pt
└── ad_model/                   # EfficientAD 異常檢測
    ├── model_config.json
    ├── params.json             # 可選
    └── weights.nst
```

### model_config.json

每個模型目錄必須包含 `model_config.json`，用於路由到正確的模型類別：

| task_type | backend_type | 模型類別 |
|-----------|-------------|---------|
| `object-detection` | `ultralytics` | AIEForObjectDetection |
| `classification` | `ultralytics` | AIEForClassification |
| `instance-segmentation` | `ultralytics` | AIEForInstanceSegmentation |
| `anomaly-detection` | `torch` | AIEForAnomalyDetection |

範例（YOLO OD）：
```json
{
  "model_architecture": "aie",
  "task_type": "object-detection",
  "backend_type": "ultralytics"
}
```

範例（EfficientAD）：
```json
{
  "model_architecture": "aie",
  "task_type": "anomaly-detection",
  "backend_type": "torch"
}
```

AD 可附帶 `params.json`（可選，預設 `imagesize=256`）：
```json
{
  "model_params": {
    "imagesize": 1024
  }
}
```

### 載入與更新

```python
import requests

BASE = "http://localhost:9001"

# 載入模型（model_id = 本地絕對路徑）
requests.post(f"{BASE}/model/add", json={"model_id": "/srv/models/locate_model"})

# 查看已載入模型
requests.get(f"{BASE}/model/registry").json()

# 卸載模型
requests.post(f"{BASE}/model/remove", json={"model_id": "/srv/models/locate_model"})

# 模型更新：覆蓋權重檔後重新 /model/add 即可（不需重啟 server）
```

---

## 環境變數

### 核心

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `PORT` | `9001` | HTTP API port |
| `HOST` | `0.0.0.0` | 綁定地址 |
| `MAX_ACTIVE_MODELS` | `3` (AIE) / `1` (原版) | 模型 LRU cache 大小 |
| `MAX_BATCH_SIZE` | `1` | 推論 batch size（GPU 安全值為 1） |
| `NUM_WORKERS` | `1` | uvicorn worker 數量 |

### AIE 專用

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES` | `True` | 允許本地路徑載入模型 |
| `ENABLE_STREAM_API` | `False` | AIE 不需要 Stream API |
| `ACTIVE_LEARNING_ENABLED` | `False` | AIE 不需要 Active Learning |
| `ENABLE_BUILDER` | `true` | 啟用 Workflow Builder UI |
| `ENABLE_DASHBOARD` | `true` | 啟用 Dashboard |
| `WORKFLOWS_PLUGINS` | `""` | 業務邏輯 plugin（逗號分隔） |

> `inference aie start` 和 `Dockerfile.aie.torch.gpu` 已內建這些預設值，不需手動設定。

---

## 啟動模式

### 一般模式（Normal Mode）

所有分發方式使用的模式。單一 uvicorn 進程，推論在同一進程完成。

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

推論路徑：
```
HTTP request → ModelManager.model_infer_sync() → model.infer_from_request() → response
```

### Parallel 模式（Enterprise）

> AIE 模型目前不支援 Parallel 模式。

5 進程 pipeline（redis + 2 celery + GPU infer + gunicorn），全部跑在單一容器內。
適用於 Roboflow 原生模型的高併發場景。詳見源專案文件。

---

## 驗證

### 啟動後檢查

```bash
# Health check
curl http://localhost:9001/healthz
# {"status":"healthy"}

# Server info
curl http://localhost:9001/info
# {"name":"Roboflow Inference Server","version":"1.2.0",...}

# 已載入模型
curl http://localhost:9001/model/registry
# {"models":[...]}

# Dashboard
# 瀏覽器開啟 http://localhost:9001/
```

### 推論測試

```python
import requests, base64
from pathlib import Path

BASE = "http://localhost:9001"

# 載入模型
requests.post(f"{BASE}/model/add", json={
    "model_id": "/srv/models/locate_model"
})

# 推論
img_b64 = base64.b64encode(Path("test.jpg").read_bytes()).decode()
resp = requests.post(f"{BASE}/infer/object_detection", json={
    "model_id": "/srv/models/locate_model",
    "image": {"type": "base64", "value": img_b64},
    "confidence": 0.5,
})
print(resp.json())
```

### Python import 檢查

```python
from inference_models.models.aie.aie_object_detection import AIEForObjectDetection
from inference_models.models.aie.aie_classification import AIEForClassification
from inference_models.models.aie.aie_instance_segmentation import AIEForInstanceSegmentation
from inference_models.models.aie.aie_anomaly_detection import AIEForAnomalyDetection
from inference.core.registries.aie import AIEModelRegistry
import AIE
print("All AIE modules OK.")
```

---

## 套件架構參考

```
inference (repo root)
├── .release/pypi/
│   └── inference.aie.setup.py         # inference-aie 單一 whl 打包腳本
├── docker/
│   ├── config/
│   │   ├── gpu_http.py                # 原版 GPU 入口（含 AL + Stream）
│   │   └── aie_http.py               # AIE Docker 入口（精簡版）
│   └── dockerfiles/
│       ├── Dockerfile.onnx.gpu        # 原版 GPU（ONNX runtime）
│       └── Dockerfile.aie.torch.gpu   # AIE GPU（PyTorch runtime）
├── inference/
│   ├── core/registries/aie.py         # AIEModelRegistry
│   ├── core/models/inference_models_adapters.py  # AD adapter
│   └── landing/out/                   # Dashboard 靜態檔案
├── inference_models/
│   └── inference_models/models/aie/   # AIE 模型類別
├── inference_cli/
│   └── aie_server.py                  # inference aie start 入口
└── inference_sdk/
```
