# 開發計畫：inference_models AIE 模型整合

## !! 重要背景 — 每次都會被壓縮遺失，必讀 !!

### AIE Training Toolkit 是什麼
- 位置：`D:\Project\TrainingTool\AIE`
- 用途：訓練 YOLO（detect/classify/segment）和 EfficientAD（anomaly detection）模型
- 產出：`.nst` 加密權重檔 + `.json` 參數檔
- **必須安裝**：`pip install -e D:\Project\TrainingTool\AIE`，否則 .nst 載入會失敗

### .nst 檔案格式（加密）
- 加密演算法在 `D:\Project\TrainingTool\AIE\AIE\utils\encrypt.py` 的 `file_encode_v2` / `file_decode_v2`
- Header: `[24, 97, 28, 98]`（4 bytes magic number）
- 內容被 reverse → 切 20 塊 → shuffle → 加 padding
- 解密後得到的可能是：
  - **YOLO .pt**（標準 ultralytics 權重，pickle 裡引用了 AIE 命名空間 → 需要 AIE 套件在 import path）
  - **TorchScript**（AIE 導出的 .torchscript 加密版，用 `torch.jit.load` 載入）
  - **EfficientAD .pt**（pickle 裡有 `AIE.models.efficient_ad.architecture.EfficientAD` 類別）
- `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\decrypt.py` 已有可用的 `decrypt_nst()` 函數

### .nst 正確的載入順序（參考 `D:\Project\Bosch\infer\preview_ad.py:90-104`）
```
1. 嘗試 torch.load(path)           → 若成功，是未加密的 .pt
2. decrypt_nst(path) → bytes
3. 嘗試 torch.load(BytesIO(bytes)) → 若成功，是加密的 .pt（YOLO 或 EfficientAD）
4. 嘗試 torch.jit.load(BytesIO(bytes)) → 若成功，是加密的 TorchScript
```

### EfficientAD 模型架構
- `EfficientAD(nn.Module)` 定義在 `D:\Project\TrainingTool\AIE\AIE\models\efficient_ad\architecture.py:218-358`
- 三個子網路：Teacher（384ch）、Student（768ch）、AutoEncoder
- `forward(x)` → `(combined_maps: Tensor[B,1,256,256], images_score: Tensor[B])`
- combined_maps 是 anomaly heatmap，images_score 是圖片級異常分數
- 輸入需要 `transforms.Resize([256,256]) + transforms.ToTensor()`

### YOLO 模型（AIE 訓練的）
- 跟標準 ultralytics 一樣，但 pickle 裡引用了 AIE 命名空間
- `YOLO(path)` 載入，支援 detect/classify/segment task
- 載入前必須確保 `import AIE` 成功

### 測試用模型位置
- `D:\Project\Bosch\infer\model\task3_2026\locate_model3.pt` — YOLO detect（class: Object）
- `D:\Project\Bosch\infer\model\task3_2026\particle_p2_v6\weights\best.pt` — YOLO detect（class: Particle）
- `D:\Project\Bosch\infer\model\task3_2026\best.nst` — EfficientAD（anomaly detection）
- 測試圖片：`D:\Project\Bosch\2026\AI_Competition_Data\Test3_Particle_Det_v4\val\images\`（134 張）

---

## 目標

在 `inference_models/inference_models/models/aie/` 模組中，實作所有 AIE 產出模型的載入和推論，
讓 `AutoModel.from_pretrained()` 可以正確路由到對應的模型類別。

### 支援的 4 種任務
1. **Object Detection** — YOLO detect，`BackendType.ULTRALYTICS`
2. **Classification** — YOLO classify，`BackendType.ULTRALYTICS`
3. **Instance Segmentation** — YOLO segment，`BackendType.ULTRALYTICS`
4. **Anomaly Detection** — EfficientAD，`BackendType.TORCH`

---

## 目前狀態（需要全部重做）

之前的實作有以下問題，已確認需要刪除重寫：
- 3 個 YOLO 模型檔案大量 copy-paste（from_pretrained、pre_process 幾乎一樣）
- .nst 載入碰到 `ModuleNotFoundError: No module named 'AIE'` 就放棄，改用 .pt 繞過
- 測試時自己寫 FastAPI server，完全繞過 inference server 的 Adapter 層
- 不支援 EfficientAD
- 不支援 TorchScript 格式的 .nst

### 需要刪除的檔案
```
D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_object_detection_ultralytics.py
D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_classification_ultralytics.py
D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_instance_segmentation_ultralytics.py
```

### 需要回退的修改
- `D:\Project\inferserver\inference\inference_models\inference_models\models\auto_loaders\models_registry.py` — 移除之前新增的 3 個 AIE 條目

### 可保留的檔案
- `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\__init__.py` — 保留（空檔）
- `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\decrypt.py` — 保留（`decrypt_nst()` 函數正確可用）

---

## 實作計畫

### Step 0: 清理舊實作
刪除 3 個舊模型檔案，回退 registry 修改。保留 `__init__.py` 和 `decrypt.py`。

### Step 1: 建立 YOLO 共用邏輯
**檔案**: `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\_aie_ultralytics_base.py`

抽出 3 個 YOLO 任務的共用邏輯到一個 base class：

```python
class _AIEUltralyticsBase:
    """共用的 .nst/.pt 載入和 ultralytics 推論邏輯"""

    @staticmethod
    def _load_ultralytics_model(model_dir: Path, device: str) -> YOLO:
        """統一的模型載入流程，處理 .pt / .nst / TorchScript"""
        # 1. 找到權重檔（weights.pt 或 weights.nst）
        # 2. .nst → decrypt_nst() → 寫暫存檔
        # 3. 嘗試 YOLO(path) 載入
        #    ⚠ 需要 AIE 套件已安裝，否則 raise 明確錯誤訊息
        # 4. 回傳 YOLO 物件

    @staticmethod
    def _images_to_numpy_list(images) -> List[np.ndarray]:
        """統一的輸入轉換：Tensor/ndarray/List → List[np.ndarray] (BGR, HWC)"""
```

### Step 2: Object Detection
**檔案**: `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_object_detection.py`

```python
class AIEForObjectDetection(
    ObjectDetectionModel[List[np.ndarray], List[Tuple[int,int]], list]
):
```

- 繼承 `ObjectDetectionModel`（`D:\Project\inferserver\inference\inference_models\inference_models\models\base\object_detection.py`）
- 使用 `_AIEUltralyticsBase` 的共用方法（mixin 或 composition）
- `from_pretrained()` → `_load_ultralytics_model()` + 建構 class_names
- `pre_process()` → `_images_to_numpy_list()` + 記錄原始尺寸
- `forward()` → `self._model.predict(images, conf=, iou=, verbose=False)` + Lock
- `post_process()` → 從 Results 提取 boxes → `List[Detections]`

### Step 3: Classification
**檔案**: `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_classification.py`

```python
class AIEForClassification(
    ClassificationModel[List[np.ndarray], list]
):
```

- 繼承 `ClassificationModel`（`D:\Project\inferserver\inference\inference_models\inference_models\models\base\classification.py`）
- **注意**：Classification 的 `pre_process` 不回傳 metadata，只回傳 `PreprocessedInputs`
- `post_process()` → `result.probs.top1` + `result.probs.data` → `ClassificationPrediction`

### Step 4: Instance Segmentation
**檔案**: `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_instance_segmentation.py`

```python
class AIEForInstanceSegmentation(
    InstanceSegmentationModel[List[np.ndarray], List[Tuple[int,int]], list]
):
```

- 繼承 `InstanceSegmentationModel`（`D:\Project\inferserver\inference\inference_models\inference_models\models\base\instance_segmentation.py`）
- `post_process()` → 從 Results 提取 boxes + masks → `List[InstanceDetections]`

### Step 5: 新增 anomaly detection base class
**檔案**: `D:\Project\inferserver\inference\inference_models\inference_models\models\base\anomaly_detection.py`

```python
@dataclass
class AnomalyDetectionResult:
    anomaly_map: torch.Tensor    # (1, H, W) — pixel-level anomaly heatmap
    anomaly_score: torch.Tensor  # scalar — image-level anomaly score

class AnomalyDetectionModel(ABC, Generic[PreprocessedInputs, RawPrediction]):
    # 與 ClassificationModel 結構類似（pre_process 不回傳 metadata）
    @abstractmethod
    def from_pretrained(cls, ...) -> "AnomalyDetectionModel": ...
    @property
    @abstractmethod
    def class_names(self) -> List[str]: ...  # 通常是 ["good", "anomaly"]
    def infer(self, images, **kwargs) -> List[AnomalyDetectionResult]: ...
    @abstractmethod
    def pre_process(self, images, **kwargs) -> PreprocessedInputs: ...
    @abstractmethod
    def forward(self, pre_processed, **kwargs) -> RawPrediction: ...
    @abstractmethod
    def post_process(self, results, **kwargs) -> List[AnomalyDetectionResult]: ...
```

### Step 6: Anomaly Detection
**檔案**: `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_anomaly_detection.py`

```python
class AIEForAnomalyDetection(
    AnomalyDetectionModel[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
):
```

- 繼承 `AnomalyDetectionModel`（`D:\Project\inferserver\inference\inference_models\inference_models\models\base\anomaly_detection.py`）
- `from_pretrained()`:
  1. 找 weights.nst 或 weights.pt
  2. .nst → decrypt → BytesIO
  3. 先嘗試 `torch.load()` → EfficientAD 物件
  4. 失敗則 `torch.jit.load()` → TorchScript 物件
  5. 讀 .json 參數檔取得 imagesize 等設定
- `pre_process()`:
  - PIL Image → `Resize([size, size])` + `ToTensor()` → `torch.Tensor [B,3,H,W]`
- `forward()`:
  - `model(tensor)` → `(combined_maps, images_score)`
- `post_process()`:
  - → `List[AnomalyDetectionResult]`

### Step 7: 更新 entities.py 和 registry

**`D:\Project\inferserver\inference\inference_models\inference_models\models\auto_loaders\entities.py`**:
- import `AnomalyDetectionModel` 並加入 `AnyModel` union

**`D:\Project\inferserver\inference\inference_models\inference_models\models\auto_loaders\models_registry.py`**:
- 新增 `ANOMALY_DETECTION_TASK = "anomaly-detection"`
- 新增 4 個 `REGISTERED_MODELS` 條目：

```python
("aie", OBJECT_DETECTION_TASK, BackendType.ULTRALYTICS): LazyClass(
    module_name="inference_models.models.aie.aie_object_detection",
    class_name="AIEForObjectDetection",
),
("aie", CLASSIFICATION_TASK, BackendType.ULTRALYTICS): LazyClass(
    module_name="inference_models.models.aie.aie_classification",
    class_name="AIEForClassification",
),
("aie", INSTANCE_SEGMENTATION_TASK, BackendType.ULTRALYTICS): LazyClass(
    module_name="inference_models.models.aie.aie_instance_segmentation",
    class_name="AIEForInstanceSegmentation",
),
("aie", ANOMALY_DETECTION_TASK, BackendType.TORCH): LazyClass(
    module_name="inference_models.models.aie.aie_anomaly_detection",
    class_name="AIEForAnomalyDetection",
),
```

---

## 檔案清單

### 新增 (5 個)
```
D:\Project\inferserver\inference\inference_models\inference_models\models\base\anomaly_detection.py      # AnomalyDetectionModel base + AnomalyDetectionResult
D:\Project\inferserver\inference\inference_models\inference_models\models\aie\_aie_ultralytics_base.py   # YOLO 共用載入/前處理邏輯
D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_object_detection.py    # AIEForObjectDetection
D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_classification.py      # AIEForClassification
D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_anomaly_detection.py   # AIEForAnomalyDetection
```

### 修改 (2 個)
```
D:\Project\inferserver\inference\inference_models\inference_models\models\auto_loaders\entities.py       # AnyModel union 加入 AnomalyDetectionModel
D:\Project\inferserver\inference\inference_models\inference_models\models\auto_loaders\models_registry.py # 4 個 REGISTERED_MODELS 條目 + ANOMALY_DETECTION_TASK
```

### 保留 (2 個)
```
D:\Project\inferserver\inference\inference_models\inference_models\models\aie\__init__.py                # 保留
D:\Project\inferserver\inference\inference_models\inference_models\models\aie\decrypt.py                 # 保留（已驗證正確）
```

### 刪除 (3 個)
```
D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_object_detection_ultralytics.py
D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_classification_ultralytics.py
D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_instance_segmentation_ultralytics.py
```

### 備註
- 如果 `aie_instance_segmentation.py` 跟 `aie_object_detection.py` 的差異只有 `post_process`，
  考慮讓 Instance Seg 繼承 OD 並只覆寫 `post_process`

---

## 關鍵 gotchas（過去反覆出錯的點）

1. **AIE 套件必須可 import** — YOLO .nst 解密後的 .pt 是 pickle 格式，裡面引用了 `AIE` 命名空間。
   如果 `import AIE` 失敗，`YOLO()` 載入就會報 `ModuleNotFoundError: No module named 'AIE'`。
   在 `from_pretrained()` 開頭檢查並給出明確錯誤訊息。

2. **EfficientAD .pt 也需要 AIE 套件** — pickle 裡有 `AIE.models.efficient_ad.architecture.EfficientAD`。

3. **TorchScript 版本不需要 AIE** — `torch.jit.load()` 不需要原始類別定義。

4. **decrypt_nst() 的實作跟 file_decode_v2() 等價** — 兩者邏輯相同（header=[24,97,28,98], random_num_range=87）。

5. **不要自己寫 FastAPI server 測試** — 用 `AutoModel.from_pretrained()` 直接測試。
   Server adapter 整合是之後的工作。

6. **ClassificationModel.pre_process 只回傳 PreprocessedInputs** — 不像 OD/InstanceSeg 回傳 tuple(inputs, metadata)。

7. **EfficientAD 輸入是 RGB [0,1] tensor** — 不是 BGR numpy，需要 PIL → Resize → ToTensor。

---

## 擴展指引：如何新增 backend_type

架構的擴展點是 `REGISTERED_MODELS` 的三元組 key `(model_architecture, task_type, backend_type)`。
新增一個 backend **不需要修改任何現有程式碼**，只需要：

1. **確認 `BackendType` enum 有對應值** — 在 `entities.py` 檢查，沒有就新增（例如 `TORCHSCRIPT = "torchscript"`）
2. **建立新的模型類別檔案** — 例如 `aie_object_detection_torchscript.py`，繼承同一個 base class（`ObjectDetectionModel`），實作不同的載入和推論邏輯
3. **在 `REGISTERED_MODELS` 新增一行** — `("aie", OBJECT_DETECTION_TASK, BackendType.TORCHSCRIPT): LazyClass(...)`
4. **使用者的 `model_config.json` 指定新 backend** — `"backend_type": "torchscript"`

範例：為 object-detection 新增 TorchScript backend
```python
# models_registry.py — 只加一行
("aie", OBJECT_DETECTION_TASK, BackendType.TORCHSCRIPT): LazyClass(
    module_name="inference_models.models.aie.aie_object_detection_torchscript",
    class_name="AIEForObjectDetectionTorchScript",
),
```

**禁止做法**：
- 不要在現有模型類別的 `from_pretrained()` 裡加 if/else 分支來處理新 backend
- 不要繞過 `REGISTERED_MODELS` 自己寫路由邏輯
- 不要修改 `AutoModel` 的路由機制

`AutoModel.from_pretrained()` 讀取 `model_config.json` → 查 `REGISTERED_MODELS` → 自動路由，這條路徑對所有 backend 通用。

---

## Model Package 目錄結構（使用者需準備）

```
model_dir/
├── model_config.json    # AutoModel 路由用
└── weights.pt           # 或 weights.nst
```

`model_config.json` 範例：
```json
// YOLO Object Detection
{"model_architecture": "aie", "task_type": "object-detection", "backend_type": "ultralytics"}

// YOLO Classification
{"model_architecture": "aie", "task_type": "classification", "backend_type": "ultralytics"}

// YOLO Instance Segmentation
{"model_architecture": "aie", "task_type": "instance-segmentation", "backend_type": "ultralytics"}

// EfficientAD Anomaly Detection
{"model_architecture": "aie", "task_type": "anomaly-detection", "backend_type": "torch"}
```

---

## 驗證步驟

### 1. 前置條件
```bash
pip install -e D:\Project\TrainingTool\AIE
```

### 2. YOLO Object Detection（.pt）
```python
# 準備 model_config.json + weights.pt 在 temp_test/locate_model/
from inference_models import AutoModel
model = AutoModel.from_pretrained("temp_test/locate_model", allow_direct_local_storage_loading=True)
assert type(model).__name__ == "AIEForObjectDetection"
assert model.class_names == ["Object"]
results = model(cv2.imread("test.jpg"))  # → List[Detections]
```

### 3. EfficientAD（.nst）
```python
# 準備 model_config.json + weights.nst（從 task3_2026/best.nst 複製）
model = AutoModel.from_pretrained("temp_test/ad_model", allow_direct_local_storage_loading=True)
assert type(model).__name__ == "AIEForAnomalyDetection"
results = model(image)  # → List[AnomalyDetectionResult]
# results[0].anomaly_map.shape → (1, 256, 256)
# results[0].anomaly_score → scalar tensor
```

### 4. YOLO .nst（加密的 YOLO .pt）
```python
# 如果有加密的 YOLO .nst 檔案，也要能正確載入
```

---

## 可複用的現有資源

| 資源 | 位置 | 用途 |
|------|------|------|
| `ObjectDetectionModel` | `D:\Project\inferserver\inference\inference_models\inference_models\models\base\object_detection.py` | OD base class + `Detections` dataclass |
| `ClassificationModel` | `D:\Project\inferserver\inference\inference_models\inference_models\models\base\classification.py` | CLS base class + `ClassificationPrediction` |
| `InstanceSegmentationModel` | `D:\Project\inferserver\inference\inference_models\inference_models\models\base\instance_segmentation.py` | Seg base class + `InstanceDetections` |
| `BackendType.ULTRALYTICS` | `D:\Project\inferserver\inference\inference_models\inference_models\models\auto_loaders\entities.py:31` | 已存在的 enum 值 |
| `LazyClass` | `D:\Project\inferserver\inference\inference_models\inference_models\utils\imports.py` | Registry 延遲載入機制 |
| `REGISTERED_MODELS` | `D:\Project\inferserver\inference\inference_models\inference_models\models\auto_loaders\models_registry.py` | 模型註冊表 |
| `decrypt_nst()` | `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\decrypt.py` | .nst 解密（已實作且正確） |
| `PreProcessingMetadata` | `D:\Project\inferserver\inference\inference_models\inference_models\models\common\roboflow\model_packages.py:45-61` | namedtuple，adapter 存取用 |
| Adapter 層 | `D:\Project\inferserver\inference\inference\core\models\inference_models_adapters.py` | 之後 server 整合用，本階段不動 |

---

## 不動的檔案
- `D:\Project\inferserver\inference\inference\core\models\inference_models_adapters.py` — adapter 整合之後再做
- `D:\Project\inferserver\inference\inference\models\utils.py` (ROBOFLOW_MODEL_TYPES) — 之後再做
- `D:\Project\inferserver\inference\inference_models\inference_models\models\auto_loaders\entities.py` 的 `BackendType` enum — `ULTRALYTICS` 和 `TORCH` 都已存在

---

## 階段規劃
- **Phase 1**：完成 inference_models 的 AutoModel 層，4 個模型類別 + 1 個 base class
- **Phase 2**：Server adapter 整合 — 讓 inference server 能載入和推論 AIE 模型
- **Phase 3**：Workflow 整合 — AD built-in block + 二次開發指引
- **Phase 4**：打包與部署 — whl 分發、container、plugin 自動發現

---

# Phase 2：Server Adapter 整合

## 背景：模型怎麼從磁碟到 HTTP API

inference server 的模型載入流程：
```
HTTP POST /model/add {"model_id": "xxx"}
  → ModelManager.add_model(model_id)
    → model_registry.get_model(model_id)  → 回傳模型類別（不是實例）
    → model = model_class(model_id, api_key)  → 實例化，載入權重
    → self._models[model_id] = model  → 存入記憶體

HTTP POST /infer/xxx
  → model_manager.infer_from_request_sync(model_id, request)
    → model.infer_from_request(request)
      → preprocess → predict → postprocess → InferenceResponse
```

## 啟動方式：跟 SAM 走相同路徑

### SAM 的載入路徑（參考）

SAM 不需要 Roboflow API，走的是 GENERIC_MODELS 路徑：

```
model_id = "sam"
  → get_model_type("sam")
    → GENERIC_MODELS["sam"] = ("embed", "sam")     ← 在 roboflow.py:49-71 直接查表，跳過 API
  → ROBOFLOW_MODEL_TYPES[("embed", "sam")]
    → InferenceModelsSAMAdapter                     ← 在 utils.py:827-835 註冊
  → adapter.__init__("sam")
    → AutoModel.from_pretrained("sam")              ← 在 adapter 內部呼叫
```

關鍵檔案：
- `D:\Project\inferserver\inference\inference\core\registries\roboflow.py:49-71` — `GENERIC_MODELS` 字典
- `D:\Project\inferserver\inference\inference\core\registries\roboflow.py:182-189` — `get_model_type()` 先查 GENERIC_MODELS
- `D:\Project\inferserver\inference\inference\models\utils.py:827-835` — SAM adapter 註冊
- `D:\Project\inferserver\inference\inference\models\sam\segment_anything_inference_models.py:76` — `AutoModel.from_pretrained()`

### AIE 走相同路徑

AIE 跟 SAM 一樣是 local 模型，但 AIE 有 4 種 task type（SAM 只有 1 種）。
model_id 直接用本地路徑（`AutoModel.from_pretrained` 已支援路徑偵測）。

AIE 不能用 GENERIC_MODELS（hardcoded key），因為 model_id 是動態的本地路徑。
解法：繼承 `RoboflowModelRegistry`，在 `get_model()` 中攔截本地路徑。

**Step 1：建立 AIEModelRegistry**

**檔案**: `D:\Project\inferserver\inference\inference\core\registries\aie.py`（新增）

繼承 `RoboflowModelRegistry`，覆寫 `get_model()`，在最前面偵測本地路徑：

```python
import json
import os
from typing import Optional

from inference.core.exceptions import ModelNotRecognisedError
from inference.core.models.base import Model
from inference.core.registries.roboflow import ModelEndpointType, RoboflowModelRegistry


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
            model_id, api_key,
            countinference=countinference,
            service_secret=service_secret,
        )
```

繼承關係：`ModelRegistry → RoboflowModelRegistry → AIEModelRegistry`
- 不修改 `roboflow.py`，不動 `get_model_type()`
- 本地路徑在 `get_model()` 最前面攔截，沒命中就 `super()` 走原本 Roboflow 流程

**Step 2：修改 cpu_http.py 和 gpu_http.py 使用 AIEModelRegistry**

兩個檔案結構幾乎一樣，都需要改同一行：

```python
# D:\Project\inferserver\inference\docker\config\cpu_http.py
# D:\Project\inferserver\inference\docker\config\gpu_http.py
# 原本
from inference.core.registries.roboflow import RoboflowModelRegistry
model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
# 改為
from inference.core.registries.aie import AIEModelRegistry
model_registry = AIEModelRegistry(ROBOFLOW_MODEL_TYPES)
```

**Step 3：在 ROBOFLOW_MODEL_TYPES 註冊 AIE adapter**

**背景：兩套模型系統**

inference server 有新舊兩套模型系統：

- **舊系統**（`inference/models/`）— 每個架構各寫一個類別（如 `YOLOv8ObjectDetection`），
  直接綁 ONNX runtime 和 Roboflow API，自己實作 preprocess/predict/postprocess。
- **新系統**（`inference_models/`）— 獨立套件，透過 `AutoModel.from_pretrained()` + `model_config.json` 路由，
  後端不限（ultralytics、torch 等），不綁 Roboflow API。
- **Adapter 層**（`inference/core/models/inference_models_adapters.py`）— 把新系統的模型包裝成舊系統的介面，
  讓 `ModelManager` 可以管理新系統的模型。

`utils.py` 中的 `ROBOFLOW_MODEL_TYPES` 字典是 server 的模型路由表。
當 `USE_INFERENCE_MODELS=True`（**預設值，不需要手動設定**）時，L:717 的 `if USE_INFERENCE_MODELS:` 區塊
會把舊系統的模型類別替換成 adapter，讓 server 使用新系統。

**AIE 模型只存在於新系統**，不需要在 `inference/models/` 寫任何舊系統的類別。
只需要在 `if USE_INFERENCE_MODELS:` 區塊中註冊 adapter：

```python
# D:\Project\inferserver\inference\inference\models\utils.py
# 在 if USE_INFERENCE_MODELS: 區塊內新增（約 L:850 之後）
ROBOFLOW_MODEL_TYPES[("object-detection", "aie")] = InferenceModelsObjectDetectionAdapter
ROBOFLOW_MODEL_TYPES[("classification", "aie")] = InferenceModelsClassificationAdapter
ROBOFLOW_MODEL_TYPES[("instance-segmentation", "aie")] = InferenceModelsInstanceSegmentationAdapter
ROBOFLOW_MODEL_TYPES[("anomaly-detection", "aie")] = InferenceModelsAnomalyDetectionAdapter
```

**環境變數需求**：
- `USE_INFERENCE_MODELS=True` — 預設已啟用，不需要設定
- `ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True` — **必須手動設定**，允許本地路徑載入

### 完整的 AIE 載入路徑

```
# 設定環境變數
ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True

# model_id = 本地路徑
model_id = "D:/models/locate_model"

HTTP POST /model/add {"model_id": "D:/models/locate_model"}
  → AIEModelRegistry.get_model("D:/models/locate_model")
    → os.path.isdir() → True
    → 讀取 model_config.json → {"task_type": "object-detection", "model_architecture": "aie"}
    → registry_dict[("object-detection", "aie")]
    → InferenceModelsObjectDetectionAdapter
  → adapter.__init__("D:/models/locate_model")
    → AutoModel.from_pretrained("D:/models/locate_model")
      → 讀取 model_config.json → 路由到 AIEForObjectDetection
      → 載入 weights.pt / weights.nst
  → 模型就緒，可推論
```

用標準的 `cpu_http.py` 啟動（只需改一行 import）。

## Adapter 實作

### 現有 Adapter（OD / Classification / InstanceSeg）

現有 adapter **已經可以直接用**，不需要新建：
- `InferenceModelsObjectDetectionAdapter`（`inference_models_adapters.py:90`）— 呼叫 `AutoModel.from_pretrained` → 路由到 `AIEForObjectDetection`
- `InferenceModelsClassificationAdapter`（`inference_models_adapters.py:609`）→ `AIEForClassification`
- `InferenceModelsInstanceSegmentationAdapter`（`inference_models_adapters.py:241`）→ `AIEForInstanceSegmentation`

### 新建 Adapter：AnomalyDetection

**檔案**: `D:\Project\inferserver\inference\inference\core\models\inference_models_adapters.py`（在現有檔案中新增）

目前不存在 AnomalyDetection adapter，需要新建。結構參考 `InferenceModelsClassificationAdapter`（L:609）：

```python
class InferenceModelsAnomalyDetectionAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        self._model = AutoModel.from_pretrained(model_id, ...)
        self.class_names = list(self._model.class_names)  # ["good", "anomaly"]

    def preprocess(self, image, **kwargs):
        return self._model.pre_process(images, **kwargs)

    def predict(self, img_in, **kwargs):
        return self._model.forward(img_in, **kwargs)

    def postprocess(self, predictions, preprocess_return_metadata, **kwargs):
        results = self._model.post_process(predictions, **kwargs)
        responses = []
        for r in results:
            responses.append(AnomalyDetectionInferenceResponse(
                anomaly_score=float(r.anomaly_score),
                anomaly_map=r.anomaly_map.cpu().numpy().tolist(),
            ))
        return responses
```

**需要新增的 Response 類別**（如果尚不存在）：
- `AnomalyDetectionInferenceResponse` — 在 `D:\Project\inferserver\inference\inference\core\entities\responses\inference.py` 中新增

## Phase 2 檔案清單

### 新增 (1 個)
```
D:\Project\inferserver\inference\inference\core\registries\aie.py                      # AIEModelRegistry（繼承 RoboflowModelRegistry）
```

### 修改 (4 個)
```
D:\Project\inferserver\inference\docker\config\cpu_http.py                             # import AIEModelRegistry 替換 RoboflowModelRegistry（一行）
D:\Project\inferserver\inference\docker\config\gpu_http.py                             # 同上（一行）
D:\Project\inferserver\inference\inference\core\models\inference_models_adapters.py    # 新增 InferenceModelsAnomalyDetectionAdapter
D:\Project\inferserver\inference\inference\models\utils.py                             # ROBOFLOW_MODEL_TYPES 加入 4 個 AIE 條目
```

### 修改 (1 個)
```
D:\Project\inferserver\inference\inference\core\entities\responses\inference.py        # 新增 AnomalyDetectionInferenceResponse
```

---

# Phase 3：Workflow 整合（僅 Anomaly Detection）

## OD/CLS/InstanceSeg 不需要自訂 block

Phase 2 完成後，OD/CLS/InstanceSeg 的 AIE 模型已可直接用 Roboflow 現有的 workflow model block。
現有的 `RoboflowObjectDetectionModelBlockV2`（`core_steps/models/roboflow/object_detection/v2.py`）
內部透過 `model_manager.add_model()` + `model_manager.infer_from_request_sync()` 呼叫模型，
只要 `model_id` 填本地路徑即可：

```json
{
  "type": "roboflow_core/roboflow_object_detection_model@v2",
  "name": "locate_step",
  "model_id": "D:/models/locate_model",
  "image": "$inputs.image"
}
```

`model_manager.add_model("D:/models/locate_model")` → `AIEModelRegistry` 攔截 → adapter → 推論。
Classification 和 Instance Segmentation 同理，分別使用現有的 classification/instance_segmentation block。

## Anomaly Detection 需要新增 built-in block

Roboflow 沒有 anomaly detection task，所以沒有對應的 workflow model block。
需要在 `core_steps/models/` 中新增 AD 專用的 **built-in block**，
與 `RoboflowObjectDetectionModelBlockV2` 同級，作為 server 基礎提供的模型。

### 為什麼是 built-in 而不是 plugin

AD block 對 inference server 而言是基礎能力（如同 OD/CLS/InstanceSeg），
不應要求使用者額外設定 `WORKFLOWS_PLUGINS` 才能使用。

Built-in block 的優勢：
- 不需要手動 `model_manager` 注入 — `loader.py` 的 `REGISTERED_INITIALIZERS`（L:592-601）自動提供
- 不需要在啟動腳本加任何程式碼
- 使用者只需要有模型目錄 + `model_config.json` 即可

### AD Block 實作

**檔案**: `D:\Project\inferserver\inference\inference\core\workflows\core_steps\models\aie\anomaly_detection\v1.py`（新增）

目錄結構（與 `models/roboflow/`、`models/foundation/`、`models/third_party/` 平級）：
```
core_steps/models/
├── aie/                            # ← 新增
│   ├── __init__.py
│   └── anomaly_detection/
│       ├── __init__.py
│       └── v1.py                   # AIEAnomalyDetectionBlockV1
├── foundation/
├── roboflow/
└── third_party/
```

```python
class AIEAnomalyDetectionBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "AIE Anomaly Detection",
            "version": "v1",
            "short_description": "Run anomaly detection on an AIE EfficientAD model.",
            "block_type": "model",
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/aie_anomaly_detection_model@v1"]
    model_id: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        description="Local path or runtime parameter for AIE anomaly detection model",
        examples=["D:/models/ad_model", "$inputs.model_id"],
    )
    image: Selector(kind=[IMAGE_KIND]) = Field(description="Input image")

class AIEAnomalyDetectionBlockV1(WorkflowBlock):
    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return AIEAnomalyDetectionBlockManifest

    def run(self, model_id: str, image: WorkflowImageData) -> BlockResult:
        request = AnomalyDetectionInferenceRequest(
            model_id=model_id,
            image=image.to_inference_format(numpy_preferred=True),
        )
        self._model_manager.add_model(
            model_id=model_id, api_key=self._api_key or "local"
        )
        response = self._model_manager.infer_from_request_sync(
            model_id=model_id, request=request
        )
        return {
            "anomaly_score": response.anomaly_score,
            "anomaly_map": response.anomaly_map,
        }
```

參考：`RoboflowObjectDetectionModelBlockV2`（`v2.py:171-220`）的 `get_init_parameters()` 和 `run_locally()`。

### 註冊到 loader.py

**檔案**: `D:\Project\inferserver\inference\inference\core\workflows\core_steps\loader.py`（修改）

```python
# 在 import 區塊加入（約 L:340 後）
from inference.core.workflows.core_steps.models.aie.anomaly_detection.v1 import (
    AIEAnomalyDetectionBlockV1,
)

# 在 load_blocks() 的 blocks list 中加入（約 L:700 後）
AIEAnomalyDetectionBlockV1,
```

`model_manager`、`api_key`、`step_execution_mode` 透過 `REGISTERED_INITIALIZERS`（L:592-601）自動注入，
不需要在啟動腳本中手動注入。

### 架構邊界：模型 vs 業務邏輯

**模型（Model）** — AIE 產出的純推論結果，與具體任務無關：
- AD block 輸出 `anomaly_map`（IMAGE_KIND, uint8 灰度圖）+ `anomaly_score`（FLOAT_KIND）
- 這就是 EfficientAD `forward()` 的直接結果，不含任何業務邏輯
- 屬於本計畫範圍（`core_steps/models/aie/anomaly_detection/v1.py`）

**業務邏輯（Business Logic）** — 如何使用模型結果解決具體問題：
- 前景過濾、二值化參數、輪廓面積閾值、多光源 OR 合併策略...
- 隨任務而變，**必須與模型分離，單獨管理**
- **不在本計畫範圍**，由使用者以外部 plugin 或應用層程式碼管理
- 可放在獨立的 plugin 中（如 `business_plugin`），與 AD built-in block 分開

### AD block 輸出格式

AD block 的 `anomaly_map` 必須輸出為 `IMAGE_KIND`（uint8 灰度圖，值域 0-255），
而不是 raw tensor。這樣才能與 Roboflow 現有 CV block 直接串接。

```python
# AD block describe_outputs
@classmethod
def describe_outputs(cls) -> List[OutputDefinition]:
    return [
        OutputDefinition(name="anomaly_map", kind=[IMAGE_KIND]),    # uint8 grayscale 256x256
        OutputDefinition(name="anomaly_score", kind=[FLOAT_KIND]),  # image-level score
    ]
```

### 業務邏輯 Block 介面定義（外部 plugin，不在本 repo）

以下 block 用於將 AD 模型的 raw 輸出轉化為實際業務結果。
它們屬於業務邏輯，**由使用者在獨立的 plugin 中實作與管理**，
不放在 AD built-in block 中，也不放在 inference server repo 中。

此處僅定義介面，讓 workflow 範例可以完整展示端到端流程。

#### `business_plugin/foreground_mask@v1`

對應 `evaluate_from_manifest.py:131-135` 的邏輯：
```python
filtered_bgr  = filter_heatmap_by_foreground(orig_bgr, heatmap_bgr)
filtered_256  = cv2.resize(filtered_bgr, (SIZE, SIZE))
fg_mask       = (cv2.cvtColor(filtered_256, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
anomaly_masked = cv2.bitwise_and(anomaly_map_np, anomaly_map_np, mask=fg_mask)
```

封裝為一個 block：
- **輸入**：`image`（IMAGE_KIND, 原圖）、`anomaly_map`（IMAGE_KIND, AD 輸出的灰度 heatmap）
- **輸出**：`image`（IMAGE_KIND, 遮罩後的 anomaly map，背景區域歸零）
- **作用**：從原圖提取前景區域 → 生成 mask → 套用到 anomaly_map 上

Roboflow 沒有 bitwise AND block，且前景提取邏輯是業務特定的，
所以必須封裝成 custom block。

### Workflow JSON 範例

以下範例對應 `evaluate_from_manifest.py` 的 `infer_single()` 完整管線。
資料流透過 **selector 引用**（`$steps.xxx.yyy`）串接，engine 自動分析依賴圖決定執行順序。

**參數化**：`model_id`、`thresh_value` 等參數用 `$inputs.xxx` 引用，
推論時才填入實際值，讓 workflow JSON 成為可重用的範本。

每個 step 標註來源：
- `[MODEL]` — built-in block（本計畫實作，`core_steps/models/aie/`）
- `[ROBOFLOW]` — Roboflow 現有 block（無需開發）
- `[BUSINESS]` — 業務邏輯 plugin（使用者自行實作與管理）

#### 範例 1：Contour-level（單張影像 → N 個異常輪廓）

對應 `infer_single()` L:113-156 的完整流程。

```
執行順序（engine 根據 selector 依賴自動決定）：
  距離 1: ad_step         ← 只依賴 $inputs.image                [MODEL]
  距離 2: fg_mask_step    ← 依賴 $inputs.image + ad_step 輸出    [BUSINESS]
  距離 3: morph_step      ← 依賴 fg_mask_step 輸出               [ROBOFLOW]
  距離 4: threshold_step  ← 依賴 morph_step 輸出                 [ROBOFLOW]
  距離 5: contour_step    ← 依賴 threshold_step 輸出             [ROBOFLOW]
```

```json
{
  "version": "1.0",
  "inputs": [
    {"type": "WorkflowImage", "name": "image"},
    {"type": "WorkflowParameter", "name": "ad_model_id"},
    {"type": "WorkflowParameter", "name": "bin_threshold", "default_value": 100}
  ],
  "steps": [
    {
      "type": "roboflow_core/aie_anomaly_detection_model@v1",
      "name": "ad_step",
      "model_id": "$inputs.ad_model_id",
      "image": "$inputs.image"
    },
    {
      "type": "business_plugin/foreground_mask@v1",
      "name": "fg_mask_step",
      "image": "$inputs.image",
      "anomaly_map": "$steps.ad_step.anomaly_map"
    },
    {
      "type": "roboflow_core/morphological_transformation@v1",
      "name": "morph_step",
      "image": "$steps.fg_mask_step.image",
      "operation": "Opening",
      "kernel_size": 3
    },
    {
      "type": "roboflow_core/threshold@v1",
      "name": "threshold_step",
      "image": "$steps.morph_step.image",
      "threshold_type": "binary",
      "thresh_value": "$inputs.bin_threshold"
    },
    {
      "type": "roboflow_core/contours_detection@v1",
      "name": "contour_step",
      "image": "$steps.threshold_step.image"
    }
  ],
  "outputs": [
    {"type": "JsonField", "name": "anomaly_score",  "selector": "$steps.ad_step.anomaly_score"},
    {"type": "JsonField", "name": "contour_count",  "selector": "$steps.contour_step.number_contours"},
    {"type": "JsonField", "name": "contours",        "selector": "$steps.contour_step.contours"},
    {"type": "JsonField", "name": "contour_image",   "selector": "$steps.contour_step.image"}
  ]
}
```

**推論呼叫**：
```python
response = requests.post("http://localhost:9001/workflows/run", json={
    "specification": workflow_spec,
    "inputs": {
        "image": {"type": "base64", "value": base64_data},
        "ad_model_id": "D:/models/ad_model",
        "bin_threshold": 120
    }
})
```

**已知限制**：`contours_detection@v1` 不支援 min_area 過濾（原始碼中的 `CONTOUR_MIN_AREA=10`）。
若需要面積過濾，使用者需在 business plugin 中新增 contour filter block。

#### 範例 2：Product-level（單張影像 → OK/NG 判定）

在 contour-level 基礎上加入 `expression@v1`，
對應 `evaluate_from_manifest.py:289` 的判定邏輯：`n_contours >= CONTOUR_COUNT_TH → NG`。
`contour_count_threshold` 也參數化，推論時指定。

```json
{
  "version": "1.0",
  "inputs": [
    {"type": "WorkflowImage", "name": "image"},
    {"type": "WorkflowParameter", "name": "ad_model_id"},
    {"type": "WorkflowParameter", "name": "bin_threshold", "default_value": 100},
    {"type": "WorkflowParameter", "name": "contour_count_threshold", "default_value": 1}
  ],
  "steps": [
    {
      "type": "roboflow_core/aie_anomaly_detection_model@v1",
      "name": "ad_step",
      "model_id": "$inputs.ad_model_id",
      "image": "$inputs.image"
    },
    {
      "type": "business_plugin/foreground_mask@v1",
      "name": "fg_mask_step",
      "image": "$inputs.image",
      "anomaly_map": "$steps.ad_step.anomaly_map"
    },
    {
      "type": "roboflow_core/morphological_transformation@v1",
      "name": "morph_step",
      "image": "$steps.fg_mask_step.image",
      "operation": "Opening",
      "kernel_size": 3
    },
    {
      "type": "roboflow_core/threshold@v1",
      "name": "threshold_step",
      "image": "$steps.morph_step.image",
      "threshold_type": "binary",
      "thresh_value": "$inputs.bin_threshold"
    },
    {
      "type": "roboflow_core/contours_detection@v1",
      "name": "contour_step",
      "image": "$steps.threshold_step.image"
    },
    {
      "type": "roboflow_core/expression@v1",
      "name": "verdict_step",
      "data": {
        "contour_count": "$steps.contour_step.number_contours",
        "threshold": "$inputs.contour_count_threshold"
      },
      "expression": "contour_count >= threshold"
    }
  ],
  "outputs": [
    {"type": "JsonField", "name": "is_defective",  "selector": "$steps.verdict_step.result"},
    {"type": "JsonField", "name": "contour_count",  "selector": "$steps.contour_step.number_contours"},
    {"type": "JsonField", "name": "anomaly_score",  "selector": "$steps.ad_step.anomaly_score"}
  ]
}
```

**多光源 OR 合併**：`evaluate_from_manifest.py:322` 的 die-level OR 邏輯
（任一 LightSet 判 NG → die 判 NG）屬於應用層邏輯。
在應用層對同一 die 的每個 LightSet 影像各呼叫一次此 workflow，
再對所有 `is_defective` 結果做 `any()` 即可。

#### 範例 3：定位 → 裁切 → 異常檢測（多階段條件管線）

展示 workflow engine 的進階功能：條件分支、動態裁切、batch 維度擴展。

**場景**：先用 YOLO 定位物件，裁切出每個物件區域，再對每個裁切做異常檢測。
若定位不到物件，後續步驟不執行。

```
執行順序：
  距離 1: locate_step      ← 定位物件                            [MODEL via ROBOFLOW block]
  距離 2: crop_step        ← 裁切每個偵測到的物件                 [ROBOFLOW]
  距離 3: ad_step          ← 對每個裁切做 AD（自動 batch）        [MODEL]
  距離 4: threshold_step   ← 二值化                              [ROBOFLOW]
  距離 5: contour_step     ← 找輪廓                              [ROBOFLOW]
  距離 6: verdict_step     ← 判定 OK/NG                          [ROBOFLOW]
```

**batch 維度擴展**：`dynamic_crop@v1` 產出 N 張裁切圖（N = 偵測數量），
下游所有步驟自動對每張裁切各執行一次，output 變成 N 筆結果的陣列。

```json
{
  "version": "1.0",
  "inputs": [
    {"type": "WorkflowImage", "name": "image"},
    {"type": "WorkflowParameter", "name": "locate_model_id"},
    {"type": "WorkflowParameter", "name": "ad_model_id"},
    {"type": "WorkflowParameter", "name": "locate_confidence", "default_value": 0.25},
    {"type": "WorkflowParameter", "name": "bin_threshold", "default_value": 100}
  ],
  "steps": [
    {
      "type": "roboflow_core/roboflow_object_detection_model@v2",
      "name": "locate_step",
      "model_id": "$inputs.locate_model_id",
      "images": "$inputs.image",
      "confidence": "$inputs.locate_confidence"
    },
    {
      "type": "roboflow_core/dynamic_crop@v1",
      "name": "crop_step",
      "images": "$inputs.image",
      "predictions": "$steps.locate_step.predictions"
    },
    {
      "type": "roboflow_core/aie_anomaly_detection_model@v1",
      "name": "ad_step",
      "model_id": "$inputs.ad_model_id",
      "image": "$steps.crop_step.crops"
    },
    {
      "type": "roboflow_core/threshold@v1",
      "name": "threshold_step",
      "image": "$steps.ad_step.anomaly_map",
      "threshold_type": "binary",
      "thresh_value": "$inputs.bin_threshold"
    },
    {
      "type": "roboflow_core/contours_detection@v1",
      "name": "contour_step",
      "image": "$steps.threshold_step.image"
    },
    {
      "type": "roboflow_core/expression@v1",
      "name": "verdict_step",
      "data": {
        "contour_count": "$steps.contour_step.number_contours"
      },
      "expression": "contour_count >= 1"
    }
  ],
  "outputs": [
    {"type": "JsonField", "name": "num_objects",     "selector": "$steps.locate_step.number_of_detections"},
    {"type": "JsonField", "name": "per_object_verdict", "selector": "$steps.verdict_step.result"},
    {"type": "JsonField", "name": "per_object_contours", "selector": "$steps.contour_step.number_contours"},
    {"type": "JsonField", "name": "per_object_score",  "selector": "$steps.ad_step.anomaly_score"}
  ]
}
```

**推論呼叫**：
```python
response = requests.post("http://localhost:9001/workflows/run", json={
    "specification": workflow_spec,
    "inputs": {
        "image": {"type": "base64", "value": base64_data},
        "locate_model_id": "D:/models/locate_model",
        "ad_model_id": "D:/models/ad_model",
        "locate_confidence": 0.3,
        "bin_threshold": 120
    }
})

result = response.json()
# result["outputs"][0]["num_objects"] → 3（偵測到 3 個物件）
# result["outputs"][0]["per_object_verdict"] → [True, False, True]（每個物件的 OK/NG）
# result["outputs"][0]["per_object_score"] → [0.82, 0.15, 0.91]（每個物件的異常分數）
```

**無偵測的情況**：若 `locate_step` 偵測到 0 個物件，`dynamic_crop@v1` 產出空陣列，
下游步驟不執行，output 中的 per_object 欄位為空陣列 `[]`。
應用層檢查 `num_objects == 0` 即可判斷。

### Workflow 可用的 Roboflow 現有 Block（部分清單）

以下是與 AIE 業務邏輯相關的現有 block，完整清單見 `core_steps/` 目錄。

| Block type | 功能 | 關鍵參數 |
|------------|------|----------|
| `roboflow_core/roboflow_object_detection_model@v2` | 物件偵測 | model_id, confidence, iou_threshold |
| `roboflow_core/dynamic_crop@v1` | 按偵測框裁切 | images, predictions |
| `roboflow_core/threshold@v1` | 二值化（8 種方法）| thresh_value, threshold_type |
| `roboflow_core/morphological_transformation@v1` | 形態學運算 | operation, kernel_size |
| `roboflow_core/contours_detection@v1` | 找輪廓 | image → contours, number_contours |
| `roboflow_core/expression@v1` | 條件判定 / switch-case | data, expression |
| `roboflow_core/continue_if@v1` | 條件流程控制 | condition, next_steps |
| `roboflow_core/detections_filter@v1` | 過濾偵測結果 | predictions, operations |
| `roboflow_core/image_preprocessing@v1` | 影像前處理（resize 等）| width, height |
| `roboflow_core/convert_grayscale@v1` | 轉灰度 | image |

所有 block 的參數幾乎都支援 `$inputs.xxx` 動態參數。

### 啟動環境變數

```bash
ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True
WORKFLOWS_PLUGINS=business_plugin    # 僅需要業務邏輯 plugin，AD block 已內建
```

---

## 二次開發指引：建立業務邏輯 Plugin

本節說明如何按照 Roboflow 官方規範建立業務邏輯 plugin。
Plugin 是獨立的 Python package，安裝到 server 的 Python 環境後由 server 載入。
**Plugin 不包含 server 啟動邏輯** — server 的部署設定見 Phase 4。

### Plugin 專案結構

對齊 Roboflow 官方 `blocks_bundling.md` 的推薦結構：

```
business_plugin/                         # ← Plugin 專案根目錄
├── pyproject.toml                       # Python package 配置
├── business_plugin/                     # Plugin package
│   ├── __init__.py                      #   load_blocks() + load_kinds()
│   ├── foreground_mask/                 #   Block 1
│   │   └── v1.py                        #     ForegroundMaskBlockV1
│   └── contour_filter/                  #   Block 2（可選）
│       └── v1.py                        #     ContourFilterBlockV1
└── tests/                               # 測試
```

模型權重、workflow JSON、應用層程式碼**不放在 plugin 專案內**，各自獨立管理。

### `pyproject.toml`

```toml
[project]
name = "business-plugin"
version = "0.1.0"
dependencies = ["opencv-python", "numpy"]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.backends._legacy:_Backend"
```

```bash
# 開發時安裝
pip install -e /path/to/business_plugin

# 正式部署
pip install business-plugin  # 從 PyPI 或私有倉庫
```

### `__init__.py` — Plugin 入口

```python
from business_plugin.foreground_mask.v1 import ForegroundMaskBlockV1

def load_blocks():
    """Server 啟動時呼叫，回傳所有 block 類別。"""
    return [ForegroundMaskBlockV1]
```

可選的進階介面（參考 `blocks_bundling.md`）：
- `load_kinds()` → 自訂 Kind 類型
- `REGISTERED_INITIALIZERS` → block 初始化參數
- `KINDS_SERIALIZERS` / `KINDS_DESERIALIZERS` → 自訂序列化

### Block 模組

每個 block 用獨立目錄 + 版本檔案管理，與 Roboflow 內建 block 結構一致：

```python
# business_plugin/foreground_mask/v1.py
class ForegroundMaskBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Foreground Mask",
            "version": "v1",
            "short_description": "Filter anomaly map by foreground region.",
            "block_type": "transformation",
        },
        protected_namespaces=(),
    )
    type: Literal["business_plugin/foreground_mask@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(...)
    anomaly_map: Selector(kind=[IMAGE_KIND]) = Field(...)

class ForegroundMaskBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ForegroundMaskBlockManifest

    def run(self, image, anomaly_map) -> BlockResult:
        # 業務邏輯：前景提取 → mask → bitwise AND
        ...
```

Block type 命名慣例：`{plugin_package_name}/{block_name}@v{version}`

### Plugin 載入機制

Server 用環境變數 `WORKFLOWS_PLUGINS` 指定要載入的 plugin：

```bash
# 單個 plugin
WORKFLOWS_PLUGINS=business_plugin

# 多個 plugin（逗號分隔）
WORKFLOWS_PLUGINS=business_plugin,another_plugin,sportvision
```

載入流程（`blocks_loader.py:254-312`）：
1. 讀取 `WORKFLOWS_PLUGINS` 環境變數
2. 對每個 plugin 名稱呼叫 `importlib.import_module()`
3. 呼叫 plugin 的 `load_blocks()` 取得 block 類別列表
4. 合併到可用 block 池中

**前提**：plugin 必須已安裝到 server 的 Python 環境（`pip install`）。
AD model block 已內建於 server（`core_steps/models/aie/`），不需要透過 `WORKFLOWS_PLUGINS` 載入。

> **未來增強**：目前 Roboflow 沒有 plugin 自動發現機制。
> 可考慮用 Python 標準的 `entry_points` 機制（pytest、flask 等框架皆採用），
> 在 plugin 的 `pyproject.toml` 中宣告 `[project.entry-points."inference.plugins"]`，
> 並修改 `blocks_loader.py` 的 `get_plugin_modules()` 掃描 `importlib.metadata.entry_points()`。
> 這樣 `pip install` 後 server 即可自動發現，不需手動設 `WORKFLOWS_PLUGINS`。
> 此功能待本計畫完成後再評估是否實作。

### Workflow JSON

應用層程式碼從本地 JSON 檔案讀取 workflow 定義，透過 HTTP POST 送出：

```python
import json, requests

with open("workflows/contour_level.json") as f:
    workflow_spec = json.load(f)

response = requests.post("http://localhost:9001/workflows/run", json={
    "specification": workflow_spec,
    "inputs": {
        "image": {"type": "base64", "value": base64_data},
        "ad_model_id": "D:/models/ad_model",
    }
})
```

Workflow JSON 放在應用層專案中，不屬於 plugin，不屬於 server。
每次 POST 都帶完整 specification，修改 JSON 不需要重啟 server。

### 熱重載

Plugin 在 server 啟動時載入一次（`@lru_cache()`），修改 plugin 程式碼後**必須重啟 server**。

| 變更類型 | 需要重啟？ |
|----------|-----------|
| 修改 plugin 程式碼 | **是** |
| 修改 workflow JSON | 否（每次 POST 帶完整 spec）|
| 新增/刪除模型目錄 | 否（`add_model()` 動態載入）|
| 修改 `model_config.json` | **是**（模型載入時才讀取）|

---

## Phase 3 檔案清單

### 新增 (3 個)
```
D:\Project\inferserver\inference\inference\core\workflows\core_steps\models\aie\__init__.py                    # 空檔
D:\Project\inferserver\inference\inference\core\workflows\core_steps\models\aie\anomaly_detection\__init__.py   # 空檔
D:\Project\inferserver\inference\inference\core\workflows\core_steps\models\aie\anomaly_detection\v1.py        # AIEAnomalyDetectionBlockV1
```

### 修改 (1 個)
```
D:\Project\inferserver\inference\inference\core\workflows\core_steps\loader.py   # import + 註冊 AIEAnomalyDetectionBlockV1
```

---

## 全部四個 Phase 的完整檔案清單

### Phase 1：inference_models AutoModel 層
| 動作 | 檔案 |
|------|------|
| 新增 | `D:\Project\inferserver\inference\inference_models\inference_models\models\base\anomaly_detection.py` |
| 新增 | `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\_aie_ultralytics_base.py` |
| 新增 | `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_object_detection.py` |
| 新增 | `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_classification.py` |
| 新增 | `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_anomaly_detection.py` |
| 修改 | `D:\Project\inferserver\inference\inference_models\inference_models\models\auto_loaders\entities.py` |
| 修改 | `D:\Project\inferserver\inference\inference_models\inference_models\models\auto_loaders\models_registry.py` |
| 刪除 | `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_object_detection_ultralytics.py` |
| 刪除 | `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_classification_ultralytics.py` |
| 刪除 | `D:\Project\inferserver\inference\inference_models\inference_models\models\aie\aie_instance_segmentation_ultralytics.py` |

### Phase 2：Server Adapter 整合
| 動作 | 檔案 |
|------|------|
| 新增 | `D:\Project\inferserver\inference\inference\core\registries\aie.py` — AIEModelRegistry（繼承 RoboflowModelRegistry） |
| 修改 | `D:\Project\inferserver\inference\docker\config\cpu_http.py` — import AIEModelRegistry 替換 RoboflowModelRegistry（一行） |
| 修改 | `D:\Project\inferserver\inference\docker\config\gpu_http.py` — 同上（一行） |
| 修改 | `D:\Project\inferserver\inference\inference\core\models\inference_models_adapters.py` — 新增 AnomalyDetection adapter |
| 修改 | `D:\Project\inferserver\inference\inference\models\utils.py` — ROBOFLOW_MODEL_TYPES 加入 4 個 AIE 條目 |
| 修改 | `D:\Project\inferserver\inference\inference\core\entities\responses\inference.py` — 新增 AnomalyDetectionInferenceResponse |

### Phase 3：Workflow 整合（僅 Anomaly Detection）
| 動作 | 檔案 |
|------|------|
| 新增 | `inference\core\workflows\core_steps\models\aie\__init__.py` — 空檔 |
| 新增 | `inference\core\workflows\core_steps\models\aie\anomaly_detection\__init__.py` — 空檔 |
| 新增 | `inference\core\workflows\core_steps\models\aie\anomaly_detection\v1.py` — AIEAnomalyDetectionBlockV1 |
| 修改 | `inference\core\workflows\core_steps\loader.py` — import + 註冊 AD block |

### Phase 4：打包與部署
| 動作 | 檔案 |
|------|------|
| 修改 | `docker\dockerfiles\Dockerfile.onnx.cpu` — 加入 AIE 本地 whl 安裝步驟 |
| 修改 | `docker\dockerfiles\Dockerfile.onnx.gpu` — 同上 |

---

# Phase 4：打包與部署

## 套件邊界

AIE 整合的程式碼分佈在兩個現有套件中，不需要建立新套件：

```
inference_models (whl)
├── models/aie/                          # Phase 1：AIE 模型類別
│   ├── _aie_ultralytics_base.py         #   YOLO 共用載入邏輯
│   ├── aie_object_detection.py          #   AIEForObjectDetection
│   ├── aie_classification.py            #   AIEForClassification
│   ├── aie_anomaly_detection.py         #   AIEForAnomalyDetection
│   └── decrypt.py                       #   .nst 解密
├── models/base/anomaly_detection.py     #   AnomalyDetectionModel base class
└── models/auto_loaders/                 #   REGISTERED_MODELS 路由

inference-core / inference-cpu / inference-gpu (whl)
├── core/registries/aie.py               # Phase 2：AIEModelRegistry
├── core/models/inference_models_adapters.py  #   AD adapter
├── core/entities/responses/inference.py #   AD response
├── core/workflows/core_steps/
│   ├── models/aie/anomaly_detection/v1.py  # Phase 3：AD built-in block
│   └── loader.py                        #   block 註冊
├── models/utils.py                      #   ROBOFLOW_MODEL_TYPES 路由
└── docker/config/cpu_http.py            #   啟動腳本

aie (外部套件，使用者自行安裝)
└── AIE Training Toolkit                 # .nst 解密所需的命名空間
```

### 為什麼不建立新套件？

- AIE 模型類別（Phase 1）屬於 `inference_models` — 這是模型定義和推論的正確位置
- Server 整合（Phase 2+3）屬於 `inference-core` — 這是 adapter/block 的正確位置
- 兩個套件皆為本地安裝，新增 AIE 程式碼只是新增檔案，不改變既有結構

## 安裝方式

所有套件皆為本地安裝（editable install 或本地 whl），不發佈到 PyPI。

### 開發環境（editable install）

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

### 部署環境（本地 whl 分發）

非開發機器上用 whl 安裝，不需要原始碼：

```bash
# ── 在開發機上構建 whl ──
pip wheel --no-deps -w dist/ D:\Project\TrainingTool\AIE
pip wheel --no-deps -w dist/ D:\Project\inferserver\inference
pip wheel --no-deps -w dist/ D:\Project\inferserver\inference\inference_models
pip wheel --no-deps -w dist/ /path/to/business_plugin          # 可選

# 產出檔案（在 dist/ 目錄下）：
#   aie-0.24.0-py3-none-any.whl
#   inference-0.x.x-py3-none-any.whl
#   inference_models-0.x.x-py3-none-any.whl
#   business_plugin-0.1.0-py3-none-any.whl

# ── 在部署機上安裝 ──
# 將 dist/ 目錄複製到部署機後：
pip install dist/aie-*.whl
pip install dist/inference-*.whl
pip install dist/inference_models-*.whl
pip install dist/business_plugin-*.whl                          # 可選
```

### Docker container

基於 `Dockerfile.onnx.gpu` 擴展。雖然檔名含 `onnx`，但這是 Roboflow 的命名慣例，
實際上是最完整的 GPU 鏡像 — 包含 PyTorch（CUDA 12.4）、ultralytics、ONNX Runtime 等全部依賴。
AIE 的 YOLO（ultralytics backend）和 EfficientAD（純 PyTorch）所需的 runtime 都已涵蓋，
只需額外安裝 AIE toolkit（提供 .nst 解密和 pickle 命名空間）。

使用上一節構建的 whl，打包進 Docker image。

```dockerfile
FROM roboflow/roboflow-inference-server-gpu:latest

# Install AIE toolkit + inference（從本地 whl，見「部署環境」段落的構建指令）
COPY dist/*.whl /tmp/wheels/
RUN pip3 install /tmp/wheels/*.whl && rm -rf /tmp/wheels

# （可選）Install business plugin
COPY business_plugin/ /tmp/business/
RUN pip3 install /tmp/business/ && rm -rf /tmp/business

ENV WORKFLOWS_PLUGINS=business_plugin
ENV ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True
```

```bash
# 構建 image
docker build -t my-inference-server:latest -f docker/dockerfiles/Dockerfile.onnx.gpu .

# 啟動（需要 NVIDIA Container Toolkit）
# -v 掛載 host 上的模型目錄，訓練工具產出的模型放到此目錄即可被 server 載入
docker run --gpus all -p 9001:9001 \
  -v /srv/models:/models \
  -e MAX_ACTIVE_MODELS=3 \
  my-inference-server:latest
```

## Server 部署設定

Server 的啟動和 plugin 載入是**部署層面**的設定，不屬於任何 plugin 專案。

### 環境變數

| 變數 | 說明 |
|------|------|
| `WORKFLOWS_PLUGINS` | 要載入的 plugin，逗號分隔（如 `business_plugin,sportvision`）|
| `ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES` | 設為 `True` 允許本地路徑載入模型 |
| `PORT` | HTTP API port（預設 9001）|
| `MAX_ACTIVE_MODELS` | LRU cache 大小（預設 1）|

### 本地啟動

```bash
# 前提：plugin 已 pip install
set WORKFLOWS_PLUGINS=business_plugin
set ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True
python D:/Project/inferserver/inference/docker/config/cpu_http.py
```

### Docker 部署

見 Phase 4「自建 Docker image」章節。

## 模型目錄的管理

### 模型目錄結構（使用者自備）

每個模型一個目錄，包含路由設定和權重檔：

```
model_dir/
├── model_config.json    # AutoModel 路由用（必須）
└── weights.pt           # 或 weights.nst（必須）
```

模型目錄**不打包成 wheel**，而是作為資料檔案獨立管理。
`model_config.json` 是使用者必須提供的，無法自動生成。

### 載入流程

AIE 模型走本地路徑載入，不經過 Roboflow 雲端 API。
需要設定 `ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True`（預設 `False`，安全考量）。

```
訓練完成 → 產出模型目錄（weights + model_config.json）
         → 放到指定的模型目錄（本地路徑或 Docker 掛載目錄）
         → POST /model/add {"model_id": "/models/locate_model"}
         → server 載入模型，可開始推論
         → 模型更新時，重新訓練並覆蓋權重檔，再次 /model/add 即可重新載入
```

`AutoModel.from_pretrained()` 用 `os.path.exists()` 判斷 model_id 是本地路徑還是遠端 ID（`core.py:715`），
本地路徑直接從目錄讀取，不經過 `MODEL_CACHE_DIR`。

### 本地開發

模型目錄放在任意位置，`model_id` 填絕對路徑：

```python
requests.post("http://localhost:9001/model/add", json={
    "model_id": "D:/Project/models/locate_model"
})
```

### Docker 部署

在 host 上規劃一個固定的模型目錄，掛載到容器內。
訓練工具產出的模型放到此目錄，server 即可透過容器內路徑載入。

```
Host 目錄結構（範例）：
/srv/models/                          # ← host 上的模型根目錄
├── locate_model/                     #   YOLO 物件偵測
│   ├── model_config.json
│   └── weights.pt
├── particle_model/                   #   YOLO 粒子偵測
│   ├── model_config.json
│   └── weights.pt
└── ad_model/                         #   EfficientAD 異常檢測
    ├── model_config.json
    └── weights.nst
```

```bash
# 掛載到容器內的 /models
docker run --gpus all -p 9001:9001 \
  -v /srv/models:/models \
  -e ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True \
  -e MAX_ACTIVE_MODELS=3 \
  my-inference-server:latest
```

```python
# 用容器內路徑呼叫
requests.post("http://localhost:9001/model/add", json={
    "model_id": "/models/locate_model"       # 容器內路徑
})
requests.post("http://localhost:9001/model/add", json={
    "model_id": "/models/ad_model"
})
```

**模型更新（迭代訓練）**：訓練工具在 host 的 `/srv/models/` 下更新權重檔後，
對 server 重新呼叫 `/model/add` 即可載入新版本。不需要重啟容器或重建 image。

## 精簡專案：刪除測試用檔案

Phase 1-3 完成並驗證後，以下測試用檔案可以刪除：

```
D:\Project\inferserver\inference\temp_test\                   # 整個目錄
D:\Project\inferserver\inference\my_linear_model.py           # Phase 0 練習用
D:\Project\inferserver\inference\my_linear_plugin.py          # Phase 0 練習用
D:\Project\inferserver\inference\my_server.py                 # Phase 0 練習用
D:\Project\inferserver\inference\benchmark.py                 # 未完成的 benchmark
```

**注意**：刪除前確認所有功能已通過正式 test 驗證。

## Phase 4 檔案清單

### 修改 (2 個)
```
D:\Project\inferserver\inference\docker\dockerfiles\Dockerfile.onnx.cpu # AIE whl 安裝步驟
D:\Project\inferserver\inference\docker\dockerfiles\Dockerfile.onnx.gpu # 同上
```

### 刪除（清理）
```
D:\Project\inferserver\inference\temp_test\                             # 測試用目錄
D:\Project\inferserver\inference\my_linear_model.py                    # 練習用
D:\Project\inferserver\inference\my_linear_plugin.py                   # 練習用
D:\Project\inferserver\inference\my_server.py                          # 練習用
D:\Project\inferserver\inference\benchmark.py                          # 未完成
```

---

## 驗證：端到端測試流程

### Phase 1 驗證
```python
from inference_models import AutoModel
# OD
model = AutoModel.from_pretrained("temp_test/locate_model", allow_direct_local_storage_loading=True)
assert type(model).__name__ == "AIEForObjectDetection"
results = model(cv2.imread("test.jpg"))
# AD
model = AutoModel.from_pretrained("temp_test/ad_model", allow_direct_local_storage_loading=True)
assert type(model).__name__ == "AIEForAnomalyDetection"
results = model(image)
assert results[0].anomaly_map.shape == (1, 256, 256)
```

### Phase 2 驗證
```bash
set ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True
python docker/config/cpu_http.py
```
```python
# 另一個終端
import requests
# 載入模型
requests.post("http://localhost:9001/model/add", json={"model_id": "D:/models/locate_model"})
# 推論
resp = requests.post("http://localhost:9001/infer/D:/models/locate_model", ...)
```

### Phase 3 驗證
```python
# Workflow 推論（AD block）
workflow_spec = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "ad_model_id"}
    ],
    "steps": [{
        "type": "roboflow_core/aie_anomaly_detection_model@v1",
        "name": "ad_step",
        "model_id": "$inputs.ad_model_id",
        "image": "$inputs.image"
    }],
    "outputs": [
        {"type": "JsonField", "name": "score", "selector": "$steps.ad_step.anomaly_score"},
        {"type": "JsonField", "name": "map",   "selector": "$steps.ad_step.anomaly_map"}
    ]
}
resp = requests.post("http://localhost:9001/workflows/run", json={
    "specification": workflow_spec,
    "inputs": {
        "image": {"type": "base64", "value": base64_data},
        "ad_model_id": "D:/models/ad_model"
    }
})
```

### Phase 4 驗證
```bash
# Docker 構建（GPU）
docker build -t inference-aie:test -f docker/dockerfiles/Dockerfile.onnx.gpu .
# Docker 運行
docker run --gpus all -p 9001:9001 -v D:/models:/models \
  -e ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True \
  inference-aie:test
# 重複 Phase 2+3 的驗證步驟
```
