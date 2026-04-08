# Workflow 工作手冊

本手冊說明如何使用 Workflow 引擎組合 AIE 模型與影像處理步驟。

---

## 1. Workflow 概念

Workflow 是一個 JSON 定義的管線，由多個 **Step**（Block 實例）組成。
每個 Step 接收輸入、執行邏輯、產出輸出。
Step 之間透過 **selector 引用**（`$steps.xxx.yyy`）串接，引擎自動分析依賴圖決定執行順序。

```
                    +-- $inputs.image
                    |
              ┌─────▼─────┐
              │  ad_step   │  ← roboflow_core/aie_anomaly_detection_model@v1
              └──┬────┬────┘
                 │    │
    anomaly_map ─┘    └─ anomaly_score
                 │
          ┌──────▼──────┐
          │ threshold    │  ← roboflow_core/threshold@v1
          └──────┬──────┘
                 │
          ┌──────▼──────┐
          │ contours     │  ← roboflow_core/contours_detection@v1
          └─────────────┘
```

---

## 2. Workflow JSON 格式

```json
{
  "version": "1.0",
  "inputs": [
    {"type": "WorkflowImage", "name": "image"},
    {"type": "WorkflowParameter", "name": "param_name", "default_value": 100}
  ],
  "steps": [
    {
      "type": "block_type@version",
      "name": "step_name",
      "field1": "$inputs.image",
      "field2": "$steps.previous_step.output_name"
    }
  ],
  "outputs": [
    {"type": "JsonField", "name": "output_name", "selector": "$steps.step_name.field"}
  ]
}
```

**參數化**：`$inputs.xxx` 在推論時填入實際值，讓 JSON 成為可重用的範本。

---

## 3. 可用的 AIE Block

### 3.1 Anomaly Detection（內建）

| 屬性 | 值 |
|------|----|
| Block type | `roboflow_core/aie_anomaly_detection_model@v1` |
| 輸入 | `model_id`（本地路徑或 `$inputs.xxx`）、`images`（IMAGE_KIND） |
| 輸出 | `anomaly_map`（IMAGE_KIND，uint8 灰度圖 3ch）、`anomaly_score`（FLOAT_KIND） |

```json
{
  "type": "roboflow_core/aie_anomaly_detection_model@v1",
  "name": "ad_step",
  "model_id": "$inputs.ad_model_id",
  "images": "$inputs.image"
}
```

### 3.2 Object Detection（使用 Roboflow 現有 Block）

AIE YOLO 偵測模型直接使用 Roboflow 的 OD block，`model_id` 填本地路徑即可：

```json
{
  "type": "roboflow_core/roboflow_object_detection_model@v2",
  "name": "locate_step",
  "model_id": "$inputs.locate_model_id",
  "images": "$inputs.image",
  "confidence": "$inputs.confidence"
}
```

### 3.3 Classification（使用 Roboflow 現有 Block）

```json
{
  "type": "roboflow_core/roboflow_classification_model@v2",
  "name": "cls_step",
  "model_id": "$inputs.cls_model_id",
  "images": "$inputs.image"
}
```

### 3.4 Instance Segmentation（使用 Roboflow 現有 Block）

```json
{
  "type": "roboflow_core/roboflow_instance_segmentation_model@v2",
  "name": "seg_step",
  "model_id": "$inputs.seg_model_id",
  "images": "$inputs.image"
}
```

---

## 4. 常用 Roboflow 內建 Block

以下 Block 可與 AIE 模型串接，不需要額外開發：

| Block type | 功能 | 關鍵參數 |
|------------|------|----------|
| `roboflow_core/dynamic_crop@v1` | 按偵測框裁切 | `images`, `predictions` |
| `roboflow_core/threshold@v1` | 二值化（8 種方法） | `thresh_value`, `threshold_type` |
| `roboflow_core/morphological_transformation@v1` | 形態學運算 | `operation`, `kernel_size` |
| `roboflow_core/contours_detection@v1` | 找輪廓 | `image` → `contours`, `number_contours` |
| `roboflow_core/expression@v1` | 條件判定 / 運算 | `data`, `expression` |
| `roboflow_core/continue_if@v1` | 條件流程控制 | `condition`, `next_steps` |
| `roboflow_core/convert_grayscale@v1` | 轉灰度 | `image` |
| `roboflow_core/image_preprocessing@v1` | resize 等 | `width`, `height` |
| `roboflow_core/detections_filter@v1` | 過濾偵測結果 | `predictions`, `operations` |

所有 Block 的參數幾乎都支援 `$inputs.xxx` 動態參數。

---

## 5. Workflow 範例

### 範例 1：單張影像 Anomaly Detection → 輪廓偵測

```
執行順序（引擎自動決定）：
  1: ad_step         ← 只依賴 $inputs.image
  2: fg_mask_step    ← 依賴 $inputs.image + ad_step           [需要 plugin]
  3: morph_step      ← 依賴 fg_mask_step
  4: threshold_step  ← 依賴 morph_step
  5: contour_step    ← 依賴 threshold_step
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
      "images": "$inputs.image"
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
    {"type": "JsonField", "name": "anomaly_score", "selector": "$steps.ad_step.anomaly_score"},
    {"type": "JsonField", "name": "contour_count", "selector": "$steps.contour_step.number_contours"},
    {"type": "JsonField", "name": "contours", "selector": "$steps.contour_step.contours"}
  ]
}
```

> `fg_mask_step` 使用 `business_plugin/foreground_mask@v1`，需要安裝對應的 plugin。
> 若不需要前景過濾，可移除此步驟，直接將 `$steps.ad_step.anomaly_map` 接到 `threshold_step`。

### 範例 2：Product-level OK/NG 判定（多光源合併）

單一產品可能有 N 張不同光源的影像。每張影像各自做 AD 判定，
最後用 Dynamic Block 做 `any()` 聚合 — **任一光源判 NG → 產品判 NG**。

```
資料流：
  image (N 張光源影像，batch)
    │
    ├─ ad_step          ← 每張各自做 AD          → Batch[N] anomaly_score
    ├─ threshold_step   ← 每張各自二值化          → Batch[N]
    ├─ contour_step     ← 每張各自找輪廓          → Batch[N] number_contours
    ├─ verdict_step     ← 每張各自判定            → Batch[N] bool
    │
    └─ aggregate_step   ← any(N 個 bool) → 單一結果  [Dynamic Block]
```

```json
{
  "version": "1.0",
  "dynamic_blocks_definitions": [
    {
      "type": "DynamicBlockDefinition",
      "manifest": {
        "type": "ManifestDescription",
        "block_type": "AnyDefective",
        "description": "Aggregate per-light verdicts: any(verdicts) → single bool",
        "inputs": {
          "verdicts": {
            "type": "DynamicInputDefinition",
            "selector_types": ["step_output"],
            "value_types": ["boolean"]
          },
          "scores": {
            "type": "DynamicInputDefinition",
            "selector_types": ["step_output"],
            "value_types": ["float"]
          }
        },
        "outputs": {
          "is_defective": {"type": "DynamicOutputDefinition", "kind": ["boolean"]},
          "max_score": {"type": "DynamicOutputDefinition", "kind": ["float"]},
          "per_light_verdicts": {"type": "DynamicOutputDefinition", "kind": []}
        },
        "accepts_batch_input": true,
        "output_dimensionality_offset": -1
      },
      "code": {
        "type": "PythonCode",
        "run_function_code": "def run(self, verdicts, scores):\n    verdict_list = [bool(v) for v in verdicts]\n    score_list = [float(s) for s in scores]\n    return {\n        'is_defective': any(verdict_list),\n        'max_score': max(score_list) if score_list else 0.0,\n        'per_light_verdicts': verdict_list,\n    }",
        "imports": []
      }
    }
  ],
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
      "images": "$inputs.image"
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
        "contour_count": "$steps.contour_step.number_contours",
        "threshold": "$inputs.contour_count_threshold"
      },
      "expression": "contour_count >= threshold"
    },
    {
      "type": "AnyDefective",
      "name": "aggregate_step",
      "verdicts": "$steps.verdict_step.result",
      "scores": "$steps.ad_step.anomaly_score"
    }
  ],
  "outputs": [
    {"type": "JsonField", "name": "is_defective",       "selector": "$steps.aggregate_step.is_defective"},
    {"type": "JsonField", "name": "max_score",           "selector": "$steps.aggregate_step.max_score"},
    {"type": "JsonField", "name": "per_light_verdicts",  "selector": "$steps.aggregate_step.per_light_verdicts"}
  ]
}
```

**呼叫方式**：`image` 輸入傳入 N 張光源影像的陣列：

```python
response = requests.post("http://localhost:9001/workflows/run", json={
    "specification": workflow_spec,
    "inputs": {
        "image": [
            {"type": "base64", "value": light1_b64},
            {"type": "base64", "value": light2_b64},
            {"type": "base64", "value": light3_b64},
        ],
        "ad_model_id": "/models/ad_model",
        "bin_threshold": 120,
        "contour_count_threshold": 1
    }
})

result = response.json()
# result["outputs"][0]["is_defective"]       → True（任一光源判 NG）
# result["outputs"][0]["max_score"]          → 2.43（最高異常分數）
# result["outputs"][0]["per_light_verdicts"] → [False, True, False]（每個光源的判定）
```

**關鍵設計**：
- `accepts_batch_input: true` — `run()` 收到完整 Batch，而非單一元素
- `output_dimensionality_offset: -1` — 輸出維度降一級（N → 1），實現聚合
- 所有聚合邏輯在 workflow 內完成，應用層只需一次呼叫

### 範例 3：定位 → 裁切 → Anomaly Detection（多階段管線）

先用 YOLO 定位物件，裁切出每個物件區域，再對每個裁切做 AD。
`dynamic_crop@v1` 產出 N 張裁切圖，下游步驟自動對每張各執行一次。

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
      "images": "$steps.crop_step.crops"
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
    {"type": "JsonField", "name": "num_objects", "selector": "$steps.locate_step.number_of_detections"},
    {"type": "JsonField", "name": "per_object_verdict", "selector": "$steps.verdict_step.result"},
    {"type": "JsonField", "name": "per_object_contours", "selector": "$steps.contour_step.number_contours"},
    {"type": "JsonField", "name": "per_object_score", "selector": "$steps.ad_step.anomaly_score"}
  ]
}
```

**無偵測的情況**：若 `locate_step` 偵測到 0 個物件，`dynamic_crop@v1` 產出空陣列，
下游步驟不執行，`per_object_*` 欄位為空陣列 `[]`。
應用層檢查 `num_objects == 0` 即可判斷。

---

## 6. 呼叫 API

### 6.1 執行 Workflow

```python
import json
import requests

# 讀取 workflow JSON
with open("workflows/product_level.json") as f:
    workflow_spec = json.load(f)

# 準備影像（base64）
import base64
with open("test_image.jpg", "rb") as f:
    base64_data = base64.b64encode(f.read()).decode()

# 執行
response = requests.post("http://localhost:9001/workflows/run", json={
    "specification": workflow_spec,
    "inputs": {
        "image": {"type": "base64", "value": base64_data},
        "ad_model_id": "/models/ad_model",
        "bin_threshold": 120,
        "contour_count_threshold": 1
    }
})

result = response.json()
# result["outputs"][0]["is_defective"]  → True / False
# result["outputs"][0]["contour_count"] → 3
# result["outputs"][0]["anomaly_score"] → 0.82
```

### 6.2 直接呼叫模型（不用 Workflow）

```python
# 載入模型
requests.post("http://localhost:9001/model/add", json={
    "model_id": "/models/ad_model",
    "api_key": "local"
})

# 推論
response = requests.post("http://localhost:9001/infer", json={
    "model_id": "/models/ad_model",
    "image": {"type": "base64", "value": base64_data}
})

result = response.json()
# result["anomaly_score"]  → 2.43
# result["anomaly_map"]    → [[...], [...], ...]  (nested list)
```

---

## 7. 自訂邏輯的三種方式

| 方式 | 適用場景 | 需要安裝？ | 需要重啟？ |
|------|----------|-----------|-----------|
| **Dynamic Block** | 一次性邏輯、快速原型、簡單轉換 | 否 | 否 |
| **Plugin** | 可重用、跨 workflow 共享、複雜邏輯 | 是（pip install） | 是 |
| **應用層程式碼** | workflow 外的後處理（如多光源 OR） | 否 | 否 |

---

## 8. Dynamic Block（自訂 Python Block）

Dynamic Block 讓你直接在 workflow JSON 中內嵌 Python 函數，不需要安裝 plugin。
代碼隨 JSON 一起發送，server 在運行時動態編譯執行。

### 8.1 基本結構

在 workflow JSON 的頂層加入 `dynamic_blocks_definitions`，定義 block 的介面和代碼：

```json
{
  "version": "1.0",
  "dynamic_blocks_definitions": [
    {
      "type": "DynamicBlockDefinition",
      "manifest": {
        "type": "ManifestDescription",
        "block_type": "MyCustomBlock",
        "inputs": { ... },
        "outputs": { ... }
      },
      "code": {
        "type": "PythonCode",
        "run_function_code": "def run(self, ...):\n    ...",
        "imports": []
      }
    }
  ],
  "inputs": [ ... ],
  "steps": [
    {
      "type": "MyCustomBlock",
      "name": "my_step",
      ...
    }
  ],
  "outputs": [ ... ]
}
```

### 8.2 manifest 欄位

```json
"manifest": {
  "type": "ManifestDescription",
  "block_type": "ForegroundMask",
  "description": "Filter anomaly map by foreground region",
  "inputs": {
    "image": {
      "type": "DynamicInputDefinition",
      "selector_types": ["step_output"],
      "value_types": ["image"]
    },
    "anomaly_map": {
      "type": "DynamicInputDefinition",
      "selector_types": ["step_output"],
      "value_types": ["image"]
    }
  },
  "outputs": {
    "masked_image": {
      "type": "DynamicOutputDefinition",
      "kind": ["image"]
    }
  }
}
```

**inputs 的 selector_types**：
- `"step_output"` — 來自前一個 step 的輸出（`$steps.xxx.yyy`）
- `"input_parameter"` — 來自 workflow 輸入參數（`$inputs.xxx`）
- 可同時指定多個

**inputs 的 value_types**：
- `"image"`, `"float"`, `"integer"`, `"boolean"`, `"string"`, `"list"`, `"dict"` 等

### 8.3 code 欄位

```json
"code": {
  "type": "PythonCode",
  "run_function_code": "def run(self, image, anomaly_map):\n    import cv2\n    orig = image.numpy_image\n    amap = anomaly_map.numpy_image\n    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)\n    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)\n    amap_gray = cv2.cvtColor(amap, cv2.COLOR_BGR2GRAY)\n    masked = cv2.bitwise_and(amap_gray, amap_gray, mask=mask)\n    result_bgr = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)\n    from inference.core.workflows.execution_engine.entities.base import WorkflowImageData, ImageParentMetadata\n    return {'masked_image': WorkflowImageData(parent_metadata=image.parent_metadata, numpy_image=result_bgr)}",
  "imports": ["cv2"]
}
```

**`run` 函數規則**：
- 簽名必須是 `def run(self, param1, param2, ...)`，參數名稱對應 manifest 的 `inputs` key
- 回傳 `dict`，key 對應 manifest 的 `outputs` key
- `self._init_results` 可存取 `init` 函數的回傳值（若有定義）

**預設可用的 import**（不需要在 `imports` 裡宣告）：

```python
from typing import Any, List, Dict, Set, Optional
import supervision as sv
import numpy as np
import math, time, json, os
import requests, cv2, shapely
from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult
```

**額外 import**：在 `imports` 列表中宣告，或直接在 `run_function_code` 裡 `import`。

### 8.4 可選的 init 函數

若 block 需要一次性初始化（如載入設定檔），可定義 `init_function_code`：

```json
"code": {
  "type": "PythonCode",
  "run_function_code": "def run(self, image):\n    threshold = self._init_results['threshold']\n    ...",
  "init_function_code": "def init() -> dict:\n    return {'threshold': 100}",
  "imports": []
}
```

`init()` 在 block 建立時執行一次，回傳的 dict 存入 `self._init_results`。

### 8.5 完整範例：前景遮罩

把原本需要 plugin 的 `foreground_mask` 改為 Dynamic Block：

```json
{
  "version": "1.0",
  "dynamic_blocks_definitions": [
    {
      "type": "DynamicBlockDefinition",
      "manifest": {
        "type": "ManifestDescription",
        "block_type": "ForegroundMask",
        "description": "Filter anomaly map by foreground region",
        "inputs": {
          "image": {
            "type": "DynamicInputDefinition",
            "selector_types": ["step_output", "input_parameter"],
            "value_types": ["image"]
          },
          "anomaly_map": {
            "type": "DynamicInputDefinition",
            "selector_types": ["step_output"],
            "value_types": ["image"]
          }
        },
        "outputs": {
          "image": {
            "type": "DynamicOutputDefinition",
            "kind": ["image"]
          }
        }
      },
      "code": {
        "type": "PythonCode",
        "run_function_code": "def run(self, image, anomaly_map):\n    orig = image.numpy_image\n    amap = anomaly_map.numpy_image\n    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)\n    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)\n    amap_gray = cv2.cvtColor(amap, cv2.COLOR_BGR2GRAY)\n    masked = cv2.bitwise_and(amap_gray, amap_gray, mask=mask)\n    result_bgr = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)\n    return {'image': WorkflowImageData(parent_metadata=image.parent_metadata, numpy_image=result_bgr)}",
        "imports": []
      }
    }
  ],
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
      "images": "$inputs.image"
    },
    {
      "type": "ForegroundMask",
      "name": "fg_mask_step",
      "image": "$inputs.image",
      "anomaly_map": "$steps.ad_step.anomaly_map"
    },
    {
      "type": "roboflow_core/threshold@v1",
      "name": "threshold_step",
      "image": "$steps.fg_mask_step.image",
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
    {"type": "JsonField", "name": "anomaly_score", "selector": "$steps.ad_step.anomaly_score"},
    {"type": "JsonField", "name": "contour_count", "selector": "$steps.contour_step.number_contours"}
  ]
}
```

注意 step 裡的 `"type": "ForegroundMask"` 直接使用 `block_type` 名稱，不需要 `@v1` 後綴。

### 8.6 Dynamic Block vs Plugin

| | Dynamic Block | Plugin |
|---|---|---|
| 代碼位置 | workflow JSON 內 | 獨立 Python package |
| 安裝 | 不需要 | `pip install` |
| 重啟 server | 不需要 | 需要 |
| 跨 workflow 重用 | 需複製 JSON | import 即可 |
| 版本管理 | 隨 JSON 管理 | 獨立版本號 |
| 可存取 model_manager | 否 | 是 |
| 適合場景 | 影像轉換、簡單計算、原型 | 複雜業務邏輯、需要模型推論 |

### 8.7 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS` | `True` | 允許 Dynamic Block 執行自訂 Python |
| `WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE` | `local` | `local`（本地 exec）或 `modal`（遠端沙箱） |

---

## 9. 啟動環境變數

```bash
# 必要
ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True

# 若使用 business plugin
WORKFLOWS_PLUGINS=business_plugin

# Dynamic Block（預設已開啟）
ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=True
```

Workflow JSON 每次 POST 都帶完整 `specification`，修改 JSON 不需要重啟 server。
