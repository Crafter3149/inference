# Workflow 開發手冊

本手冊是 workflow 開發的完整指引。涵蓋 JSON 格式、模型 block、dynamic block 開發、
常見操作模式、完整案例、以及常見錯誤。

**完整案例檔案**（開發新 workflow 時務必先讀）：
- Workflow JSON: `D:\Project\inferserver\temp\task3_workflow.json`
- Client 測試腳本: `D:\Project\inferserver\temp\test_workflow_task3.py`
- 模型目錄: `D:\Project\inferserver\temp\models\` (locate/particle/ad)

---

## 目錄

1. [概念與 JSON 格式](#1-概念與-json-格式)
2. [模型 Block 參考](#2-模型-block-參考)
3. [常用 Roboflow 內建 Block](#3-常用-roboflow-內建-block)
4. [完整案例：Task3 多模型檢測管線](#4-完整案例task3-多模型檢測管線)
5. [常見操作模式](#5-常見操作模式)
6. [Dynamic Block 開發手冊](#6-dynamic-block-開發手冊)
7. [呼叫 API](#7-呼叫-api)
8. [環境變數](#8-環境變數)
9. [常見錯誤與陷阱](#9-常見錯誤與陷阱)
10. [速查表](#10-速查表-quick-reference)

---

## 1. 概念與 JSON 格式

### 1.1 核心概念

Workflow 是一個 JSON 定義的管線，由多個 **Step**（Block 實例）組成。
引擎根據 selector 引用（`$steps.xxx.yyy`、`$inputs.xxx`）自動分析依賴圖、決定執行順序。

**沒有依賴關係的 step 會並行執行**（ThreadPoolExecutor）。

### 1.2 JSON 頂層結構

```json
{
  "version": "1.3.0",
  "inputs": [ ... ],
  "dynamic_blocks_definitions": [ ... ],
  "steps": [ ... ],
  "outputs": [ ... ]
}
```

- `version` — workflow engine 版本（用 `"1.3.0"` 即可）
- `inputs` — 外部輸入的定義
- `dynamic_blocks_definitions` — 可選，內嵌的自訂 Python block
- `steps` — 管線步驟
- `outputs` — 最終回傳的欄位

### 1.3 inputs — 三種類型

#### WorkflowImage — 影像輸入

每張影像在管線中獨立處理。傳入 N 張 → 每個 step 執行 N 次（batch）。

```json
{"type": "WorkflowImage", "name": "image"}
```

Client 端送 1 張：
```python
"image": {"type": "base64", "value": base64_data}
```

Client 端送 N 張（batch）：
```python
"image": [
    {"type": "base64", "value": img1_b64},
    {"type": "base64", "value": img2_b64},
    {"type": "base64", "value": img3_b64},
]
```

#### WorkflowBatchInput — 與影像一一對應的 metadata

**必須與 WorkflowImage 等長**。用途：為每張影像附帶 metadata（如 cam_id），
讓 step 可以根據 metadata 做條件判斷。

```json
{"type": "WorkflowBatchInput", "name": "cam_id", "dimensionality": 1}
```

Client 端：
```python
"cam_id": [1, 2, 3, 4, 5, 6]  # 與 image 陣列等長
```

**重要**：`dimensionality: 1` 表示與 WorkflowImage 同一 batch 維度。

#### WorkflowParameter — 全域參數

所有影像共享的固定值。不隨 batch 展開。

```json
{"type": "WorkflowParameter", "name": "ad_cams", "default_value": [1, 4]}
```

Client 端：
```python
"ad_cams": [1, 4]
```

### 1.4 steps 與依賴解析

每個 step 必須有 `type`（block 類型）和 `name`（唯一名稱）。
其餘欄位是 block 的參數，可以用 selector 引用其他 step 的輸出或 inputs：

```json
{
  "type": "roboflow_core/threshold@v1",
  "name": "threshold_step",
  "image": "$steps.ad_step.anomaly_map",
  "thresh_value": "$inputs.bin_threshold"
}
```

引擎自動拓撲排序。引用關係決定執行順序，不需要手動指定。

### 1.5 outputs

定義 API 回傳的 JSON 欄位：

```json
{"type": "JsonField", "name": "verdict", "selector": "$steps.verdict_step.verdict"}
```

---

## 2. 模型 Block 參考

### 2.1 Anomaly Detection（AIE 內建）

| 屬性 | 值 |
|------|----|
| Block type | `roboflow_core/aie_anomaly_detection_model@v1` |
| 輸入 | `model_id`（本地路徑或 `$inputs.xxx`）、`images`（IMAGE_KIND） |
| 輸出 | `anomaly_map`（IMAGE_KIND，uint8 灰度 3ch）、`anomaly_score`（FLOAT_KIND） |

```json
{
  "type": "roboflow_core/aie_anomaly_detection_model@v1",
  "name": "ad_step",
  "model_id": "$inputs.ad_model_id",
  "images": "$inputs.image"
}
```

### 2.2 Object Detection（Roboflow 現有 Block）

AIE YOLO 偵測模型直接使用 Roboflow OD block，`model_id` 填本地路徑即可：

```json
{
  "type": "roboflow_core/roboflow_object_detection_model@v2",
  "name": "locate_step",
  "model_id": "$inputs.locate_model_id",
  "images": "$inputs.image",
  "confidence": 0.5
}
```

### 2.3 Classification（Roboflow 現有 Block）

```json
{
  "type": "roboflow_core/roboflow_classification_model@v2",
  "name": "cls_step",
  "model_id": "$inputs.cls_model_id",
  "images": "$inputs.image"
}
```

### 2.4 Instance Segmentation（Roboflow 現有 Block）

```json
{
  "type": "roboflow_core/roboflow_instance_segmentation_model@v2",
  "name": "seg_step",
  "model_id": "$inputs.seg_model_id",
  "images": "$inputs.image"
}
```

---

## 3. 常用 Roboflow 內建 Block

| Block type | 功能 | 關鍵參數 |
|------------|------|----------|
| `roboflow_core/roboflow_object_detection_model@v2` | 物件偵測 | `model_id`, `confidence`, `images` |
| `roboflow_core/dynamic_crop@v1` | 按偵測框裁切 | `images`, `predictions` |
| `roboflow_core/image_slicer@v2` | 切片（SAHI） | `image`, `slice_width`, `slice_height`, `overlap_ratio_*` |
| `roboflow_core/detections_stitch@v1` | 合併切片偵測 | `reference_image`, `predictions`, `iou_threshold` |
| `roboflow_core/threshold@v1` | 二值化（8 種方法） | `image`, `thresh_value`, `threshold_type` |
| `roboflow_core/morphological_transformation@v1` | 形態學運算 | `image`, `operation`, `kernel_size` |
| `roboflow_core/contours_detection@v1` | 找輪廓 | `image` → `contours`, `number_contours` |
| `roboflow_core/expression@v1` | 條件判定 / 運算 | `data`, `expression` |
| `roboflow_core/continue_if@v1` | 條件流程控制（gate） | `condition_statement`, `next_steps` |
| `roboflow_core/convert_grayscale@v1` | 轉灰度 | `image` |
| `roboflow_core/image_preprocessing@v1` | resize 等 | `width`, `height` |
| `roboflow_core/detections_filter@v1` | 過濾偵測結果 | `predictions`, `operations` |

所有 Block 的參數幾乎都支援 `$inputs.xxx` 動態參數。

---

## 4. 完整案例：Task3 多模型檢測管線

**參考檔案**：`temp/task3_workflow.json`、`temp/test_workflow_task3.py`

### 4.1 案例說明

每個產品有 6 張不同角度（cam 1-6）的影像，需要同時跑三個模型：
- **Locate**（YOLO OD）— 定位產品位置
- **Particle**（YOLO OD + SAHI 切片）— 偵測微粒缺陷
- **AD**（EfficientAD）— 異常檢測，但只需在 cam 1,4 執行

最終輸出 product-level 的 OK/NG 判定。

### 4.2 管線架構圖

```
$inputs.image (batch of 6 images)
    │
    ├─── slicer_step          ← image_slicer@v2 (SAHI 切片)
    │        │
    │        └── particle_step ← OD model (在切片上偵測微粒)
    │               │
    │               └── stitch_step ← detections_stitch (合併回原圖座標)
    │                      │
    │                      └── image_after_stitch ← [Dynamic] image passthrough
    │                              │
    │                              └── locate_step ← OD model (定位產品)
    │                                      │
    │                                      ├── roi_filter_step ← [Dynamic] ROI 過濾微粒
    │                                      │
    │                                      └── (下接 AD 分支)
    │
    ├─── ad_gate              ← continue_if (cam_id in ad_cams?)
    │        │
    │        └── ad_step      ← AD model (異常檢測)
    │               │
    │               └── ad_analysis_step ← [Dynamic] Ring mask 分析
    │
    └─── verdict_step         ← [Dynamic] batch 聚合 → OK/NG
```

### 4.3 inputs 設計

```json
"inputs": [
  {"type": "WorkflowImage", "name": "image"},
  {"type": "WorkflowBatchInput", "name": "cam_id", "dimensionality": 1},
  {"type": "WorkflowParameter", "name": "cam_ids"},
  {"type": "WorkflowParameter", "name": "particle_model_id"},
  {"type": "WorkflowParameter", "name": "locate_model_id"},
  {"type": "WorkflowParameter", "name": "ad_model_id"},
  {"type": "WorkflowParameter", "name": "ad_cams", "default_value": [1, 4]},
  {"type": "WorkflowParameter", "name": "ad_contour_th", "default_value": 1},
  {"type": "WorkflowParameter", "name": "roi_expand", "default_value": 0.13},
  {"type": "WorkflowParameter", "name": "center_mask_ratio", "default_value": 0.78},
  {"type": "WorkflowParameter", "name": "ad_binarize_threshold", "default_value": 100},
  {"type": "WorkflowParameter", "name": "ad_contour_min_area", "default_value": 50}
]
```

**設計重點**：

| Input | 類型 | 用途 |
|-------|------|------|
| `image` | WorkflowImage | 6 張影像，引擎自動 batch 展開 |
| `cam_id` | WorkflowBatchInput | 每張的 cam 編號，與 image 一一對應，用於 continue_if 閘門 |
| `cam_ids` | WorkflowParameter | cam 編號清單（全域），傳給聚合 block |
| `ad_cams` | WorkflowParameter | 需要跑 AD 的 cam 清單 `[1, 4]` |
| 其餘 | WorkflowParameter | 各種閾值，帶預設值 |

**為什麼 `cam_id`（BatchInput）和 `cam_ids`（Parameter）同時存在？**
- `cam_id` 是 per-image 的，讓 `continue_if` 能對每張影像做條件判斷
- `cam_ids` 是全域的 list，傳給聚合 block 做最終報表

### 4.4 10 個 Step 逐一解說

#### Step 1-3: SAHI 切片偵測（Particle）

```json
{"type": "roboflow_core/image_slicer@v2", "name": "slicer_step",
 "image": "$inputs.image", "slice_width": 960, "slice_height": 960,
 "overlap_ratio_width": 0.2, "overlap_ratio_height": 0.2},

{"type": "roboflow_core/roboflow_object_detection_model@v2", "name": "particle_step",
 "images": "$steps.slicer_step.slices", "model_id": "$inputs.particle_model_id",
 "confidence": 0.3},

{"type": "roboflow_core/detections_stitch@v1", "name": "stitch_step",
 "reference_image": "$inputs.image", "predictions": "$steps.particle_step.predictions",
 "overlap_filtering_strategy": "nms", "iou_threshold": 0.5}
```

**模式**：image_slicer 把大圖切成小塊 → 模型在小塊上推論 → stitch 把偵測結果合併回原圖座標。
適用於小物體偵測（微粒、文字等）。

#### Step 4: Image Passthrough（GPU 序列化）

```json
{"type": "ImageAfterStitch", "name": "image_after_stitch",
 "image": "$inputs.image",
 "stitch_predictions": "$steps.stitch_step.predictions"}
```

**為什麼需要 passthrough？** 如果 locate_step 直接引用 `$inputs.image`，它會跟 particle pipeline
**並行執行**。在 `MAX_BATCH_SIZE=1` 的 GPU 環境下，兩個模型同時載入會 OOM。

passthrough block 引用了 `stitch_step.predictions`，迫使引擎等 particle pipeline 完成後，
才執行 locate_step。**這是 GPU 序列化的標準技巧**。

```python
# Dynamic block 實作：什麼都不做，只是傳遞 image
"run_function_code": "def run(self, image, stitch_predictions):\n    return {'output_image': image}\n"
```

#### Step 5: 物件定位（Locate）

```json
{"type": "roboflow_core/roboflow_object_detection_model@v2", "name": "locate_step",
 "images": "$steps.image_after_stitch.output_image",
 "model_id": "$inputs.locate_model_id", "confidence": 0.5}
```

注意 `images` 引用 passthrough 的輸出，確保在 particle pipeline 之後執行。

#### Step 6: ROI 過濾（Dynamic Block）

```json
{"type": "ParticleRoiFilter", "name": "roi_filter_step",
 "image": "$inputs.image",
 "locate_predictions": "$steps.locate_step.predictions",
 "particle_predictions": "$steps.stitch_step.predictions",
 "roi_expand": "$inputs.roi_expand",
 "center_mask_ratio": "$inputs.center_mask_ratio"}
```

用 locate 的 bounding box 建立圓形 ROI，只計算 ROI 內的微粒數量。
同時 passthrough image 給下游 AD 使用。

#### Step 7: AD 閘門（continue_if）

```json
{"type": "roboflow_core/continue_if@v1", "name": "ad_gate",
 "condition_statement": {
   "type": "StatementGroup",
   "statements": [{
     "type": "BinaryStatement",
     "left_operand": {"type": "DynamicOperand", "operand_name": "cam_id"},
     "comparator": {"type": "in (Sequence)"},
     "right_operand": {"type": "DynamicOperand", "operand_name": "ad_cams"}
   }]
 },
 "evaluation_parameters": {
   "cam_id": "$inputs.cam_id",
   "ad_cams": "$inputs.ad_cams"
 },
 "next_steps": ["$steps.ad_step"]}
```

**效果**：只有 `cam_id` 在 `ad_cams` 清單中的影像，才會執行 `ad_step`。
cam 2,3,5,6 的 AD 結果為 `None`，不會佔用 GPU。

#### Step 8-9: AD 推論 + Ring Mask 分析

```json
{"type": "roboflow_core/aie_anomaly_detection_model@v1", "name": "ad_step",
 "images": "$steps.roi_filter_step.output_image",
 "model_id": "$inputs.ad_model_id"},

{"type": "AdRingAnalysis", "name": "ad_analysis_step",
 "image": "$inputs.image",
 "locate_predictions": "$steps.locate_step.predictions",
 "anomaly_map": "$steps.ad_step.anomaly_map", ...}
```

`AdRingAnalysis` 用 locate 的 bbox 建立環形 mask（外圓 - 內圓），
只在環形區域做異常分析。輸出 contour 數量和分數。

#### Step 10: Product Verdict（Batch 聚合）

```json
{"type": "ProductVerdict", "name": "verdict_step",
 "n_particles": "$steps.roi_filter_step.n_particles",
 "ad_contours": "$steps.ad_analysis_step.ad_contours",
 "ad_scores": "$steps.ad_analysis_step.ad_score",
 "cam_ids": "$inputs.cam_ids",
 "ad_cams": "$inputs.ad_cams",
 "ad_contour_th": "$inputs.ad_contour_th"}
```

manifest 設定：
```json
"accepts_batch_input": true,
"output_dimensionality_offset": -1
```

`run()` 收到所有 6 張影像的結果，聚合為單一 OK/NG 判定。
見 [6.6 Batch 聚合 Block](#66-batch-聚合-block-accepts_batch_input)。

### 4.5 Client 端呼叫

```python
import json, base64, requests

SERVER = "http://localhost:9001"
workflow_spec = json.loads(Path("task3_workflow.json").read_text())

# 先預載模型（首次需要，後續可跳過）
for model_id in [LOCATE_MODEL_ID, PARTICLE_MODEL_ID, AD_MODEL_ID]:
    requests.post(f"{SERVER}/model/add", json={"model_id": model_id})

# 準備 6 張影像
images = []
cam_ids = []
for cam_id, img_path in cam_images:  # [(1, "cam1.jpg"), (2, "cam2.jpg"), ...]
    with open(img_path, "rb") as f:
        images.append({"type": "base64", "value": base64.b64encode(f.read()).decode()})
    cam_ids.append(cam_id)

# 執行 workflow
resp = requests.post(f"{SERVER}/workflows/run", json={
    "specification": workflow_spec,
    "inputs": {
        "image": images,          # 6 張影像陣列
        "cam_id": cam_ids,        # [1,2,3,4,5,6] — 與 image 等長
        "cam_ids": cam_ids,       # 同上，但作為 Parameter 傳給聚合 block
        "particle_model_id": "D:/models/particle",
        "locate_model_id": "D:/models/locate",
        "ad_model_id": "D:/models/ad",
    },
})

result = resp.json()["outputs"][0]
# result["verdict"]  → "OK" 或 "NG"
# result["reason"]   → "clean" 或 "YOLO(3det)+AD(1cam)"
# result["details"]  → [{cam:1, n_particles:0, ad_contours:2, ad_score:0.34}, ...]
```

### 4.6 性能數據

| 指標 | 值 |
|------|----|
| GPU memory | ~3-4 GB（3 模型 + 推論） |
| 速度 | ~5s / product（6 張影像） |
| 模型權重 | locate 6MB + particle 42MB + AD 81MB = ~130MB |
| 環境 | MAX_BATCH_SIZE=1, MAX_ACTIVE_MODELS=1 |

---

## 5. 常見操作模式

### 5.1 Batch 輸入 — 多張影像一次送入

**場景**：一個產品有多個角度/光源的影像，需要一次全部處理。

```json
"inputs": [
  {"type": "WorkflowImage", "name": "image"},
  {"type": "WorkflowBatchInput", "name": "cam_id", "dimensionality": 1}
]
```

Client 端：
```python
"inputs": {
    "image": [img1, img2, img3, img4, img5, img6],  # N 張
    "cam_id": [1, 2, 3, 4, 5, 6],                    # 與 image 等長
}
```

pipeline 中的每個 step 會對 N 張影像各執行一次。
直到遇到 `accepts_batch_input: true` 的聚合 block，才合併為單一結果。

### 5.2 條件執行 — continue_if 閘門

**場景**：某些 step 只需要對部分影像執行（例如 AD 只跑特定 cam）。

```json
{
  "type": "roboflow_core/continue_if@v1",
  "name": "ad_gate",
  "condition_statement": {
    "type": "StatementGroup",
    "statements": [{
      "type": "BinaryStatement",
      "left_operand": {"type": "DynamicOperand", "operand_name": "cam_id"},
      "comparator": {"type": "in (Sequence)"},
      "right_operand": {"type": "DynamicOperand", "operand_name": "ad_cams"}
    }]
  },
  "evaluation_parameters": {
    "cam_id": "$inputs.cam_id",
    "ad_cams": "$inputs.ad_cams"
  },
  "next_steps": ["$steps.ad_step"]
}
```

**重點**：
- `cam_id` 必須是 `WorkflowBatchInput`（per-image），不能是 Parameter
- `ad_cams` 是 `WorkflowParameter`（全域 list）
- `next_steps` 指定條件為 true 時才執行的下游 step
- 條件為 false 的影像，下游 step 輸出為 `None`

**可用的 comparator**：
`"equal (==)"`, `"not equal (!=)"`, `"lower than (<)"`, `"greater than (>)"`,
`"lower or equal than (<=)"`, `"greater or equal than (>=)"`,
`"in (Sequence)"`, `"not in (Sequence)"`,
`"str contains"`, `"str not contains"`,
`"is True"`, `"is False"`, `"is None"`, `"is not None"`

### 5.3 Image Slicer + Stitch（SAHI 模式）

**場景**：大圖中偵測小物體。將大圖切成重疊的小塊，分別偵測，再合併結果。

```json
{"type": "roboflow_core/image_slicer@v2", "name": "slicer",
 "image": "$inputs.image",
 "slice_width": 960, "slice_height": 960,
 "overlap_ratio_width": 0.2, "overlap_ratio_height": 0.2},

{"type": "roboflow_core/roboflow_object_detection_model@v2", "name": "detect",
 "images": "$steps.slicer.slices",
 "model_id": "$inputs.model_id", "confidence": 0.3},

{"type": "roboflow_core/detections_stitch@v1", "name": "stitch",
 "reference_image": "$inputs.image",
 "predictions": "$steps.detect.predictions",
 "overlap_filtering_strategy": "nms", "iou_threshold": 0.5}
```

- `slicer.slices` 輸出 M 張切片（維度增加）
- `detect` 對每張切片各執行一次
- `stitch` 把 M 份偵測結果合併回原圖座標，NMS 去重

### 5.4 Image Passthrough — GPU 序列化

**場景**：多個模型 step 如果沒有依賴關係，引擎會並行執行。
在 `MAX_BATCH_SIZE=1` 的 GPU 環境下，多模型同時推論會 OOM。

**解法**：插入一個什麼都不做的 passthrough block，建立人為依賴。

```json
{
  "type": "DynamicBlockDefinition",
  "manifest": {
    "type": "ManifestDescription",
    "block_type": "ImageAfterStitch",
    "description": "Image passthrough after particle pipeline completes",
    "inputs": {
      "image": {"type": "DynamicInputDefinition", "selector_types": ["input_image"]},
      "stitch_predictions": {"type": "DynamicInputDefinition", "selector_types": ["step_output"]}
    },
    "outputs": {
      "output_image": {"type": "DynamicOutputDefinition", "kind": ["image"]}
    }
  },
  "code": {
    "type": "PythonCode",
    "run_function_code": "def run(self, image, stitch_predictions):\n    return {'output_image': image}\n"
  }
}
```

使用：
```json
{"type": "ImageAfterStitch", "name": "image_after_stitch",
 "image": "$inputs.image",
 "stitch_predictions": "$steps.stitch_step.predictions"},

{"type": "roboflow_core/roboflow_object_detection_model@v2", "name": "locate_step",
 "images": "$steps.image_after_stitch.output_image", ...}
```

`locate_step` 引用 passthrough 輸出 → 引擎排在 particle pipeline 之後。

### 5.5 Batch 聚合 — 多張結果合併為單一判定

**場景**：N 張影像各自處理完後，需要合併為一個 product-level 結果。

必須在 dynamic block 的 manifest 中設定：
```json
"accepts_batch_input": true,
"output_dimensionality_offset": -1
```

- `accepts_batch_input: true` → `run()` 收到完整 Batch 而非單一元素
- `output_dimensionality_offset: -1` → 輸出維度降一級（N → 1）

詳見 [6.6 Batch 聚合 Block](#66-batch-聚合-block-accepts_batch_input)。

### 5.6 模型前載

Workflow 內的 model block 會在首次執行時自動載入模型。但首次載入較慢。
建議在 workflow 執行前先 POST /model/add 預載：

```python
for model_id in [LOCATE_MODEL_ID, PARTICLE_MODEL_ID, AD_MODEL_ID]:
    requests.post(f"{SERVER}/model/add", json={"model_id": model_id}, timeout=120)
```

**注意**：`MAX_ACTIVE_MODELS` 控制同時駐留記憶體的模型數量。
設為 1 時，載入新模型會踢掉舊的（LRU cache）。設為 3 可以同時保留 3 個模型。

---

## 6. Dynamic Block 開發手冊

Dynamic Block 讓你直接在 workflow JSON 中內嵌 Python 函數，不需要安裝 plugin。
代碼隨 JSON 一起發送，server 在運行時動態編譯執行。

### 6.1 基本結構

```json
{
  "type": "DynamicBlockDefinition",
  "manifest": {
    "type": "ManifestDescription",
    "block_type": "MyBlockName",
    "description": "What this block does",
    "inputs": { ... },
    "outputs": { ... }
  },
  "code": {
    "type": "PythonCode",
    "run_function_code": "def run(self, param1, param2):\n    return {'output1': value}",
    "imports": []
  }
}
```

在 steps 中使用：`"type": "MyBlockName"`（直接用 block_type 名稱，不需要 `@v1` 後綴）。

`run` 函數規則：
- 簽名 `def run(self, param1, param2, ...)` — 參數名稱對應 manifest inputs 的 key
- 回傳 `dict` — key 對應 manifest outputs 的 key

### 6.2 selector_types 完整對照表

manifest 的 input 定義中，`selector_types` 決定這個參數可以接受什麼來源的資料。
**這是最容易出錯的地方，必須正確選擇。**

| selector_type | 接受的 selector | run() 收到的型別 | 用途 |
|---------------|----------------|-----------------|------|
| `input_image` | `$inputs.image`, `$steps.xxx.output_image` | `WorkflowImageData` | 接收影像（原圖或 step 產出的影像） |
| `step_output` | `$steps.xxx.predictions`, `$steps.xxx.score` | 原始值（sv.Detections, int, float, etc.） | 接收非影像的 step 輸出 |
| `step_output_image` | `$steps.xxx.anomaly_map`, `$steps.xxx.image` | `WorkflowImageData` | 接收 step 產出的 IMAGE_KIND 輸出 |
| `input_parameter` | `$inputs.xxx` | 原始值（int, float, list, str, etc.） | 接收 WorkflowParameter |

**選擇規則**：
- 影像來自 `$inputs.image` 或 passthrough → 用 `input_image`
- 影像來自 step 的 IMAGE_KIND 輸出（如 `anomaly_map`）→ 用 `step_output_image`
- 偵測結果、數字、字串來自 step → 用 `step_output`
- 固定參數來自 `$inputs.xxx` → 用 `input_parameter`
- 一個 input 可以同時指定多個 selector_type

**搭配的 value_types**（用於 `input_parameter` 和 `step_output`）：

| value_type | 說明 |
|------------|------|
| `"float"` | 浮點數 |
| `"integer"` | 整數 |
| `"boolean"` | 布林 |
| `"string"` | 字串 |
| `"list"` | 清單 |
| `"dict"` | 字典 |
| `"image"` | 影像（配合 `input_image` / `step_output_image`） |

### 6.3 存取影像資料（WorkflowImageData）

當 input 的 selector_type 是 `input_image` 或 `step_output_image` 時，
`run()` 收到的是 `WorkflowImageData` 物件：

```python
def run(self, image, anomaly_map):
    # 取得 numpy array (BGR, HWC, uint8)
    img_np = image.numpy_image          # shape: (H, W, 3)
    amap_np = anomaly_map.numpy_image   # shape: (H, W, 3)

    # 取得尺寸
    h, w = img_np.shape[:2]

    # 灰度轉換
    gray = cv2.cvtColor(amap_np, cv2.COLOR_BGR2GRAY)  # → (H, W)
```

**注意**：`anomaly_map` 是 3 通道 BGR 灰度圖（三個通道值相同），
使用前通常需要 `cv2.cvtColor(amap, cv2.COLOR_BGR2GRAY)`。

### 6.4 存取偵測結果（sv.Detections）

當 input 的 selector_type 是 `step_output` 且來源是 OD model 的 `predictions` 時，
`run()` 收到的是 `sv.Detections` 物件：

```python
def run(self, locate_predictions, particle_predictions):
    # 判斷是否為空
    if locate_predictions is None or len(locate_predictions) == 0:
        return {'count': 0}

    # 存取 bounding box — shape: (N, 4)，格式 [x1, y1, x2, y2]
    for i in range(len(locate_predictions)):
        x1, y1, x2, y2 = locate_predictions.xyxy[i]

    # 存取 confidence — shape: (N,)
    best_idx = int(np.argmax(locate_predictions.confidence))

    # 偵測數量
    n = len(locate_predictions)
```

**重要**：dynamic block 收到的 predictions 可能是 `None`（如被 continue_if 跳過），
**一定要先檢查 `is None or len(...) == 0`**。

### 6.5 回傳影像（WorkflowImageData）

如果 output kind 是 `["image"]`，必須回傳 `WorkflowImageData`：

```python
def run(self, image):
    result_bgr = cv2.GaussianBlur(image.numpy_image, (5, 5), 0)
    return {
        'output_image': WorkflowImageData(
            parent_metadata=image.parent_metadata,
            numpy_image=result_bgr
        )
    }
```

**或者直接回傳原圖**（passthrough）：
```python
def run(self, image, some_dependency):
    return {'output_image': image}  # 直接傳遞 WorkflowImageData
```

`WorkflowImageData` 和 `ImageParentMetadata` 在預設 import 中已可用（來自 `Batch` import）。

### 6.6 Batch 聚合 Block（accepts_batch_input）

**場景**：把 N 張影像的結果合併為單一 product-level 判定。

manifest 必須設定：
```json
"accepts_batch_input": true,
"output_dimensionality_offset": -1
```

**完整範例** — 產品判定（從 task3_workflow.json 提取）：

```json
{
  "type": "DynamicBlockDefinition",
  "manifest": {
    "type": "ManifestDescription",
    "block_type": "ProductVerdict",
    "description": "Aggregate per-image results into product-level verdict",
    "inputs": {
      "n_particles": {
        "type": "DynamicInputDefinition",
        "selector_types": ["step_output"],
        "value_types": ["integer"]
      },
      "ad_contours": {
        "type": "DynamicInputDefinition",
        "selector_types": ["step_output"],
        "value_types": ["integer"]
      },
      "cam_ids": {
        "type": "DynamicInputDefinition",
        "selector_types": ["input_parameter"],
        "value_types": ["list"]
      }
    },
    "outputs": {
      "verdict": {"type": "DynamicOutputDefinition", "kind": ["string"]},
      "details": {"type": "DynamicOutputDefinition", "kind": []}
    },
    "accepts_batch_input": true,
    "output_dimensionality_offset": -1
  },
  "code": {
    "type": "PythonCode",
    "run_function_code": "def run(self, n_particles, ad_contours, cam_ids):\n    def flatten(x):\n        items = []\n        for item in x:\n            if hasattr(item, '__iter__') and not isinstance(item, (str, bytes, float, int)):\n                items.extend(flatten(item))\n            else:\n                items.append(item)\n        return items\n\n    np_list = flatten(n_particles)\n    ac_list = flatten(ad_contours)\n    # ... 聚合邏輯 ...\n    return {'verdict': 'OK' if all_clean else 'NG', 'details': details}\n"
  }
}
```

**關鍵注意事項**：

1. `run()` 收到的 batch input 是 `Batch` 容器（可能巢狀），**必須 `flatten()` 後使用**
2. 被 continue_if 跳過的元素值為 `None`，flatten 後記得處理
3. `output_dimensionality_offset: -1` 讓 N 筆輸入合併為 1 筆輸出
4. output kind 用 `[]`（空 list）表示任意型別

**`flatten()` 是必需的**：因為 image_slicer 等 block 會增加 batch 維度，
導致結果可能是巢狀結構 `[[v1, v2], [v3, v4], ...]`。flatten 確保得到扁平 list。

### 6.7 預設可用 import

以下在 `run_function_code` 中直接可用，不需要在 `imports` 宣告：

```python
from typing import Any, List, Dict, Set, Optional
import supervision as sv
import numpy as np
import math, time, json, os
import requests, cv2, shapely
from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult
```

需要其他 library：在 `imports` 列表宣告或在 `run_function_code` 中 inline import。

### 6.8 init 函數（可選）

若 block 需要一次性初始化，可定義 `init_function_code`：

```json
"code": {
  "type": "PythonCode",
  "run_function_code": "def run(self, image):\n    th = self._init_results['threshold']\n    ...",
  "init_function_code": "def init() -> dict:\n    return {'threshold': 100}"
}
```

`init()` 在 block 建立時執行一次，回傳的 dict 存入 `self._init_results`。

### 6.9 Dynamic Block vs Plugin

| | Dynamic Block | Plugin |
|---|---|---|
| 代碼位置 | workflow JSON 內 | 獨立 Python package |
| 安裝 | 不需要 | `pip install` |
| 重啟 server | 不需要 | 需要 |
| 跨 workflow 重用 | 需複製 JSON | import 即可 |
| 可存取 model_manager | 否 | 是 |
| 適合場景 | 影像轉換、ROI 計算、聚合 | 複雜業務邏輯、需要模型推論 |

---

## 7. 呼叫 API

### 7.1 直接執行（POST /workflows/run）

每次帶完整 spec，不儲存。修改 JSON 不需重啟 server。

```python
import json, base64, requests
SERVER = "http://localhost:9001"

with open("workflow.json") as f:
    workflow_spec = json.load(f)
with open("test.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post(f"{SERVER}/workflows/run", json={
    "specification": workflow_spec,
    "inputs": {
        "image": {"type": "base64", "value": img_b64},
        "ad_model_id": "/models/ad",
    }
})
result = resp.json()["outputs"][0]
```

### 7.2 註冊後呼叫（Builder API）

先儲存，之後用 ID 呼叫。需要 `ENABLE_BUILDER=true`。

```python
# 取得 CSRF token（管理端點需要）
from pathlib import Path
csrf_token = Path("/tmp/cache/workflow/local/.csrf").read_text()

# 註冊
requests.post(f"{SERVER}/build/api/{workflow_id}",
              json=workflow_spec, headers={"X-CSRF": csrf_token})

# 用 ID 呼叫（不需要 CSRF）
requests.post(f"{SERVER}/local/workflows/{workflow_id}", json={
    "inputs": {"image": {"type": "base64", "value": img_b64}, ...}
})

# 管理
requests.get(f"{SERVER}/build/api")                                    # 列出全部
requests.get(f"{SERVER}/build/api/{workflow_id}")                      # 取得特定
requests.delete(f"{SERVER}/build/api/{workflow_id}", headers=headers)  # 刪除
```

CSRF token 在 server 啟動時自動產生，寫入 `{MODEL_CACHE_DIR}/workflow/local/.csrf`。
只有 `/build/api/*` 的管理端點需要，執行端點（`/workflows/run`、`/local/workflows/{id}`）不需要。

### 7.3 直接呼叫模型（不用 Workflow）

```python
# 載入
requests.post(f"{SERVER}/model/add", json={"model_id": "/models/ad"})
# 推論
resp = requests.post(f"{SERVER}/infer", json={
    "model_id": "/models/ad",
    "image": {"type": "base64", "value": img_b64}
})
```

---

## 8. 環境變數

### 啟動 server

```bash
# 必要：允許本地路徑載入模型
ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True

# 必要：允許 Dynamic Block 執行自訂 Python（預設已開啟）
ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=True

# GPU 環境建議
MAX_BATCH_SIZE=1               # 避免 OOM
MAX_ACTIVE_MODELS=1            # 或根據 GPU memory 調整

# 可選：啟用 Builder UI
ENABLE_BUILDER=true

# 可選：若使用 plugin
WORKFLOWS_PLUGINS=business_plugin
```

完整啟動範例見 `start_gpu_server.bat`。

### 模型目錄結構

每個模型一個目錄：
```
model_dir/
├── model_config.json    # 必須
└── weights.pt           # 或 weights.nst
```

`model_config.json` 範例：
```json
{"model_architecture": "aie", "task_type": "object-detection", "backend_type": "ultralytics"}
{"model_architecture": "aie", "task_type": "anomaly-detection", "backend_type": "torch"}
```

---

## 9. 常見錯誤與陷阱

### 9.1 selector_types 選錯

**症狀**：`KeyError`、`TypeError`、或收到 `None`。

| 錯誤 | 正確 |
|------|------|
| 影像用 `step_output` | 影像用 `input_image` 或 `step_output_image` |
| 參數用 `step_output` | 參數用 `input_parameter` |
| anomaly_map 用 `step_output` | anomaly_map 用 `step_output_image`（IMAGE_KIND） |

### 9.2 忘記處理 None（continue_if 跳過的影像）

被 continue_if 跳過的影像，下游 step 的輸出為 `None`。
聚合 block 必須處理：

```python
val = int(np_list[i]) if np_list[i] is not None else 0
```

### 9.3 忘記 flatten（batch 聚合時）

image_slicer 會增加 batch 維度，導致聚合 block 收到巢狀結構。
**必須 `flatten()` 後再處理**：

```python
def flatten(x):
    items = []
    for item in x:
        if hasattr(item, '__iter__') and not isinstance(item, (str, bytes, float, int)):
            items.extend(flatten(item))
        else:
            items.append(item)
    return items
```

### 9.4 GPU OOM（多模型並行）

引擎會並行執行無依賴的 step。如果兩個模型 step 都引用 `$inputs.image`，
它們會同時載入 GPU → OOM。

**解法**：用 Image Passthrough 建立人為依賴（見 [5.4](#54-image-passthrough--gpu-序列化)）。

### 9.5 anomaly_map 是 3 通道

AD block 輸出的 `anomaly_map` 是 3 通道 BGR（灰度值複製三次）。
做 threshold / contour 前需要轉灰度：

```python
amap = anomaly_map.numpy_image  # (H, W, 3)
if amap.ndim == 3:
    amap = cv2.cvtColor(amap, cv2.COLOR_BGR2GRAY)  # → (H, W)
```

### 9.6 WorkflowBatchInput vs WorkflowParameter 混淆

- `WorkflowBatchInput` — per-image metadata，與 WorkflowImage 等長，引擎自動展開
- `WorkflowParameter` — 全域常數，所有影像共享

`continue_if` 的條件變數**必須用 BatchInput**，否則無法 per-image 判斷。

### 9.7 Dynamic Block 內的 import

`cv2`, `numpy`, `math`, `json` 等已經預設可用。
但如果需要 `torch` 等非預設 library，必須在 `run_function_code` 內 inline import：

```python
"run_function_code": "def run(self, ...):\n    import torch\n    ..."
```

### 9.8 output kind 為空表示任意型別

如果 output 可能是複雜結構（如 list of dict），kind 設為空陣列：

```json
"details": {"type": "DynamicOutputDefinition", "kind": []}
```

---

## 10. 速查表 (Quick Reference)

### 開發新 workflow 的步驟

1. 讀 `temp/task3_workflow.json` 作為參考
2. 定義 inputs（WorkflowImage + BatchInput + Parameter）
3. 排列 steps，注意 GPU 序列化（passthrough pattern）
4. 需要條件執行 → `continue_if@v1`
5. 需要自訂邏輯 → Dynamic Block，注意 selector_types
6. 需要聚合 → `accepts_batch_input: true` + `output_dimensionality_offset: -1`
7. 測試：先 POST /model/add 預載，再 POST /workflows/run

### selector_types 速查

```
影像（原圖/passthrough）  → selector_types: ["input_image"]
影像（step 輸出，如 anomaly_map） → selector_types: ["step_output_image"]
偵測結果 / 數字 / 字串（step 輸出） → selector_types: ["step_output"]
全域參數 → selector_types: ["input_parameter"]
```

### Dynamic Block 接收的型別速查

```
input_image         → WorkflowImageData → .numpy_image (BGR HWC uint8)
step_output_image   → WorkflowImageData → .numpy_image (BGR HWC uint8)
step_output (predictions) → sv.Detections → .xyxy, .confidence, len()
step_output (number)      → int / float
step_output (被 continue_if 跳過) → None
input_parameter     → 原始 Python 型別 (int, float, list, str, etc.)
```

### 回傳值速查

```
回傳影像 → WorkflowImageData(parent_metadata=image.parent_metadata, numpy_image=bgr_array)
回傳數字 → int / float
回傳字串 → str
回傳任意 → dict / list（output kind 設 []）
```

### batch 聚合速查

```json
manifest 加上:
  "accepts_batch_input": true,
  "output_dimensionality_offset": -1

run() 中:
  1. flatten() 所有 batch input
  2. 處理 None（被 continue_if 跳過的）
  3. 回傳單一結果（不是 list）
```

### 常用 step JSON 片段

```json
// OD 模型
{"type": "roboflow_core/roboflow_object_detection_model@v2",
 "name": "detect", "images": "$inputs.image",
 "model_id": "$inputs.model_id", "confidence": 0.5}

// AD 模型
{"type": "roboflow_core/aie_anomaly_detection_model@v1",
 "name": "ad", "images": "$inputs.image",
 "model_id": "$inputs.ad_model_id"}

// 條件判定
{"type": "roboflow_core/expression@v1",
 "name": "judge", "data": {"count": "$steps.contour.number_contours"},
 "expression": "count >= 1"}

// SAHI 切片偵測
{"type": "roboflow_core/image_slicer@v2", "name": "slicer",
 "image": "$inputs.image", "slice_width": 960, "slice_height": 960,
 "overlap_ratio_width": 0.2, "overlap_ratio_height": 0.2}
```
