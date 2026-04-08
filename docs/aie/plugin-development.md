# Plugin 開發手冊

本手冊說明如何建立 Workflow Plugin，為 Inference Server 新增自訂 Block。

> AD 模型推論 block（`aie_anomaly_detection_model@v1`）已內建於 server，
> 不需要 plugin。Plugin 用於**業務邏輯**，例如前景遮罩、輪廓過濾等。

> **不想建 plugin？** 對於一次性邏輯或快速原型，可以使用 **Dynamic Block**
> 直接在 workflow JSON 中內嵌 Python 代碼，免安裝。詳見 [Workflow 工作手冊 §8](workflow-guide.md#8-dynamic-block自訂-python-block)。

---

## 1. Plugin 是什麼

Plugin 是一個獨立的 Python package，安裝到 server 的 Python 環境後，
server 啟動時自動載入其中的 Block 類別。

Plugin **不包含**：
- Server 啟動邏輯（那是部署設定，見 Build 手冊）
- 模型權重（放在模型目錄，獨立管理）
- Workflow JSON（放在應用層，獨立管理）

---

## 2. 專案結構

```
business_plugin/                         # Plugin 專案根目錄
├── pyproject.toml                       # Python package 配置
├── business_plugin/                     # Plugin package
│   ├── __init__.py                      #   load_blocks() 入口
│   ├── foreground_mask/                 #   Block 1
│   │   └── v1.py                        #     ForegroundMaskBlockV1
│   └── contour_filter/                  #   Block 2
│       └── v1.py                        #     ContourFilterBlockV1
└── tests/                               # 測試
```

每個 Block 一個目錄 + 版本檔案，與 Roboflow 內建 Block 結構一致。

---

## 3. pyproject.toml

```toml
[project]
name = "business-plugin"
version = "0.1.0"
dependencies = ["opencv-python", "numpy"]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.backends._legacy:_Backend"
```

安裝：

```bash
# 開發時（editable install，修改即生效）
pip install -e /path/to/business_plugin

# 部署時
pip install business-plugin
```

---

## 4. Plugin 入口：`__init__.py`

```python
from business_plugin.foreground_mask.v1 import ForegroundMaskBlockV1

def load_blocks():
    """Server 啟動時呼叫，回傳所有 Block 類別。"""
    return [ForegroundMaskBlockV1]
```

Server 啟動時會呼叫 `load_blocks()` 取得 Block 類別列表，合併到可用 Block 池。

進階介面（可選）：
- `load_kinds()` — 自訂 Kind 類型
- `REGISTERED_INITIALIZERS` — Block 初始化參數注入
- `KINDS_SERIALIZERS` / `KINDS_DESERIALIZERS` — 自訂序列化

---

## 5. Block 開發

每個 Block 由兩個類別組成：**Manifest**（宣告輸入輸出）和 **Block**（實作邏輯）。

### 5.1 Manifest

```python
from typing import List, Literal, Optional, Type
from pydantic import ConfigDict, Field
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


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
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Original input image"
    )
    anomaly_map: Selector(kind=[IMAGE_KIND]) = Field(
        description="Anomaly heatmap from AD model"
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="image", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"
```

### 5.2 Block

```python
import cv2
import numpy as np
from inference.core.workflows.execution_engine.entities.base import (
    WorkflowImageData,
    ImageParentMetadata,
)


class ForegroundMaskBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ForegroundMaskBlockManifest

    def run(
        self,
        image: WorkflowImageData,
        anomaly_map: WorkflowImageData,
    ) -> BlockResult:
        orig_bgr = image.numpy_image           # (H, W, 3) uint8 BGR
        amap_bgr = anomaly_map.numpy_image     # (H, W, 3) uint8 grayscale in BGR

        # Business logic: extract foreground mask and apply to anomaly map
        gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
        _, fg_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        amap_gray = cv2.cvtColor(amap_bgr, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(amap_gray, amap_gray, mask=fg_mask)

        # Convert back to 3-channel for IMAGE_KIND compatibility
        result_bgr = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)

        result_image = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=result_bgr,
        )
        return {"image": result_image}
```

### 5.3 重點

| 項目 | 說明 |
|------|------|
| Block type 命名 | `{plugin_package_name}/{block_name}@v{version}` |
| `describe_outputs()` | 每個輸出的 `name` 必須與 `run()` 回傳 dict 的 key 完全一致 |
| `run()` 參數名稱 | 必須與 Manifest 中的 field 名稱完全一致 |
| IMAGE_KIND | numpy array `(H, W, 3)` uint8 BGR，包在 `WorkflowImageData` 裡 |
| FLOAT_KIND | Python `float` |

---

## 6. 需要 model_manager 的 Block

若 Block 需要呼叫模型推論（而不只是影像處理），可以注入 `model_manager`：

```python
class MyModelBlockV1(WorkflowBlock):
    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]
```

`model_manager`、`api_key`、`step_execution_mode` 由 server 的
`REGISTERED_INITIALIZERS` 自動注入，不需要手動設定。

---

## 7. 常用 Kind 類型

| Kind | Python 類型 | 說明 |
|------|-------------|------|
| `IMAGE_KIND` | `WorkflowImageData` | 影像（numpy BGR uint8） |
| `FLOAT_KIND` | `float` | 浮點數 |
| `FLOAT_ZERO_TO_ONE_KIND` | `float` | 0~1 之間的浮點數 |
| `INTEGER_KIND` | `int` | 整數 |
| `BOOLEAN_KIND` | `bool` | 布林值 |
| `STRING_KIND` | `str` | 字串 |
| `OBJECT_DETECTION_PREDICTION_KIND` | `sv.Detections` | 物件偵測結果 |
| `DICTIONARY_KIND` | `dict` | 字典 |
| `LIST_OF_VALUES_KIND` | `list` | 清單 |

---

## 8. Plugin 載入機制

Server 用環境變數 `WORKFLOWS_PLUGINS` 指定要載入的 plugin：

```bash
# 單個 plugin
WORKFLOWS_PLUGINS=business_plugin

# 多個 plugin（逗號分隔）
WORKFLOWS_PLUGINS=business_plugin,another_plugin
```

載入流程：
1. Server 啟動時讀取 `WORKFLOWS_PLUGINS` 環境變數
2. 對每個 plugin 名稱呼叫 `importlib.import_module()`
3. 呼叫 plugin 的 `load_blocks()` 取得 Block 類別列表
4. 合併到可用 Block 池中

**前提**：plugin 必須已 `pip install` 到 server 的 Python 環境。

---

## 9. 熱重載規則

| 變更類型 | 需要重啟 server？ |
|----------|-------------------|
| 修改 plugin 程式碼 | **是** |
| 修改 workflow JSON | 否（每次 POST 帶完整 spec） |
| 新增/刪除模型目錄 | 否（`add_model()` 動態載入） |
| 修改 `model_config.json` | **是**（模型載入時才讀取） |

Plugin 在 server 啟動時載入一次（`@lru_cache()`），修改後必須重啟 server。
