# Inference Server App 開發指南

本文件供 **C# app 開發者**使用，說明如何開發推論 app：對**workflow** 送出推論，並接收與處理回應。

---

## 1. 架構：app / server / workflow

| 角色 | 職責 |
|---|---|
| **Workflow**（一份 JSON） | 定義整個推論流程的規格：要哪些輸入、跑哪些步驟、回哪些輸出。是「設定與邏輯的本體」。 |
| **App** | 推論終端app。取得 workflow → 解析它要什麼 → 蒐集輸入值 → 送伺服器 → 接收並呈現輸出。**不定義邏輯。** |
| **Server** | 推論伺服器。收到 workflow + 輸入即時執行，回傳輸出。 |

---

## 2. API 參考

### 2.1 `GET /info` — 連線與版本
```
GET {server}/info
return { "name": "...", "version": "...", "uuid": "..." }
```

### 2.2 `POST /workflows/describe_interface` — 查詢 workflow 的輸入與輸出
輸入 workflow，回傳每個輸入/輸出的**名稱與型別**，以及各型別的結構。
```
POST {server}/workflows/describe_interface
{ "specification": { ...整份 workflow... }, "api_key": "" }     // api_key 欄位可為空字串
return {
  "inputs":  { "<輸入名>": ["<型別>"], ... },
  "outputs": { "<輸出名>": ["<型別>"], ... },
  "typing_hints":  { "<型別>": "<序列化後對應的型別>", ... },   // 如 "image":"dict","string":"str","float":"float","*":"Any"
  "kinds_schemas": { "<型別>": { ...OpenAPI 3.0 結構... }, ... }   // 有結構的型別（如 image）才會出現
}
```
> 範例（對某 TASC workflow 的實際回應，節錄）：
> `outputs`: `{ "verdict":["string"], "die_score":["float"], "details":["*"], "overlay_ls1":["image"], ... }`
> `inputs`: `{ "image_ls1":["image"], "model_id_ls1":["roboflow_model_id"], "binarize_threshold":["*"], ... }`

型別一覽（出現在 inputs/outputs）：`image`、`string`、`integer`、`float`、`boolean`、`object_detection_prediction`（及其他預測類）、`list`、`dictionary`、`*`（Any，自訂區塊的自由格式輸出）。各型別的 JSON 形狀見 §3（輸入）與 §4（輸出）。

### 2.3 `GET /models/local` — 列出伺服器上可用的模型
```
GET {server}/models/local
return { "configured": true, "root": "...",
    "models": [ { "name":"...", "model_id":"<伺服器端路徑>", "task_type":"...", "model_architecture":"..." }, ... ] }
```
- 回傳的 `model_id` 即為填入模型輸入的字串值。
- 若 `configured` 為 `false`（伺服器未設定本地模型目錄），`models` 為空陣列；此時由使用者手動輸入模型路徑。

### 2.4 `POST /workflows/run` — 執行推論
```
POST {server}/workflows/run
{ "specification": { ...整份 workflow... },
  "inputs": { "<輸入名>": <值>, ... } }
return { "outputs": [ { "<輸出名>": <值>, ... }, ... ] }
```
- `inputs` 的 key 為 workflow 宣告的輸入名；值的形狀依該輸入的型別。
- 回應 `outputs` 為**陣列**：每個 batch 元素一筆；若 workflow 把整批合併為單一結果，則只有 `outputs[0]`。每筆是「輸出名 → 值」的物件。

### 2.5 `POST /model/add` — 預先載入模型（Optional）
```
POST {server}/model/add
{ "model_id": "<伺服器端路徑>" }
return 200 + 已載入模型清單（路徑錯誤則回非 200）
```
選用暖機，避免首次推論因載入而較慢。非必要：執行推論時模型會自動載入。



### 2.6 `GET /workflows/blocks/describe` — 各 block 的欄位 schema （Optional）
回傳所有可用 block 的定義，含每個 block 的輸入/輸出欄位 schema（型別、`description` 等）。用來查內建 block 某欄位的語意（如模型區塊 `model_id` 欄位的 description）。
```
GET {server}/workflows/blocks/describe
return { "blocks": [
  { "manifest_type_identifier": "roboflow_core/aie_anomaly_detection_model@v1",
    "block_schema": { "properties": {
      "model_id": { "description": "Local path to AIE anomaly detection model directory", ... },
      "images":   { "description": "The image to infer on.", ... } } } },
  ... ], ... }
```

---

## 3. 輸入解析

透過API ：`POST /workflows/describe_interface`（§2.2）獲得每個輸入的名稱與型別。

### 3.1 依型別提供值
`describe_interface` 的 `inputs` 是「名稱 → 型別」。各型別與要送出的值：

| 型別 | 送出的值 |
|---|---|
| `image` | 影像物件 `{ "type": "<來源>", "value": <值> }`（`type` 選項見下表） |
| `roboflow_model_id` | 模型 id 字串：本地模型路徑（由 `GET /models/local` 取得）|
| `string` | 字串 |
| `integer` | 整數 |
| `float` | 浮點數 |
| `boolean` | 布林（`true` / `false`） |
| `list` / `dictionary` | JSON 陣列／物件 |
| `*`（Any） | 任意 JSON 值，依該參數實際用途 |

**影像物件 `type` 的完整選項**：

| `type` | `value` 內容 |
|---|---|
| `base64` | 影像檔位元組的 base64 字串 |
| `url` | 圖片的 http(s) 網址字串 |
| `file` | 影像檔在**伺服器**那台機器上的絕對路徑（由 server 端讀檔） |

> 影像格式：`base64` 與 `file` 皆由 OpenCV（`cv2.imdecode` / `cv2.imread`）解碼，支援 JPEG、PNG、BMP、WEBP、TIFF、JPEG2000 等常見格式。

### 3.2 必填/可選與預設值
`describe_interface`（§2.2）只回名稱與型別，**不會**告訴你某輸入是必填或可選、預設值是多少。這兩項要看 workflow JSON 的 `inputs[]`：每個輸入項若帶 `default_value` 即為**可選**（不送就用該預設），沒有則為**必填**。可選參數若不需調整，可直接省略不送。

### 3.3 各類輸入的值從哪裡來

| 型別 | 值的來源 | 做法 |
|---|---|---|
| `image` | app 端| 讀檔轉 base64，包成 `{type:"base64", value:...}` |
| `roboflow_model_id`（模型） | 使用者選擇 | 本地模型路徑由 `GET /models/local`（§2.3）清單選 |
| `string` / 其他純量 / `list` | `default_value` 或使用者輸入 | 有預設帶入、可改；無預設則必填 |

> app 依 `describe_interface` 動態產生 UI：型別 `image` 給檔案選擇器、`roboflow_model_id` 接 `/models/local` 下拉、其餘給輸入框。數量與名稱由 workflow 決定。

---

## 4. 輸出：如何得知會收到什麼、如何處理每一種

### 4.1 回應整體結構
```
return { "outputs": [ { "<輸出名>": <值>, ... }, ... ] }
```
- `outputs` 為陣列，每個 batch 元素一筆（合併型 workflow 僅 `outputs[0]`）。
- 每筆的 key 即輸出名稱；**所有輸出名稱與其型別，由 `describe_interface`（§2.2）取得**。

### 4.2 各型別的值結構與 C# 接收方式

**影像（型別 `image`）** — 如熱力圖、疊圖、裁切圖：
```json
{ "type": "base64", "value": "<base64 影像>", "video_metadata": { ... 或 null } }
```
`value` 解碼後是 **JPEG** 影像位元組（server 一律以 JPEG／quality 95 編碼，與來源影像格式無關）；要存檔請用 `.jpg`。`video_metadata` 為附帶資訊（可能為 `null` 或一個 metadata 物件），顯示影像時可忽略。解碼只需 `value`：
```csharp
byte[] bytes = Convert.FromBase64String(value["value"]!.GetValue<string>());
```

**物件偵測（型別 `object_detection_prediction` 等）**：
```json
{
  "image": { "width": 1448, "height": 610 },
  "predictions": [
    { "x": 720.5, "y": 300.0, "width": 64.0, "height": 48.0,
      "confidence": 0.91, "class": "particle", "class_id": 0, "detection_id": "<uuid>" }
  ]
}
```
> **`x`、`y` 是框的「中心」座標**，不是左上角；左上角 = `(x - width/2, y - height/2)`。
> 實例分割另含 `points`（多邊形 `[{x,y},…]`）；關鍵點另含 `keypoints`。

**純量（型別 `string`/`integer`/`float`/`boolean`）**：直接是對應的 JSON 值（`"OK"`、`3`、`0.267`、`true`）。

**自由格式（型別 `*`/Any，或 `list`/`dictionary`）** — 自訂區塊的輸出：序列化後是原樣的 JSON 物件/陣列，其中若內含影像或偵測，會以上述格式一併序列化。

### 4.3 通用呈現：以遞迴處理自由格式輸出

自由格式（Any）輸出的內部欄位事先未宣告，以遞迴呈現處理：
- 影像（含 `type:"base64"`）→ 解碼顯示。
- 物件 → 遞迴展開每個 key。
- 陣列 → 遞迴展開每個元素。
- 其餘（字串/數字/布林）→ 以文字顯示。

例如 `details` = `[{ "lightset":"LightSet1", "n_contours":0, "pred":"OK" }, …]`，遞迴後呈現 `details[0].lightset = LightSet1`、`details[0].n_contours = 0`…。

若 app 是為特定 workflow 而做，亦可直接依輸出名稱取用欄位做進階呈現（如把 `pred` 顯示成徽章、把偵測畫成框）。

---

## 5. 端到端範例

app 實作兩個介面：`IInputProvider`（提供輸入值）與 `IOutputSink`（呈現輸出）。`InferenceClient` 先呼叫 `describe_interface` 取得輸入/輸出的名稱與型別，再依型別組裝輸入、送出、逐欄位呈現輸出。

```csharp
using System.Text;
using System.Text.Json.Nodes;

// app 提供輸入值；依「型別」決定怎麼給（型別由 server 的 describe_interface 告知）
public interface IInputProvider
{
    string ProvideImage(string name);                                       // 型別 image：回影像檔路徑
    JsonNode? ProvideValue(string name, string kind, JsonNode? defaultValue); // 其他型別：回值；可選輸入(有 default)可回 null 省略，必填輸入(無 default)必須回值
}

// app 呈現輸出
public interface IOutputSink
{
    void Image(string path, byte[] image);   // path 例：「[0].overlay_ls1」
    void Text(string path, string text);      // path 例：「[0].details[0].pred」
}

public static class InferenceClient
{
    public static async Task RunAsync(string server, JsonNode workflow,
                                      IInputProvider input, IOutputSink output)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };

        async Task<JsonNode> PostAsync(string path, JsonNode body)
        {
            var resp = await http.PostAsync($"{server}{path}",
                new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"));
            var text = await resp.Content.ReadAsStringAsync();
            if (!resp.IsSuccessStatusCode)
                throw new HttpRequestException($"HTTP {(int)resp.StatusCode}: {text}");
            return JsonNode.Parse(text)!;
        }

        // (1) 呼叫 describe_interface 取得每個輸入/輸出的名稱與型別
        var di = await PostAsync("/workflows/describe_interface",
            new JsonObject { ["specification"] = workflow.DeepClone(), ["api_key"] = "" });
        var diInputs = di["inputs"]!.AsObject();

        // (2) 從 workflow 的 inputs[] 讀取各輸入的預設值
        var defaults = new Dictionary<string, JsonNode?>();
        foreach (var inp in workflow["inputs"]!.AsArray())
            if (inp!["default_value"] is { } dv)
                defaults[inp["name"]!.GetValue<string>()] = dv.DeepClone();

        // (3) 依型別提供每個輸入的值
        var inputs = new JsonObject();
        foreach (var kv in diInputs)
        {
            var name = kv.Key;
            var kind = kv.Value!.AsArray()[0]!.GetValue<string>();
            if (kind == "image")
                inputs[name] = new JsonObject { ["type"] = "base64",
                    ["value"] = Convert.ToBase64String(File.ReadAllBytes(input.ProvideImage(name))) };
            else
            {
                defaults.TryGetValue(name, out var def);
                var v = input.ProvideValue(name, kind, def);
                if (v is not null) inputs[name] = v;
            }
        }

        // (4) 執行推論
        var result = await PostAsync("/workflows/run",
            new JsonObject { ["specification"] = workflow.DeepClone(), ["inputs"] = inputs });

        // (5) 依輸出型別逐欄位呈現
        var outs = result["outputs"]!.AsArray();
        for (int i = 0; i < outs.Count; i++)
            foreach (var field in outs[i]!.AsObject())
                Render($"[{i}].{field.Key}", field.Value, output);
    }

    // 影像→顯示；物件/陣列→遞迴展開；其餘→文字
    static void Render(string path, JsonNode? value, IOutputSink sink)
    {
        if (value is JsonObject obj)
        {
            if (obj["type"]?.GetValue<string>() == "base64" && obj["value"]?.GetValue<string>() is { } b64)
                { sink.Image(path, Convert.FromBase64String(b64)); return; }
            foreach (var kv in obj) Render($"{path}.{kv.Key}", kv.Value, sink);
        }
        else if (value is JsonArray arr)
            for (int i = 0; i < arr.Count; i++) Render($"{path}[{i}]", arr[i], sink);
        else
            sink.Text(path, value?.ToString() ?? "null");
    }
}
```

呼叫端：執行時取得 workflow，提供輸入/輸出實作：
```csharp
var workflow = JsonNode.Parse(workflowJson)!;   // workflowJson 由操作員選檔或後端推送取得
await InferenceClient.RunAsync("http://SERVER:9001", workflow, new MyInput(), new MyOutput());
```
非影像輸入由 `IInputProvider.ProvideValue` 提供，它收到 `kind`：當 `kind == "roboflow_model_id"` 即為模型輸入，app 可用 `/models/local`（§2.3）清單供選；其餘照型別給輸入框。要對特定 workflow 做進階呈現（畫框、徽章…），在 `IOutputSink` 實作裡依欄位 `path` 客製。
