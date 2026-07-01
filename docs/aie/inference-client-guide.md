# Inference Server App 開發指南

本文件供 **C# app 開發者**使用，說明如何開發推論 app：對**執行時才取得、結構事先未知的 workflow** 送出推論，並接收與處理回應。文中的具體 workflow（TASC）僅為範例。

---

## 1. 架構：app / server / workflow 的職責

| 角色 | 職責 |
|---|---|
| **Workflow**（一份 JSON） | 定義整個推論流程：要哪些輸入、跑哪些步驟、回哪些輸出。是「設定與邏輯的本體」。 |
| **App** | 取得 workflow → 解析它要什麼 → 蒐集輸入值 → 送伺服器 → 接收並呈現輸出。**不定義邏輯。** |
| **Server** | 無狀態。收到 workflow + 輸入即時執行，回傳輸出，不保存任何設定。 |

- app 每次請求都帶**整份 workflow** 與輸入；換 workflow 不需改 server。
- workflow 裡的 `model_id` 是一個**伺服器端路徑字串**——app 不上傳模型，只送路徑（路徑哪裡來見 §3）。

---

## 2. API 參考

所有端點皆為 HTTP/JSON。基底位址記為 `{server}`（如 `http://10.0.0.5:9001`）。

### 2.1 `GET /info` — 連線與版本
```
GET {server}/info
→ { "name": "...", "version": "...", "uuid": "..." }
```

### 2.2 `POST /workflows/describe_interface` — 查詢 workflow 的輸入與輸出
把整份 workflow 丟進去，回傳每個輸入/輸出的**名稱與型別**，以及各型別的結構。這是得知「要送什麼、會收什麼」最直接的方式。
```
POST {server}/workflows/describe_interface
{ "specification": { ...整份 workflow... }, "api_key": "" }     // api_key 欄位可為空字串
→ {
  "inputs":  { "<輸入名>": ["<型別>"], ... },
  "outputs": { "<輸出名>": ["<型別>"], ... },
  "typing_hints":  { "<型別>": "<序列化後對應的型別>", ... },   // 如 "image":"dict","string":"str","float":"float","*":"Any"
  "kinds_schemas": { "<型別>": { ...OpenAPI 3.0 結構... }, ... }   // 有結構的型別（如 image）才會出現
}
```
> 範例（對某 TASC workflow 的實際回應，節錄）：
> `outputs`: `{ "verdict":["string"], "die_score":["float"], "details":["*"], "overlay_ls1":["image"], ... }`
> `inputs`: `{ "image_ls1":["image"], "model_id_ls1":["string"], "binarize_threshold":["*"], ... }`

型別一覽（出現在 inputs/outputs）：`image`、`string`、`integer`、`float`、`boolean`、`object_detection_prediction`（及其他預測類）、`list`、`dictionary`、`*`（Any，自訂區塊的自由格式輸出）。各型別的 JSON 形狀見 §3（輸入）與 §4（輸出）。

### 2.3 `GET /models/local` — 列出伺服器上可用的模型
```
GET {server}/models/local
→ { "configured": true, "root": "...",
    "models": [ { "name":"...", "model_id":"<伺服器端路徑>", "task_type":"...", "model_architecture":"..." }, ... ] }
```
- 回傳的 `model_id` 即為填入模型輸入的字串值。
- 若 `configured` 為 `false`（伺服器未設定本地模型目錄），`models` 為空陣列；此時由使用者手動輸入模型路徑。

### 2.4 `POST /model/add` — 預先載入模型（選用）
```
POST {server}/model/add
{ "model_id": "<伺服器端路徑>" }
→ 200 + 已載入模型清單（路徑錯誤則回非 200）
```
選用暖機，避免首次推論因載入而較慢。非必要：執行推論時模型會自動載入。

### 2.5 `POST /workflows/run` — 執行推論
```
POST {server}/workflows/run
{ "specification": { ...整份 workflow... },
  "inputs": { "<輸入名>": <值>, ... } }
→ { "outputs": [ { "<輸出名>": <值>, ... }, ... ] }
```
- `inputs` 的 key 為 workflow 宣告的輸入名；值的形狀依該輸入的型別（見 §3）。
- 回應 `outputs` 為**陣列**：每個 batch 元素一筆；若 workflow 把整批合併為單一結果，則只有 `outputs[0]`。每筆是「輸出名 → 值」的物件（見 §4）。

---

## 3. 輸入：如何得知要送什麼、值從哪裡來

一份 workflow 的輸入可由 `describe_interface`（§2.2）取得名稱與型別，或直接解析 workflow JSON 的 `inputs[]`。
**直接解析 JSON 能多得到兩項資訊**：參數的預設值（`default_value`），以及「哪個輸入是模型路徑」（須看 `steps[]`），故下面以直接解析說明。

### 3.1 解析 `inputs[]` — 取得每個輸入
`inputs[]` 每一項都有 `type` 與 `name`：

| `type`（含舊別名） | 類別 | 送出時的值 |
|---|---|---|
| `WorkflowImage` / `InferenceImage` | 影像 | `{ "type":"base64", "value":"<base64 影像>" }`（`type` 亦可為 `url`/`numpy`） |
| `WorkflowParameter` / `InferenceParameter` | 參數 | 一個 JSON 值（數字／字串／布林／陣列／物件） |
| `WorkflowBatchInput` | 批次 | 一個陣列，每個 batch 元素一筆 |

必填/可選（僅參數）：`WorkflowParameter` 有 `default_value`（非 null）為**可選**（不送即用預設）；無則**必填**。影像、批次一律必填。

### 3.2 找出「哪些輸入是模型路徑」 — 解析 `steps[]`
模型輸入在 `inputs[]` 裡只是普通字串參數，無法單看 `inputs[]` 區分。要靠 `steps[]`：凡某步驟有 `"model_id": "$inputs.<X>"`，則輸入 `<X>` 就是模型路徑。
```
模型輸入 = { steps[].model_id 形如 "$inputs.X" 的那些 X }
```

### 3.3 各類輸入的值從哪裡來

| 類別 | 值的來源 | 做法 |
|---|---|---|
| 影像 | app 端（檔案、相機…） | 讀檔轉 base64，包成 `{type:"base64", value:...}` |
| 模型路徑（§3.2 找出的） | 伺服器 `GET /models/local` 的 `model_id` 清單 | 做成下拉讓使用者選；清單為空則手動輸入 |
| 一般參數 | `default_value` 或使用者輸入 | 有預設帶入欄位、可改；無預設則必填 |

> app 依解析結果動態產生 UI：每個影像輸入一個檔案選擇器、模型輸入給下拉、其餘參數給輸入框。數量與名稱由 workflow 決定。

---

## 4. 輸出：如何得知會收到什麼、如何處理每一種

### 4.1 回應整體結構
```
→ { "outputs": [ { "<輸出名>": <值>, ... }, ... ] }
```
- `outputs` 為陣列，每個 batch 元素一筆（合併型 workflow 僅 `outputs[0]`）。
- 每筆的 key 即輸出名稱；**所有輸出名稱與其型別，由 `describe_interface`（§2.2）取得**。

### 4.2 各型別的值結構與 C# 接收方式

**影像（型別 `image`）** — 如熱力圖、疊圖、裁切圖：
```json
{ "type": "base64", "value": "<base64 影像>", "video_metadata": { ... 或 null } }
```
`video_metadata` 為附帶資訊（可能為 `null` 或一個 metadata 物件）；解碼影像只需 `value`：
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

## 5. 端到端範例（可編譯，適用任何 workflow）

app 實作兩個介面：`IInputProvider`（提供輸入值）與 `IOutputSink`（呈現輸出）。`InferenceClient` 解析 workflow、組裝輸入、送出、再逐欄位呈現輸出。數量、名稱、結構皆由 workflow 決定。

```csharp
using System.Text;
using System.Text.Json.Nodes;

// app 提供輸入值（影像、模型、參數）
public interface IInputProvider
{
    string PickImage(string inputName);                                // 回傳影像檔路徑
    string PickModel(string inputName, IReadOnlyList<string> options); // 從 options(來自 /models/local)選一個
    JsonNode? GetParameter(string inputName, JsonNode? defaultValue);  // 回傳值；回 null 表示沿用預設
}

// app 呈現輸出（任何欄位最終都會落到這兩個方法之一）
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

        // (1) 解析 steps[]：哪些輸入是模型路徑
        var modelInputs = new HashSet<string>();
        foreach (var step in workflow["steps"]!.AsArray())
            if (step!["model_id"]?.GetValue<string>() is { } sel && sel.StartsWith("$inputs."))
                modelInputs.Add(sel["$inputs.".Length..]);

        // 模型可選清單（伺服器提供）
        var models = JsonNode.Parse(await http.GetStringAsync($"{server}/models/local"))!
            ["models"]!.AsArray().Select(m => m!["model_id"]!.GetValue<string>()).ToList();

        // (2) 解析 inputs[]：逐項取值（數量、名稱由 workflow 決定）
        var inputs = new JsonObject();
        foreach (var inp in workflow["inputs"]!.AsArray())
        {
            var name = inp!["name"]!.GetValue<string>();
            var type = inp["type"]!.GetValue<string>();
            if (type is "WorkflowImage" or "InferenceImage")
                inputs[name] = new JsonObject { ["type"] = "base64",
                    ["value"] = Convert.ToBase64String(File.ReadAllBytes(input.PickImage(name))) };
            else if (modelInputs.Contains(name))
                inputs[name] = input.PickModel(name, models);
            else  // 一般參數 / 批次
            {
                var v = input.GetParameter(name, inp["default_value"]?.DeepClone());
                if (v is not null) inputs[name] = v;
            }
        }

        // (3) 送出
        var payload = new JsonObject { ["specification"] = workflow.DeepClone(), ["inputs"] = inputs };
        var resp = await http.PostAsync($"{server}/workflows/run",
            new StringContent(payload.ToJsonString(), Encoding.UTF8, "application/json"));
        var result = JsonNode.Parse(await resp.Content.ReadAsStringAsync())!;

        // (4) 逐筆、逐欄位呈現輸出
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

要對特定 workflow 做進階呈現（畫框、徽章…），在 `IOutputSink` 實作裡依欄位 `path` 客製。
