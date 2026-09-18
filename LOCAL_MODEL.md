# 本地模型与桌面端使用说明

本项目支持两种 LLM 提供方，通过 `.env` 里的一个开关切换，**代码无需改动**：

| 模式 | 说明 | 适用场景 |
| --- | --- | --- |
| `remote`（默认） | 调用云端 OpenAI 兼容 API（当前 DeepSeek） | 联网、无需本地算力 |
| `local` | 调用本机 OpenAI 兼容服务（Ollama / llama.cpp） | 离线、数据不出本机、CPU/核显 |

后端 `load_model.py` 统一用 `init_chat_model(..., model_provider="openai")`，
两种模式只是 `base_url` / 模型名不同，所以切换成本极低。

---

## 1. 打包成桌面应用（pywebview）

桌面端把 FastAPI 服务跑在后台线程，再用系统自带的 WebView2 内核开一个本地窗口，
无需浏览器、无需 HTTPS 证书。

```cmd
:: 首次安装依赖
pip install pywebview

:: 启动桌面端
python desktop.py
```

- 服务监听 `http://127.0.0.1:8000`（可用环境变量 `DESKTOP_PORT` 改端口）。
- 网页版仍可单独运行：`python app.py`。

> Windows 需装一次 [WebView2 Runtime](https://developer.microsoft.com/microsoft-edge/webview2/)（Win11 / 新版 Office 通常已内置）。

---

## 2. 切换到本地模型

### 2.1 安装并启动 Ollama（CPU 即可）

1. 下载安装 [Ollama for Windows](https://ollama.com/download)。
2. 拉取一个适合纯 CPU / 核显的小模型（代码向，量化版）：

```cmd
ollama pull qwen2.5-coder:1.5b
:: 若机器内存 ≥ 16GB、想要更好效果：
ollama pull qwen2.5-coder:3b
```

3. 验证 OpenAI 兼容接口已就绪（默认 `http://localhost:11434/v1`）：

```cmd
curl http://localhost:11434/v1/models
```

### 2.2 配置 `.env`

在 `.env` 中加入：

```ini
LLM_PROVIDER=local
LOCAL_MODEL=qwen2.5-coder:1.5b
LOCAL_BASE_URL=http://localhost:11434/v1
```

改回云端：`LLM_PROVIDER=remote`（或删除该行，默认即 remote）。

- `LOCAL_MODEL`：Ollama 的模型标签（`ollama list` 可查）。
- `LOCAL_BASE_URL`：换成 llama.cpp server 等其它 OpenAI 兼容服务时改这里。

然后正常启动 `python desktop.py` 或 `python app.py` 即可，前端无感知。

### 2.3 CPU-only 选型建议

| 模型 | 量化后体积 | 内存需求 | 效果 |
| --- | --- | --- | --- |
| `qwen2.5-coder:1.5b` (Q4) | ~1 GB | ~3–4 GB | 简单 pandas 提问够用，速度快 |
| `qwen2.5-coder:3b` (Q4) | ~2 GB | ~6–8 GB | 代码质量更稳，仍可在 CPU 跑 |
| `qwen2.5-coder:7b` (Q4) | ~4.7 GB | ~10–12 GB | 效果更好但 CPU 上明显变慢 |

CPU 推理速度以 token/秒计，7B 在无独显机器上通常偏慢，建议从 1.5b/3b 起步。

---

## 3. 微调路径（后续）

目标是最终用**本地微调小模型**替换通用模型，专门擅长「自然语言 → pandas 代码」。

### 3.1 语料已经在自动积累

`agent.py` 每次生成代码后，会把 `(prompt, code, ok, err)` 追加写入
`data/codegen_log.jsonl`（成功样本 `ok=true` 是主要训练语料，报错样本可用于对比/纠错）。
正常使用应用即可收集数据，`data/` 已被 `.gitignore` 忽略。

### 3.2 从日志到训练集

1. 收集一段时间后，导出 `data/codegen_log.jsonl`，筛选 `ok==true` 的样本。
2. 转换成指令格式（`instruction= prompt, output= code`）。
3. 用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 做 **LoRA / QLoRA** 微调
   （QLoRA 量化微调对显存更友好；无 GPU 时也可用云端/租卡跑一次训练）。

### 3.3 部署回本地

1. 微调得到的 LoRA 权重先 **merge** 进基座模型。
2. 用 `llama.cpp` 的 `convert_hf_to_gguf.py` + `quantize` 导出 **GGUF（Q4_K_M）**。
3. 导入 Ollama（Modelfile `FROM ./xxx.Q4_K_M.gguf`），
   再把 `.env` 的 `LOCAL_MODEL` 指向它即可，无需改代码。

---

## 4. 相关环境变量一览

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `LLM_PROVIDER` | `remote` | `remote` / `local` |
| `LOCAL_MODEL` | `qwen2.5-coder:1.5b` | 本地模型标签 |
| `LOCAL_BASE_URL` | `http://localhost:11434/v1` | 本地 OpenAI 兼容地址 |
| `DESKTOP_PORT` | `8000` | 桌面端内嵌服务端口 |
| `TRAIN_DATA_DIR` | `data` | 微调语料 JSONL 目录 |
