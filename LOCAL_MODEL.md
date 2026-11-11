# 远程模型与桌面客户端

## 桌面发行版配置

桌面客户端在本机处理工作簿，并直接通过 HTTPS 调用 OpenAI 兼容的远程模型 API；不需要部署本项目的远程后端。上传的文件和提问会发送到模型服务商，发布者应事先披露服务商、数据处理方式和隐私政策。

当前选定的发行方式会把模型 API Key 写入安装包中的 `llm_config.json`。该文件和安装程序都不能保护密钥，用户可以提取并滥用它。只使用专用于该应用、设置了严格额度和速率限制的 Key。不要提交真实 Key、将其粘贴到聊天中，或打印到日志。

## 配置与构建

可在构建机设置 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 和可选的 `DEEPSEEK_MODEL` 环境变量，或将它们放在项目根目录未提交的 `.env` 文件中。发布构建需要 API Key 和 HTTPS 地址：

```ini
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

```powershell
python -m pip install -r requirements.txt -r requirements-build.txt
.\packaging\build_windows.ps1 -PythonPath "D:\path\to\python.exe"
```

`packaging/write_model_config.py` 在打包过程中验证配置并生成 `dist\ExcelAssistant\llm_config.json`。密钥不应被加入源码控制；`.gitignore` 忽略生成的模型配置文件。构建完成后，`release/` 中的安装包包含相同的密钥。

构建需要 64 位 Python 3.12、项目依赖、PyInstaller 和 Inno Setup 6。安装包为 64 位 Windows 用户级安装程序，默认不需要管理员权限。目标电脑须有 Microsoft Edge WebView2 Runtime；Windows 11 通常已预装。公开发布还需代码签名证书；没有签名时 Windows SmartScreen 可能显示未知发布者警告。

## 本地模型（开发/实验）

开发时可以在 `.env` 中设置 `LLM_PROVIDER=local` 连接 Ollama 或其他 OpenAI 兼容服务。本地模型及权重不会打包进安装程序。

```ini
LLM_PROVIDER=local
LOCAL_MODEL=qwen2.5-coder:1.5b
LOCAL_BASE_URL=http://localhost:11434/v1
```

本地开发使用远程模型时，`LLM_PROVIDER=remote`（默认）并配置 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`。旧式 `deepseek_api` JSON 配置仍可用于开发兼容，但新发行包使用上文所述的专用 `llm_config.json`。

### llama-server 启动参数（重要）

用 llama.cpp 部署微调模型时，**必须显式中和采样器**：

```bat
llama-server.exe -m models\excel-assistant-dpo2-q4km.gguf ^
  --host 127.0.0.1 --port 8080 -c 8192 --jinja -np 1 ^
  --temp 0 --top-k 1 --top-p 1.0 --min-p 0.0 --repeat-penalty 1.0
```

原因：llama-server 的默认值是
`temperature 0.7 / top_k 20 / top_p 0.8 / min_p 0.05 / repeat_penalty 1.1`。
即使客户端在请求里传 `temperature=0`，其余采样器仍按默认值生效。
`repeat_penalty 1.1` 尤其实在有影响：工具调用会反复输出
`"name"`、`"arguments"`、`aggregate_data` 这类相同短语，penalty 会压低这些
token 的概率，导致模型改口、跳步或陷入重复调用。

微调模型是按**贪心解码**训练的对照目标（评测侧 transformers 用
`do_sample=False`），所以部署侧也要对齐到纯贪心。

### 部署与评测的分数差异

同一份权重在 `transformers`（bf16、贪心）与 `llama-server`（bf16 GGUF）上
分数不同，20 条样本实测：transformers 17/20，llama-server 12–14/20。

排查过程与结论（每一步都做了字节级比对）：

| 检查项 | 结果 |
| --- | --- |
| tools schema 是否一致 | 曾经不一致（5,704 vs 4,558 字符），已统一到 `tool_schema.py` |
| GGUF 内嵌 chat template | 与 HF 逐字节相同（2507 字符） |
| 分词 | `/tokenize` 与 HF `input_ids` 完全相同（2130 tokens） |
| 贪心生成 | 同一 prompt 两边输出逐字符相同 |
| 采样器默认值 | **默认值偏离贪心，必须显式中和** |

中和采样器后仍有 1–3 条的差距，且同配置重复测量本身波动 1–2 条。
因此这属于**小样本测量噪声 + 后端数值差异**，不是可以通过改代码消除的
格式 bug。正式对比时请用 ≥200 条样本，或对同配置重复测量取中位数。

## 数据与许可

- 上传的文件和分析结果保存在 `%LOCALAPPDATA%\ExcelAssistant`。
- 桌面发行版会禁用训练样本日志。开发服务器日志默认关闭；开启 `TRAIN_DATA_LOGGING=1` 可能记录提示词和表格样例。
- 软件仅限个人、非商业使用。允许转发官方发布的未修改安装包，不允许修改或发布衍生版本，详见 [LICENSE.txt](./LICENSE.txt)。
