# 本地小模型（Qwen2.5-1.5B + llama.cpp）

目标：在普通用户电脑上离线运行模型，不依赖远程 API。当前选定的基座是
**Qwen2.5-1.5B-Instruct**，推理运行时为 **llama.cpp / GGUF**。

## 为什么这个任务是可行的

Agent 每轮只做一件事：从 9 个固定工具中选一个、填参数。不需要一次规划整条链。
这比"通用 Excel 问答"窄得多，是 1.5B 量级模型有机会做好的关键前提。

已知的弱项与对策：

| 弱项 | 对策 |
| --- | --- |
| 输出格式不合法（缺字段、多写散文） | **约束解码**（GBNF / JSON Schema），格式错误直接归零 |
| 列名拼写不一致（"销售额" vs "销售额合计"） | 程序侧做精确→去空格→近似匹配兜底；这是微调能改善的重点 |
| 汇总后仍引用原始列名 | system prompt 中明确汇总列名规则 + 微调 |
| 歧义时直接猜而不澄清 | 需要专门的澄清样本微调 |
| 复杂语义理解 | 不指望，超出工具能力时明确说明无法完成 |

## 启动本地服务

```powershell
# 1. 获取 GGUF 量化模型（Q4_K_M 约 1 GB 内存占用）
#    从 Hugging Face 下载 Qwen2.5-1.5B-Instruct 的 GGUF 版本

# 2. 启动 OpenAI 兼容服务
llama-server -m .\models\qwen2.5-1.5b-instruct-q4_k_m.gguf `
  --host 127.0.0.1 --port 8080 `
  -c 8192 --jinja
```

约束解码：llama.cpp 的 `llama-server` 支持 GBNF 语法约束（`--grammar` / `--grammar-file`）
与 JSON Schema 约束（`--json-schema`，或请求体里的 `response_format`）。参数名与支持范围
随版本变化，请以本机 `llama-server --help` 为准。启用后，模型输出必定是可解析的
JSON，这对 1.5B 模型是刚需，而不是可选项。

## 模型文件分发

安装包**不包含**模型文件（否则会增大到约 1.1 GB）。首次使用时由应用下载：

1. 启动后界面顶部显示下载提示与预计大小
2. 点击「开始下载」后显示实时进度条
3. 下载完成自动校验 SHA-256，通过后自动启动本地服务
4. 之后每次启动直接可用，不需要再下载

模型保存在 `%LOCALAPPDATA%\ExcelAssistant\models\`，卸载程序不会删除它，重装后无需重新下载。

实现细节：

| 行为 | 说明 |
| --- | --- |
| 断点续传 | 下载写入 `.part` 文件，中断后用 HTTP Range 继续 |
| 完整性校验 | 校验通过才重命名为正式文件；失败会删除并报错 |
| 大小预检 | 下载前检查磁盘空间，留 256 MB 余量 |
| 失败恢复 | 网络中断可在界面点「重试下载」，已下载部分会保留 |
| 校验缺失 | 未配置 `LOCAL_MODEL_SHA256` 时只做大小检查，构建时会打印警告 |

## 自动启动本地服务

打包后的程序可以在启动时自己拉起 `llama-server`，退出时关闭它，普通用户不需要开命令行。
在构建配置中加上以下几项：

```ini
LLM_PROVIDER=local
LOCAL_BASE_URL=http://127.0.0.1:8080/v1
LOCAL_MODEL=qwen2.5-1.5b-instruct
LOCAL_AUTO_START=1
LOCAL_SERVER_PATH=llama\llama-server.exe
# 二选一：随包分发模型，或首次启动下载
LOCAL_MODEL_PATH=models\model.gguf
LOCAL_MODEL_URL=https://models.example.com/qwen2.5-1.5b-instruct-q4_k_m.gguf
LOCAL_MODEL_SHA256=<文件 SHA-256>
LOCAL_MODEL_SIZE=<字节数>
```

相对路径会按安装目录解析。可选字段：

| 配置项 | 构建环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `auto_start` | `LOCAL_AUTO_START` | `false` | 是否由应用拉起本地服务 |
| `local_server` | `LOCAL_SERVER_PATH` | — | `llama-server` 可执行文件路径 |
| `local_model_path` | `LOCAL_MODEL_PATH` | — | 随包分发的 GGUF 路径 |
| `model_url` | `LOCAL_MODEL_URL` | — | 首次启动下载地址（必须 HTTPS） |
| `model_sha256` | `LOCAL_MODEL_SHA256` | — | 下载后的完整性校验值 |
| `model_size` | `LOCAL_MODEL_SIZE` | — | 文件字节数，用于进度与磁盘预检 |
| `model_filename` | `LOCAL_MODEL_FILENAME` | 从 URL 推断 | 保存到磁盘的文件名 |
| `context_size` | `LOCAL_CONTEXT_SIZE` | `8192` | 传给 `-c` 的上下文长度 |
| `startup_timeout` | `LOCAL_STARTUP_TIMEOUT` | `90` | 等待服务就绪的秒数 |
| `extra_args` | `LOCAL_EXTRA_ARGS` | — | 追加参数，如 `--threads,4` |

启用 `LOCAL_AUTO_START` 时必须给出 `LOCAL_SERVER_PATH` 以及模型来源之一
（`LOCAL_MODEL_PATH` 或 `LOCAL_MODEL_URL`），否则构建直接失败，避免产出一个启动后
才发现缺文件的安装包。

### 运行时的行为

| 情况 | 行为 |
| --- | --- |
| 目标地址已有可用服务 | 直接复用，不重复启动；应用退出时也不会关闭它 |
| 模型未下载且配置了下载地址 | 界面显示下载横幅，不弹错误 |
| 模型未下载且无下载地址 | 报错并指出缺失的模型路径 |
| 端口在监听但不是模型服务 | 报错提示端口被占用，建议更换端口或关闭占用程序 |
| 程序或模型文件缺失 | 报错并指出具体缺失路径 |
| 启动后立即退出 | 提示退出码，并指向日志文件 |
| 超过 `startup_timeout` 仍未就绪 | 提示可调大超时，并指向日志文件 |
| 上述任何失败 | 弹出警告后仍打开界面，聊天时返回同一原因，便于修复后重试 |

启动日志写入 `%LOCALAPPDATA%\ExcelAssistant\logs\local-model.log`（超过 2 MB 会在下次启动时重建）。
子进程以无控制台窗口方式启动，退出时先正常终止，10 秒内未退出则强制结束。

**性能提示：** 健康检查先做 0.3 秒的 TCP 探测，确认端口在监听后才发 HTTP 请求。
这不是微优化——在部分 Windows 环境里，连接已关闭的本地端口不会立即被拒绝，
而是等满超时；缺少这一步时每次健康检查要慢 4 秒。

## 构建本地模型版安装包

在构建环境或未提交的 `.env` 中声明本地提供方与模型来源：

```ini
LLM_PROVIDER=local
LOCAL_BASE_URL=http://127.0.0.1:8080/v1
LOCAL_MODEL=qwen2.5-1.5b-instruct
LOCAL_AUTO_START=1
LOCAL_SERVER_PATH=llama\llama-server.exe
LOCAL_MODEL_URL=https://models.example.com/qwen2.5-1.5b-instruct-q4_k_m.gguf
LOCAL_MODEL_SHA256=<文件 SHA-256>
LOCAL_MODEL_SIZE=<字节数>
```

然后构建，并用 `-LlamaDir` 把推理程序一起打包：

```powershell
.\packaging\build_windows.ps1 -PythonPath "D:\path\to\python.exe" `
  -LlamaDir "D:\llama-b****-bin-win-avx2-x64"
```

`-LlamaDir` 指向包含 `llama-server.exe` 的目录，其全部内容会被复制到
`dist\ExcelAssistant\llama\`，因此 `local_server` 写 `llama\llama-server.exe` 即可。
安装包体积只增加几十 MB，模型在首次使用时下载。

生成的 `dist\ExcelAssistant\llm_config.json` 形如：

```json
{
  "provider": "local",
  "api_key": "local",
  "base_url": "http://127.0.0.1:8080/v1",
  "model": "qwen2.5-1.5b-instruct",
  "auto_start": true,
  "local_server": "llama\\llama-server.exe",
  "model_url": "https://models.example.com/qwen2.5-1.5b-instruct-q4_k_m.gguf",
  "model_sha256": "…",
  "model_size": 1048576000,
  "model_filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf"
}
```

如果改用 `LOCAL_MODEL_PATH` 随包分发模型，则会出现 `local_model_path` 字段而没有
`model_url`；两者可以同时存在，此时以 `local_model_path` 为准，文件缺失才走下载。

约束与行为：

- 本地模式的 `base_url` **必须是回环地址**（`127.0.0.1`、`localhost`、`::1`）。构建脚本和
  打包程序都会校验，防止安装包把表格数据发往任意 HTTP 端点。
- 本地模式不需要 API Key；`api_key` 会写入占位值 `local`，因为 OpenAI 客户端要求非空。
- 本地模式不会发送远程专用的 `extra_body`（DeepSeek 的 thinking 开关），避免本地模板
  收到不认识的参数。
- 本地服务不可用时，聊天接口返回 503 和可操作提示，而不是 500。未启用自动启动时，
  提示会说明需要先手工启动服务以及应监听的地址。
- 模型尚未下载完成时，聊天接口同样返回 503，并提示先完成模型下载。

## 模型接口

界面依赖两个接口，也可用于排查问题：

| 接口 | 说明 |
| --- | --- |
| `GET /api/model/status` | 返回 `state`（`needs_download` / `downloading` / `verifying` / `starting` / `ready` / `error` / `disabled`）、进度、错误信息与 `can_download` |
| `POST /api/model/download` | 触发后台下载；进度请轮询上一个接口 |

## 跑基线评测

评测驱动的是应用真实的 Agent 图，因此分数代表线上行为：

```powershell
# 先小样本冒烟，确认服务连通
python -m dataset_builder.cli eval --data datasets --split test --limit 2 `
  --model qwen2.5-1.5b-instruct --base-url http://127.0.0.1:8080/v1

# 完整测试集
python -m dataset_builder.cli eval --data datasets --split test `
  --model qwen2.5-1.5b-instruct --base-url http://127.0.0.1:8080/v1 `
  --out datasets/eval-qwen15b.json
```

报告包含：任务成功率（主指标）、工具序列一致率、首个工具正确率、参数完全一致率、
触顶轮数上限比例、平均工具调用次数、按任务类别的成功率、失败原因分布。

`--base-url` 不填时会使用项目当前配置（`.env` 中的远程模型），可用于跑参照基线。

## 参照基线（远程模型）

在 8 条冻结测试样本上的远程模型结果：

| 指标 | 数值 |
| --- | --- |
| 任务成功率 | 100% |
| 工具序列一致 | 100% |
| 首个工具正确 | 100% |
| 参数完全一致 | 75% |
| 平均工具调用次数 | 2.12 |
| 平均模型轮数 | 3.12 |

这是小模型需要接近的目标。注意 8 条样本量很小，只能作方向参考；扩展到试点规模后
需要重跑。参数完全一致率 75% 也说明：结果正确但参数写法不同（例如给汇总列起了别名）
是常见情况，评测同时报告"结果正确"与"参数一致"两个口径就是为了区分这类差异。

## 打包前需要解决的阻塞

1. ~~**冻结程序强制走远程模型。**~~ **已解决**：`llm_config.json` 现在包含 `provider`
   字段，打包程序会按配置选择 remote 或 local；旧配置缺少该字段时仍按 remote 处理。
2. ~~**模型文件如何分发。**~~ **已解决**：安装包只带 `llama-server`（几十 MB），模型在
   首次使用时下载，支持断点续传与 SHA-256 校验（见上文"模型文件分发"）。
3. ~~**本地服务生命周期。**~~ **已解决**：应用可以按配置自动拉起与关闭 `llama-server`，
   复用已有服务，并对端口占用、文件缺失、启动失败给出可操作提示（见上文"自动启动本地服务"）。
4. **CPU 指令集兼容性。** llama.cpp 需要与本机 CPU 匹配的二进制（AVX2 等），
   需要在低配机器上验证。

只剩第 4 项。发布时应至少提供 AVX2 版本，并在低端机器上实测启动时间与内存占用。
在此之前的本地模型版安装包已经可以做到"装完 → 点下载 → 直接用"，无需手工启动服务。

## 下一步

1. 用约束解码跑 Qwen2.5-1.5B 基线，记录分数与失败类型。
2. 按失败类型扩充数据（尤其澄清与"无法完成"样本）。
3. 用 LoRA/QLoRA 做 SFT，训练与推理使用同一套 system prompt 与工具 schema。
4. 重新评测，与远程基线对比后决定是否发布本地模型版本。

详见数据集方案 [dataset-plan.md](./dataset-plan.md)。
