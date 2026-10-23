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

1. **冻结程序强制走远程模型。** [load_model.py](../load_model.py) 中
   `provider = "remote" if frozen else ...`，安装包目前无法使用本地模型。需要让打包配置
   能声明 provider 与本地服务地址。
2. **模型文件如何分发。** 随包分发会让安装包增加约 1 GB；首次启动下载则依赖网络与
   断点续传，且需要校验完整性。两种方式都需要先定。
3. **本地服务生命周期。** 应用需要负责拉起/关闭 `llama-server`，并处理端口占用与启动失败。
4. **WebView2 之外的系统依赖。** llama.cpp 需要与本机 CPU 指令集匹配的二进制
   （AVX2 等），需要在低配机器上验证。

在 1–3 解决前，安装包只能继续使用远程模型。

## 下一步

1. 用约束解码跑 Qwen2.5-1.5B 基线，记录分数与失败类型。
2. 按失败类型扩充数据（尤其澄清与"无法完成"样本）。
3. 用 LoRA/QLoRA 做 SFT，训练与推理使用同一套 system prompt 与工具 schema。
4. 重新评测，与远程基线对比后决定是否发布本地模型版本。

详见数据集方案 [dataset-plan.md](./dataset-plan.md)。
