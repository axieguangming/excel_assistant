# LoRA 微调示例

把规则生成的数据集转换成对话轨迹，用 LoRA 对 `Qwen2.5-1.5B-Instruct` 做监督微调。

## 1. 安装依赖

训练依赖与运行依赖分开（见 `requirements-train.txt`），避免打包安装包时被带上：

```powershell
python -m pip install -r requirements-train.txt
```

需要 CUDA 环境。1.5B 在纯 CPU 上训练不现实（推理都要 24 tok/s）。

## 2. 转换数据

```powershell
python finetune/convert.py --data datasets --out build/sft
```

输出：

```
build/sft/
├── train.jsonl   888 条（多步 452）
├── val.jsonl      96 条（多步 48）
├── test.jsonl     96 条
└── tools.json     9 个工具定义，便于核对
```

每条记录形如 `{"sample_id": ..., "tools": [...], "messages": [...]}`，`messages` 是完整的
「system → user → assistant(tool_call) → tool(观察) → assistant(...)」轨迹。

转换的关键点：

- **中间观察结果是真实执行的**。不是编造文本，而是重新跑一遍操作链后用
  `excel_ops._frame_observation` 渲染，和线上 Agent 看到的一致。
- **system prompt 与工具 schema 直接取自线上代码**（`agent.AGENT_SYSTEM`、
  `excel_ops.EXCEL_TOOLS`），避免训练/推理格式漂移。
- 每步执行前会复制 DataFrame。`_computed_column` 与 `_fill_missing_values` 是原地修改的，
  不复制会污染同一工作簿的其他样本（这个 bug 在开发时真实触发过）。

## 3. 训练

```powershell
python finetune/train_lora.py --data build/sft --output build/lora-out
```

默认配置：

| 项 | 值 |
| --- | --- |
| LoRA rank / alpha | 16 / 32 |
| target modules | q/k/v/o + gate/up/down |
| 学习率 | 1e-4，cosine |
| batch × 梯度累积 | 1 × 8 |
| epochs | 3 |
| 量化 | 4bit（`--no-4bit` 关闭） |
| 损失范围 | 只在 assistant 片段 |

先做一次冒烟，确认能跑通再正式训练：

```powershell
python finetune/train_lora.py --data build/sft --max-steps 10
```

## 4. 评测

训练完导出合并后的模型或直接用 LoRA 起一个 OpenAI 兼容服务，然后用现成 CLI 评测：

```powershell
python -m dataset_builder.cli eval --data datasets --split val --base-url http://127.0.0.1:8080/v1
```

对照基线（同一测试集，96 条）：

| | deepseek-flash | 1.5B 微调前 |
| --- | --- | --- |
| 任务成功率 | 100% | 47.92% |
| 单步任务 | 100% | 90% |
| 多步任务 | 100% | **4%** |

**微调目标很明确：补多步续接。** 单步决策与参数填充已经够用，数据集里 452 条多步样本
正是为此准备的。

## 已知局限

1. **问句泄漏实现细节**。当前训练集约 47% 的问句包含机器生成的列名（如
   `预算金额_x_差异`）和参数细节（"保留 2 位小数"、"取前 10 条"）。模型可能学会
   "照着念"而不是"理解意图"。建议先用 `--proposer llm` 生成更自然的问句再训练。
2. **类别不均衡**。`dedupe`、`describe` 各只有 37 条训练样本，微调后这两类可能退化，
   值得在评测里单独看。
3. **样本量偏小**。888 条训练样本足够验证流程与方向，但要达到产品级还需要数千条。
