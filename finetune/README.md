# LoRA 微调示例

把数据集转换成对话轨迹，用 LoRA 对 `Qwen2.5-1.5B-Instruct` 做监督微调。

完整的实验记录（两轮数据对照、失败模式分析、工程注意事项）见
[docs/finetune-experiments.md](../docs/finetune-experiments.md)。

## 1. 安装依赖

训练依赖与运行依赖分开（见 `requirements-train.txt`），避免打包安装包时被带上：

```powershell
python -m pip install -r requirements-train.txt
```

**需要 CUDA 版 torch。** 检查：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

若输出 `2.x.x+cpu` 或 `False`，说明装的是 CPU 版，1.5B 训练不可行
（本机就是这种情况：`2.14.1+cpu`，无 CUDA）。需要先装 CUDA 版：

```powershell
python -m pip uninstall -y torch
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 2. 准备可训练的模型权重

**GGUF 不能用于微调。** 这是最容易踩的坑：

| 格式 | 用途 | 能否训练 |
| --- | --- | --- |
| `*.gguf`（如 q4_k_m） | llama.cpp 推理，已量化到 4bit | ❌ |
| `*.safetensors` / `*.bin` | 原始 fp16/bf16 权重 | ✅ |

`models/qwen2.5-1.5b-instruct-q4_k_m.gguf` 只供 llama.cpp 推理。训练脚本现在会直接
拒绝 GGUF 路径并说明原因，而不是静默去 HuggingFace 下载 3.1 GB。

需要另外准备原始权重，二选一：

```powershell
# 方式 A：让脚本自动下载（约 3.1 GB）
# 国内可先设镜像：set HF_ENDPOINT=https://hf-mirror.com

# 方式 B：先下载再指向本地目录（推荐，可控且可复用）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='models')"

python finetune/train_lora.py --data build/sft --model models/Qwen/Qwen2.5-1.5B-Instruct
```

脚本会校验 `--model` 指向的目录是否真的含 `*.safetensors` 或 `*.bin`。

## 3. 转换数据

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

## 4. 训练

```powershell
python finetune/train_lora.py --data build/sft --model <权重目录> --output build/lora-out
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

v2 实际使用的是单 epoch：

```powershell
python finetune/train_lora.py --data build/sft-expanded `
  --model models/Qwen2.5-1.5B-Instruct --output build/lora-expanded `
  --epochs 1 --max-length 5184 --save-every 4
```

两个参数值得注意：

- **`--max-length` 必须覆盖完整轨迹**。提示部分（数据概况 + 8 行预览）约占 2680 token，
  v2 的轨迹 p50 3810、max 5181。取 2048 会把所有 assistant 标签截掉，等于训练「复述提示」；
  取 4608 仍有 5.3% 样本被截断，被截断的样本只学工具调用、不学汇报。
- **`--save-every` 按步保存**。单 epoch 训练数小时，默认的按 epoch 保存意味着中断后一无所有。
  现在每个 epoch 内保存 4 次，保留最近 2 个。

单 epoch 已经足够：v1 的 eval_loss 在第 1 个 epoch 后基本不再变化（0.00287 → 0.0032），
继续训练只增加过拟合风险。

先做一次冒烟，确认能跑通再正式训练：

```powershell
python finetune/train_lora.py --data build/sft --max-steps 10
```

## 5. 评测

两种方式：一是用现成 CLI 驱动真实 Agent 图（需要数据集批次，含 `samples.jsonl`）：

```powershell
python -m dataset_builder.cli eval --data datasets --split val --base-url http://127.0.0.1:8080/v1
```

二是直接评测本地适配器。`evaluate_local.py` 读数据集批次，`eval_sft.py` 读 SFT
轨迹目录，两者都驱动 `agent.invoke_agent` 而非简化调用：

```powershell
# 手工构造的口语化评测集（检验是否只记住模板句式）
python finetune/evaluate_colloquial.py --set evalsets/colloquial_2026-10.json `
  --adapter build/lora-expanded

# 训练同分布的测试分区
python finetune/eval_sft.py --split test --adapter build/lora-expanded
```

### 实测结果

> **最新数据以 [docs/post-training-report-v2.md](../docs/post-training-report-v2.md) 为准**。
> 下面都是历史测量，口径与当前报告不同（v1/v2 时代的数据集与判据）。其中「澄清类 100%」
> 是旧判据的假象——旧判据只检查「是否调用执行类工具」，对从不调用工具的模型天然有利，
> 修正判据后 v2 真实值为 74.7%（91 条）。

**第 1 批留出集（396 条，现降级为 dev 集）**：

| 评测集 | 强模型 `deepseek-flash` | 未微调基座 | 微调后（v2） |
| --- | --- | --- | --- |
| A：新数据值，沿用训练表结构（247 条） | 85.4% | 1.6% | 79.8% |
| B：新业务域，列名基本全新（149 条） | 81.2% | 4.0% | 66.4% |
| 合计 | 83.8% | 2.5% | 74.7% |

**第 2 批留出集（799 条，v4 与 DPO 的最终对比口径）**：

| 版本 | A 组（514） | B 组（285） | 合计（799） |
| --- | --- | --- | --- |
| v4（SFT） | 85.8% | 77.2% | 82.7% |
| v4 + 两轮 DPO | **91.1%** | **86.7%** | **89.5%** |

> 第一轮 DPO 曾报「396 条上 84.3%，超过强模型 83.8%」，但它的偏好对取自该测试集，
> 11 条新通过全部落在训练样本内。该数字作废，详见报告 6.8 节。

**口语化评测集**（32 条人工改写样本，句式与训练数据完全不同）：

| 类别 | 规则数据训练（v1） | LLM 链 + 口语化（v2） |
| --- | --- | --- |
| 语义等价请求 | 30.4% | **73.9%** |
| 不可完成请求 | 33.3% | **66.7%** |
| 合计 | 31.2% | **71.9%** |

**同分布测试集**（v2，499 条，表结构见过）：执行类 78.8%。

> 早先一版在规则生成的测试集上拿到 100%，但同一模型在口语化集上只有 30.4%。
> 两个数字的落差就是「模板泄漏」的量级——同模板测试集不能用来判断模型能力。

## 已知局限

1. **多步链会漏掉排序**。v2 在同分布测试集失败的 102 条中，31 条是缺少
   `sort_rows`。原因是排序不改变行数与列名，SFT 的 token 似然对它的惩罚很弱。
   越是没有「副作用」的必要步骤，越容易被省略。
2. **问句中的生成列名**。由工具产生的列名（`销售额_sum`）若出现在提问里，模型会
   "照着念"而不是理解意图。现已按「是否含下划线」区分：自然业务命名（`销售额合计`）
   允许出现在问句，程序拼装的内部名则禁止。
3. **步数上限只有 3**。`agent` 允许 10 轮，但训练集最大 3 步，4 步以上的长任务未训练。
4. **零错误恢复样本**。所有训练轨迹的工具调用都成功，模型没见过「工具报错后修正参数」
   这一模式。
5. **不可完成请求仍会硬套工具**。工具名与请求字面相似时尤甚（「标红」→ 筛选、
   「移动平均」→ 绘图），9 条不可完成请求中仍有 3 条如此。
6. **`eval_loss` 不能跨数据集比较**。v1 的 0.0029 对应模板数据（同一链只有 2 条句式），
   v2 的 0.1006 才对应真实口语的多样性。
