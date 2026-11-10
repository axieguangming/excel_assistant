# Excel AI 智能助手

Windows 桌面版 Excel 数据助手。用户上传最大 50 MB 的 `.xlsx`、`.xls` 或 `.csv` 文件后，可以用自然语言请求筛选、排序、分组汇总、Top N、去重、缺失值填充、新增计算列、描述统计和基础图表。数据处理通过经过校验的工具执行，不运行模型生成的 Python 代码。

## Windows 桌面版

桌面客户端由 `desktop.py` 启动本地 FastAPI 服务并在 pywebview 窗口中显示界面。客户端直接通过 HTTPS 调用远程模型 API，不需要单独部署本项目的 API 后端。

**发布密钥风险：** 按当前选定方案，构建时提供的模型 API Key 会写入安装包的 `llm_config.json`。安装包使用者可以提取该密钥并消耗账户额度、产生费用；密钥不能视为保密。发布者应使用专用且限额的 API Key，并配置服务商侧的额度/速率限制。不要把真实密钥提交到 Git、聊天或公开日志中。

### 构建 Windows 安装包

构建机需要 64 位 Python 3.12、项目依赖、`requirements-build.txt` 中的 PyInstaller，以及 Inno Setup 6。通过构建环境变量或项目根目录未提交的 `.env` 提供配置：

```ini
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

然后执行：

```powershell
python -m pip install -r requirements.txt -r requirements-build.txt
.\packaging\build_windows.ps1 -PythonPath "D:\path\to\python.exe"
```

配置生成器要求有效 API Key 和 HTTPS Base URL；不会将密钥打印到控制台。安装包输出到 `release/`。用户电脑需安装 [Microsoft Edge WebView2 Runtime](https://developer.microsoft.com/microsoft-edge/webview2/)；Windows 11 通常已包含该运行时。当前安装包尚未配置代码签名证书，公开发布时 Windows SmartScreen 可能显示未知发布者警告。

也可以构建**本地模型版**：在构建环境设置 `LLM_PROVIDER=local`、`LOCAL_BASE_URL`（仅限回环地址）和 `LOCAL_MODEL`，安装包会按本地 OpenAI 兼容服务运行，不需要远程密钥。若再加上 `LOCAL_AUTO_START=1` 与 `LOCAL_SERVER_PATH`、`LOCAL_MODEL_PATH`，打包程序会在启动时自动拉起 `llama-server`、退出时关闭它。详见 [docs/local-model.md](./docs/local-model.md)。

正式发布前还需配置代码签名、隐私政策和模型服务商披露。用户上传的工作簿及提问会从本机直接发送给配置的模型服务商，请勿上传不应离开本机的敏感数据。桌面客户端本地生成的数据保存在 `%LOCALAPPDATA%\ExcelAssistant`；桌面发行版会禁用训练样本记录。

## 开发运行

远程模型开发配置可放在项目根目录的 `.env` 中（不要提交真实密钥），然后运行：

```powershell
python desktop.py
```

网页调试入口仍可通过 `python app.py` 启动。通过 `LLM_PROVIDER=local` 可在开发环境使用 Ollama 等本地 OpenAI 兼容模型。Excel 加载项相关说明见 [addin/README.md](./addin/README.md)。

## 数据集构建

仓库提供用于微调与评测的离线数据集流水线，当前只生成仿真表格，默认输出 `.xlsx`（与用户实际上传的 Excel 文件一致）：

```powershell
# 小样验证
python -m dataset_builder.cli build --out datasets --table-count 6 --per-table 6

# 规则方法批量生成约 1000 条（纯本地，无 API 费用）
python -m dataset_builder.cli build --out datasets --table-count 45 --per-table 24 --max-attempts 30

# 回放校验：重新执行期望操作链并核对结果（自动使用最新日期目录）
python -m dataset_builder.cli replay --data datasets

# 在冻结测试集上评测模型（驱动真实 Agent）
python -m dataset_builder.cli eval --data datasets --split test

# 用分层抽样重新划分已有样本（不重新调用模型）
python -m dataset_builder.cli resplit --data datasets --seed 20251023
```

每次构建都会写入独立目录 `datasets/YYYY-MM-DD/`（同日再构建自动加 `-2` 等序号），旧批次不会被覆盖；读命令自动解析到最新的日期目录，也可以指定 `--data datasets/2026-10-23` 使用历史批次。

输出包含表格索引、样本、结果真值、拒绝原因、按工作簿分组的训练/验证/测试划分与覆盖度报告。样本真值在 **Excel 读回后的表格**上计算，因此与用户上传同一文件后得到的数据一致。详见 [docs/dataset-plan.md](./docs/dataset-plan.md)。`datasets/` 为本地生成目录，不纳入版本控制。

本地小模型（Qwen2.5-1.5B + llama.cpp）的基线与打包注意事项见 [docs/local-model.md](./docs/local-model.md)。

## 微调

`finetune/` 提供把数据集转成对话轨迹并用 LoRA 微调的示例：

```powershell
python -m pip install -r requirements-train.txt
python finetune/convert.py --data datasets --out build/sft
python finetune/train_lora.py --data build/sft --model <safetensors 权重目录> --output build/lora-out
```

轨迹里的中间观察结果是真实执行的，system prompt 与工具 schema 直接复用线上代码。
注意：**GGUF 不能用于微调**，训练需要原始 safetensors 权重。详见 [finetune/README.md](./finetune/README.md)。

当前最佳配置使用 LLM 生成的操作链、5 种口语化问法和歧义澄清样本（约 3900 条），
在 RTX 3050 上单 epoch 约 5 小时，再叠加 **56 对真实失败偏好对做两轮 DPO（共约 32 分钟）**。

**dev 集（396 条）上的 SFT 迭代过程**：

| 阶段 | 强模型 `deepseek-flash` | 未微调基座 | 微调后 |
| --- | --- | --- | --- |
| SFT 基线（v2） | 83.8% | 2.5% | 74.7% |
| + 观察文本修复（v3） | 83.8% | 2.5% | 78.5% |
| + 失败-修正轨迹（v4） | 83.8% | 2.5% | 82.1% |

**test 集（799 条，全新独立样本）上的 DPO 验证**：

| 版本 | A 组（514） | B 组（285） | 合计（799） |
| --- | --- | --- | --- |
| v4 | 85.8% | 77.2% | 82.7% |
| **v4 + DPO 两轮（当前最优）** | **91.1%** | **86.7%** | **89.5%** |
| 变化 | +5.3 | +9.5 | **+6.8 点（p < 0.0001）** |

未微调基座在 305 条执行类样本上**一次工具都没调用**——微调解决的首要问题是「会不会调用工具」。

> **关于第一轮 DPO 的重要更正**：第一轮的 44 对偏好对取自测试集的失败样本，之后又在同一批上报得 84.3%。逐条追溯发现 11 条新通过**全部落在训练样本内**，未接触的 352 条反而退步 2 条——那个数字度量的是记忆能力。
> 修正方法是 dev/test 划分：第 1 批只用来找缺陷，第 2 批只用来汇报。上表中的 89.5% 来自模型从未见过的样本。

**与强模型对比**（第 2 批，同批样本、同代代码）：强模型 78.7%，v4 82.7%，DPO-2 89.5%。
配对 McNemar：DPO-2 vs 强模型 **+10.8 点，p < 0.0001**，Wilson 95% 区间不重叠（`[87.2%, 91.4%]` vs `[75.8%, 81.4%]`）。

> **远端模型不可复现**：同一批样本两次运行只有 70.9% 的工具链相同，总分波动 2.8 点（本地微调模型重跑一致率 100%）。
> 但影响判定的差异只有 17.4%，且 +10.8 点是噪声的 3.8 倍——**单次测量足以定论**。
> 判断是否需要重复测量的依据是「差距 vs 噪声」的比值，详见报告 8.6 节。

已知短板：

- **剩余 84 条失败**（10.5%），其中执行类 70 条、澄清类 14 条，已从「集中缺陷」变为「长尾」
- **重试行为未被消除**：有重复调用的任务从 172 升到 187，但通过率从 49.4% 升到 73.3%
- **DPO margin 是否过度未验证**：收到 +4.25，而 `checkpoint-7` 是 +0.99

**当前判断：能力已足够，不需要再投入训练。** 再训一轮的影响面仅 8.8%、预期 89.5% → 约 93.9%，
成本约 8 小时，在已领先强模型 10.8 点的情况下没有决策价值。详见报告 8.3 节。

完整报告见 [docs/post-training-report-v2.md](./docs/post-training-report-v2.md)。

数据构造流水线（链生成、问法生成、忠实度校验）位于 [`pipeline/`](./pipeline/README.md)，产物写入 `build/`。

## 文档

- [**后训练微调实验报告（v2 · P0 + 两轮 DPO）**](./docs/post-training-report-v2.md) — dev/test 划分、数据泄漏修正、副作用分析与偏好学习
- [后训练微调实验报告（v2 基线）](./docs/post-training-report.md) — 数据 → 训练 → 三方对照 → 失败分析
- [项目描述（求职用）](./docs/project-description.md) — 简历版、完整版与面试问答准备
- [Excel Agent 微调数据集构造方案](./docs/dataset-plan.md)
- [LoRA 微调实验记录](./docs/finetune-experiments.md)
- [评测口径](./docs/eval-protocol.md)
- [后续工作计划](./docs/next-steps.md)
- [数据构造流水线](./pipeline/README.md)
- [本地小模型（Qwen2.5-1.5B + llama.cpp）](./docs/local-model.md)

## 许可

本软件仅限个人、非商业用途；允许转发版权方发布的未修改官方安装包，不允许商业使用、修改或发布衍生版本。详见 [LICENSE.txt](./LICENSE.txt)。
