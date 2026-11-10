# 数据构造流水线

把仿真的 Excel 表格加工成可训练、可复现的 SFT 轨迹。所有产物写入 `build/`（该目录不入库）。

本目录只放**流程编排脚本**；可复用的算法在 [`dataset_builder/`](../dataset_builder)，训练与评测在 [`finetune/`](../finetune)。完整背景、失败分析与实测结果见 [docs/finetune-experiments.md](../docs/finetune-experiments.md)。

## 顺序

| 步骤 | 脚本 | 产物 | 是否调用模型 |
| --- | --- | --- | --- |
| 1 | `prepare_chains.py` | `build/chains-pool.json` | 是 |
| 2 | `generate_variants.py` | `build/variants.json` | 是 |
| 3 | `verify_all_variants.py` | `build/variants-fidelity.json` | 是 |
| 4 | `verify_by_execution.py` | `build/variants-faithful-v2.json` | 否 |
| 5 | `make_ambiguity_dataset.py` | `build/clarifications.json` | 是 |
| 6 | `variants_to_sft.py` | `build/sft-expanded/*.jsonl` | 否 |

```powershell
python pipeline/prepare_chains.py --no-rule --llm-target 1000 --llm-rounds 3
python pipeline/generate_variants.py --count 5 --out build/variants.json
python pipeline/verify_all_variants.py --workers 8
python pipeline/verify_by_execution.py
python pipeline/make_ambiguity_dataset.py --per-kind 1
python pipeline/variants_to_sft.py --variants build/variants-faithful-v2.json `
  --no-templates --clarifications build/clarifications.json --out build/sft-expanded
```

## 小样试跑

批量生成前建议先用单张表验证措辞质量，避免浪费 API 费用：

```powershell
python pipeline/trial_variants.py survey-20251078 --count 4   # 12 条链 × 4 变体
python pipeline/verify_variants.py                            # 反向解析校验忠实度
```

## 三个关键判断

**1. 忠实度最终以执行结果为准。** `verify_all_variants.py` 用反向解析把问题还原成操作链再比对，但参数级比对只能近似：模型省略可选的 `name` 或换一个列名，结果数据可能完全相同。因此 `verify_by_execution.py` 会把两条链都执行一遍，比较最终数据。这一步曾把 2400 余条被误判的变体找回来。

**2. 歧义必须可验证。** `make_ambiguity_dataset.py` 依赖 `dataset_builder/ambiguity.py` 的检测结果，只在两个候选确实给出不同数据时才算歧义。例如计数口径仅在「去重数 ≠ 行数」时成立；编号恰好唯一时，反问用户只是打扰。

**3. 澄清样本要限量。** `variants_to_sft.py` 默认把澄清样本压到训练集 3% 以内（`--max-clarify-ratio`）。反问过多会让模型对什么都先问一句，比猜错更烦人。

## 缓存

调用模型的步骤都会把原始回复写入 `build/*-cache*.json` 或 `build/*-raw*.json`，重跑同一条命令不会重复付费。要强制重新生成，删掉对应缓存文件，或用 `--no-reuse-llm-cache`。

## 留出集与评测

训练用的数据在 `build/`，评测用的留出集在 `datasets-holdout*/`（需入库，体积小）。

| 脚本 | 作用 |
| --- | --- |
| `build_holdout_eval.py` | 构建独立留出集：A 组（训练域新数据）/ B 组（全新业务域）。`--seed-preset 2` 生成第二批 |
| `compare_versions.py` | 多版本逐项对比（通过率 + 关注缺陷指标） |
| `compare_paired.py` | **配对 McNemar 精确检验 + Wilson 区间**，并列出不一致样本清单 |
| `bench_inference.py` | 单次推理延迟基准（首 token、解码速度、按输出长度换算） |

```powershell
# 第一批 = dev 集（找失败、调 DPO）
python pipeline/build_holdout_eval.py --out datasets-holdout `
  --tables-per-domain 3 --chains-per-table 8 --clarify-per-table 2
# 第二批 = test 集（全新种子，只汇报）
python pipeline/build_holdout_eval.py --out datasets-holdout2 --seed-preset 2 `
  --tables-per-domain 12 --chains-per-table 8 --clarify-per-table 2
```

> **划分纪律**：dev 集用来找失败样本、构造偏好对；test 集只用于汇报成绩。从 test 集取训练样本会让结论失效（第一轮 DPO 就踩过这个坑，见报告 6.8）。

## DPO 偏好学习

| 脚本 | 作用 |
| --- | --- |
| `capture_traces.py` | 重跑失败样本，抓取带参数的完整轨迹 + 每次调用的工具观察 |
| `make_dpo_pairs_from_traces.py` | 从真实失败轨迹构造偏好对，支持三类：`fix_after_error`（报错后修正）/ `silent_wrong_args`（参数写错但不报错）/ `missing_call`（漏步却声称完成） |
| `check_dpo_pairs.py` | **训练前自检**：计算初始 chosen 胜率，按类型分组。胜率接近 100% 时拒绝训练 |
| `demo_temperature.py` | 演示 `temperature` 与重复运行一致性的关系（澄清「温度=1 更稳定」的常见误解） |

```powershell
python pipeline/capture_traces.py --adapter build/lora-v4-dpo `
  --samples build/exp-a-dpo.json,build/exp-b-dpo.json --out build/traces-dpo.json
python pipeline/make_dpo_pairs_from_traces.py `
  --traces build/traces-dpo.json --out build/dpo-pairs-round2.jsonl
python pipeline/check_dpo_pairs.py --pairs build/dpo-pairs-round2.jsonl --init build/lora-v4-dpo
```

`check_dpo_pairs.py` 的判据来自一次真实失败：第一版偏好对用「显然正确」的样本造（初始胜率 100%），DPO 只把已经压到 −19 的概率继续压低，属于纯过拟合。**合格标准是初始胜率接近 50%**；若某类只有 30%（模型主动偏好错误动作），信号反而最强。

## 批量评测

`eval_batch2.ps1` / `eval_batch2_strong.ps1` / `eval_batch1_strong.ps1` 顺序跑多个任务，已存在的输出文件会跳过，支持中断续跑。

```powershell
powershell -File pipeline/eval_batch2.ps1          # 本地模型 4 次运行（约 7 小时）
powershell -File pipeline/eval_batch2_strong.ps1   # 远端强模型（走 API，不占 GPU）
```

> 脚本内注释与输出均为 ASCII，避免 PowerShell 以 GBK 读取 UTF-8 时解析失败。
