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
