"""构建 Excel Agent 微调数据集：仿真表格 → 操作链 → 执行真值 → 自然语言问题。

该包实现 [docs/dataset-plan.md](../docs/dataset-plan.md) 的流水线，当前只支持
仿真生成的表格，因此不涉及第三方数据许可或个人信息。
"""

from dataset_builder.chain import (
    ChainValidationError,
    chains_equivalent,
    normalize_chain,
)
from dataset_builder.executor import ExecutionOutcome, execute_chain
from dataset_builder.pipeline import BuildConfig, build_dataset
from dataset_builder.tables import (
    DOMAINS,
    SyntheticTable,
    generate_table,
    generate_tables,
    load_tables,
    save_tables,
)

__all__ = [
    "BuildConfig",
    "ChainValidationError",
    "DOMAINS",
    "ExecutionOutcome",
    "SyntheticTable",
    "build_dataset",
    "chains_equivalent",
    "execute_chain",
    "generate_table",
    "generate_tables",
    "load_tables",
    "normalize_chain",
    "save_tables",
]
