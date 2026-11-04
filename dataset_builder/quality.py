"""检测语义可疑的操作链。

规则生成器按「列 × 操作 × 参数」枚举，能产出数学上可执行、但业务上无意义的链。
LLM 生成器同样可能产出这类链。这类样本即使问题写得再自然，也会教模型做错事，
应当在生成变体前先剔除。

已识别的模式：

1. **饼图没有分类轴**：`create_chart(chart_type="pie", chart_y=X)` 会给每一行
   画一个扇区。90 行就是 90 个扇区，这不是任何用户想要的东西。
2. **筛到单值后又分组**：`filter(受访者编号 == "R501")` 之后
   `group_by=["受访者编号"]`，结果恒为 1 行 1 组。
3. **对量表列求和**：把 1–5 分、0–10 分的评分直接加总没有业务含义
   （应为平均或计数）。
4. **对标识列做数值运算**：对订单号、员工编号、受访者编号求 sum/mean，
   或把它们放进计算列，结果没有意义。
5. **TopN 条数覆盖全表**：`limit` 不小于表行数时，「前 N 名」退化成整表。
"""

from __future__ import annotations

import re
from typing import Any, Sequence

# 常见的评分/量表列关键词；对这些列求和需要额外人工确认。
RATING_HINTS = ("满意度", "评分", "打分", "星级", "推荐意愿", "得分", "评分值")

# 标识类列名：求和、求均值、参与计算列都无业务意义。
IDENTIFIER_PATTERN = re.compile(r"(编号|号码|序号|代码|编码|单号|ID|_id|code)$", re.I)

# 量纲分组：跨组做加减没有业务含义（运费 − 毛重 = 元 − 千克）。
DIMENSION_GROUPS: dict[str, tuple[str, ...]] = {
    "重量": ("毛重", "净重", "重量", "库存数量", "安全库存", "数量"),
    "金额": ("运费", "金额", "销售额", "单价", "支出", "预算", "回款", "费用", "成本"),
    "时间": ("耗时", "时长", "工时", "加班时长"),
}

# 派生指标（比率、单位值、均值）与原始指标做加减同样无意义。
DERIVED_HINTS = ("率", "占比", "单位", "平均值", "_mean", "_ratio")


def _dimension_of(name: str) -> str | None:
    for dimension, hints in DIMENSION_GROUPS.items():
        if any(hint in name for hint in hints):
            return dimension
    return None


def _is_derived(name: str) -> bool:
    return any(hint in name for hint in DERIVED_HINTS)


def is_identifier_column(name: str) -> bool:
    return bool(IDENTIFIER_PATTERN.search(str(name).strip()))


def chain_quality_warnings(
    steps: Sequence[dict[str, Any]],
    *,
    total_rows: int | None = None,
    columns: Sequence[str] | None = None,
) -> list[str]:
    warnings: list[str] = []
    filtered_single: set[str] = set()

    for index, step in enumerate(steps):
        tool = step.get("tool")
        args = step.get("args") or {}

        if tool == "create_chart" and args.get("chart_type") == "pie" and not args.get("chart_x"):
            warnings.append(
                "pie_without_category: 饼图未指定分类轴，会给每行画一个扇区"
                + (f"（当前 {total_rows} 行）" if total_rows else "")
            )

        if tool == "filter_rows":
            for condition in args.get("filters") or []:
                if not isinstance(condition, dict):
                    continue
                if condition.get("operator") in {"eq", "contains", "not_contains"} and condition.get("value"):
                    filtered_single.add(str(condition.get("column")))

        if tool == "top_n_rows":
            limit = args.get("limit")
            if (
                total_rows
                and isinstance(limit, int)
                and limit >= total_rows
            ):
                warnings.append(
                    f"top_n_covers_whole_table: limit={limit} 不小于表行数 {total_rows}，"
                    "「前 N 名」退化为整表"
                )

        if tool == "aggregate_data":
            group_by = [str(name) for name in args.get("group_by") or []]
            for column in group_by:
                if column in filtered_single:
                    warnings.append(
                        f"degenerate_group: 已把 {column} 筛成单值，再按它分组恒为 1 组"
                    )
            for metric in args.get("metrics") or []:
                if not isinstance(metric, dict):
                    continue
                target = str(metric.get("column", ""))
                function = metric.get("function")
                if is_identifier_column(target) and function in {
                    "sum",
                    "mean",
                    "min",
                    "max",
                }:
                    warnings.append(
                        f"numeric_op_on_identifier: 对标识列 {target} 做 {function} "
                        "没有业务含义（计数可用 count/nunique）"
                    )
                if function in {"sum", "nunique"} and any(
                    hint in target for hint in RATING_HINTS
                ):
                    warnings.append(
                        f"sum_on_rating_scale: 对量表列 {target} 做 {function}，业务含义存疑"
                    )

        if tool == "add_computed_column":
            for key in ("left", "right_column"):
                column = args.get(key)
                if column and is_identifier_column(str(column)):
                    warnings.append(
                        f"arithmetic_on_identifier: 计算列引用了标识列 {column}"
                    )
            # 加减法要求两侧同量纲；除法/乘法常用于构造比率或总量，不在此列。
            if args.get("operator") in {"add", "subtract"}:
                left = str(args.get("left") or "")
                right = str(args.get("right_column") or "")
                if left and right:
                    left_dim = _dimension_of(left)
                    right_dim = _dimension_of(right)
                    if left_dim and right_dim and left_dim != right_dim:
                        warnings.append(
                            f"cross_dimension_arithmetic: {left}（{left_dim}）与 "
                            f"{right}（{right_dim}）量纲不同，相加减无业务含义"
                        )
                    elif _is_derived(left) != _is_derived(right):
                        derived = left if _is_derived(left) else right
                        warnings.append(
                            f"arithmetic_on_derived_metric: {derived} 是派生指标"
                            "（比率/单位值），与原始指标相加减无业务含义"
                        )

    return warnings
