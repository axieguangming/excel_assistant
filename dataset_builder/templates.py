"""确定性模板生成器：离线生成操作链与自然语言问题。

模板生成器用于自测、无网络环境和流水线冒烟验证。正式的语义多样数据
应使用 `dataset_builder.llm.LLMChainProposer` 等大模型实现。
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Mapping

from dataset_builder.tables import SyntheticTable, column_kind

CATEGORIES: dict[str, str] = {
    "filter_rows": "filter",
    "sort_rows": "sort",
    "aggregate_data": "aggregate",
    "top_n_rows": "top_n",
    "deduplicate_rows": "dedupe",
    "fill_missing_values": "fill_missing",
    "add_computed_column": "computed_column",
    "describe_data": "describe",
    "create_chart": "chart",
}

FUNCTION_NAMES = {
    "sum": "合计",
    "mean": "平均值",
    "min": "最小值",
    "max": "最大值",
    "count": "计数",
    "nunique": "去重计数",
}

OPERATOR_NAMES = {
    "eq": "等于",
    "ne": "不等于",
    "gt": "大于",
    "gte": "大于等于",
    "lt": "小于",
    "lte": "小于等于",
    "contains": "包含",
    "not_contains": "不包含",
    "is_empty": "为空",
    "not_empty": "不为空",
}

ARITHMETIC_NAMES = {
    "add": "加上",
    "subtract": "减去",
    "multiply": "乘以",
    "divide": "除以",
}

CHART_NAMES = {"bar": "柱状图", "line": "折线图", "pie": "饼图", "scatter": "散点图"}


def categories_for(steps: Iterable[Mapping[str, Any]]) -> list[str]:
    seen = {CATEGORIES.get(str(step["tool"]), str(step["tool"])) for step in steps}
    return sorted(seen)


def summarize_steps(steps: Iterable[Mapping[str, Any]]) -> str:
    descriptions = [_describe_step(step) for step in steps]
    return "，然后".join(descriptions)


def _describe_step(step: Mapping[str, Any]) -> str:
    tool = str(step["tool"])
    args = step.get("args", {})

    if tool == "filter_rows":
        parts = []
        for condition in args.get("filters", []):
            operator = OPERATOR_NAMES.get(condition.get("operator"), condition.get("operator"))
            if condition.get("operator") in {"is_empty", "not_empty"}:
                parts.append(f"{condition.get('column')} {operator}")
            else:
                parts.append(
                    f"{condition.get('column')} {operator} {condition.get('value')}"
                )
        return "筛选出 " + " 且 ".join(parts)
    if tool == "sort_rows":
        keys = []
        for sort_key in args.get("sort_by", []):
            order = "升序" if sort_key.get("ascending", True) else "降序"
            keys.append(f"{sort_key.get('column')}（{order}）")
        return "按 " + "、".join(keys) + " 排序"
    if tool == "aggregate_data":
        metrics = []
        for metric in args.get("metrics", []):
            function = FUNCTION_NAMES.get(metric.get("function"), metric.get("function"))
            name = metric.get("name") or metric.get("column")
            metrics.append(f"{name} 的{function}")
        group_by = args.get("group_by") or []
        if group_by:
            return f"按 {'、'.join(group_by)} 分组统计 " + "、".join(metrics)
        return "统计 " + "、".join(metrics)
    if tool == "top_n_rows":
        keys = []
        for sort_key in args.get("sort_by", []):
            order = "升序" if sort_key.get("ascending", True) else "降序"
            keys.append(f"{sort_key.get('column')}（{order}）")
        return f"按 {'、'.join(keys)} 取前 {args.get('limit')} 条"
    if tool == "deduplicate_rows":
        subset = args.get("subset") or []
        if subset:
            return f"按 {'、'.join(subset)} 去重"
        return "按整行去重"
    if tool == "fill_missing_values":
        strategy = args.get("fill_strategy")
        subset = args.get("subset") or []
        scope = f"{'、'.join(subset)}" if subset else "所有字段"
        if strategy == "ffill":
            return f"对 {scope} 使用向前填充"
        if strategy == "bfill":
            return f"对 {scope} 使用向后填充"
        return f"把 {scope} 的空值填充为 {args.get('fill_value')}"
    if tool == "add_computed_column":
        operator = ARITHMETIC_NAMES.get(args.get("operator"), args.get("operator"))
        right = args.get("right_column", args.get("right_value"))
        text = f"新增列 {args.get('name')}，等于 {args.get('left')} {operator} {right}"
        if args.get("round_digits") is not None:
            text += f" 并保留 {args['round_digits']} 位小数"
        return text
    if tool == "describe_data":
        return "生成描述统计"
    if tool == "create_chart":
        chart = CHART_NAMES.get(args.get("chart_type"), args.get("chart_type"))
        if args.get("chart_x"):
            return f"绘制{chart}，横轴 {args['chart_x']}，纵轴 {args['chart_y']}"
        return f"绘制{chart}，数值为 {args['chart_y']}"
    return tool


class TemplateChainProposer:
    """按表格结构枚举可执行的操作链模板，再按种子打乱。"""

    def propose(
        self,
        table: SyntheticTable,
        count: int,
        *,
        max_steps: int = 4,
        allowed_tools: Iterable[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        allowed = set(allowed_tools) if allowed_tools else None
        candidates = [
            chain
            for chain in _candidates(table, max_steps=max_steps)
            if allowed is None
            or all(step["tool"] in allowed for step in chain)
        ]

        rng = random.Random(table.table_id)
        rng.shuffle(candidates)

        selected: list[list[dict[str, Any]]] = []
        remaining: list[list[dict[str, Any]]] = []
        signatures: set[tuple[str, ...]] = set()
        for chain in candidates:
            signature = tuple(categories_for(chain))
            if signature in signatures:
                remaining.append(chain)
            else:
                signatures.add(signature)
                selected.append(chain)

        return [*selected, *remaining][: max(0, count)]


class TemplateQuestionWriter:
    """由操作链直接生成自然语言问题与预期答复。"""

    def write(
        self,
        table: SyntheticTable,
        steps: list[dict[str, Any]],
        *,
        rows: int | None = None,
        columns: list[str] | None = None,
    ) -> dict[str, str]:
        summary = summarize_steps(steps)
        question = f"请对{table.sheet}执行以下处理：{summary}。"
        answer = f"已按要求完成操作：{summary}。"
        if rows is not None and columns is not None:
            answer += f"结果为 {rows} 行 × {len(columns)} 列。"
        return {"question": question, "answer": answer}


def _numeric_columns(table: SyntheticTable) -> list[str]:
    return [
        str(name)
        for name in table.frame.columns
        if column_kind(table.frame[name]) == "numeric"
    ]


def _columns_by_kind(table: SyntheticTable, kind: str) -> list[str]:
    return [
        str(name)
        for name in table.frame.columns
        if column_kind(table.frame[name]) == kind
    ]


def _first_value(table: SyntheticTable, column: str) -> Any:
    values = table.frame[column].dropna()
    return values.iloc[0] if not values.empty else None


def _candidates(table: SyntheticTable, *, max_steps: int) -> list[list[dict[str, Any]]]:
    numeric = _numeric_columns(table)
    text = _columns_by_kind(table, "text")
    datetimes = _columns_by_kind(table, "datetime")

    chains: list[list[dict[str, Any]]] = []

    if text:
        column = text[0]
        value = _first_value(table, column)
        if value is not None:
            chains.append(
                [
                    {
                        "tool": "filter_rows",
                        "args": {
                            "filters": [
                                {"column": column, "operator": "eq", "value": str(value)}
                            ]
                        },
                    }
                ]
            )
        chains.append(
            [
                {
                    "tool": "filter_rows",
                    "args": {
                        "filters": [
                            {"column": column, "operator": "not_empty", "value": None}
                        ]
                    },
                }
            ]
        )

    if numeric:
        column = numeric[0]
        chains.append(
            [
                {
                    "tool": "filter_rows",
                    "args": {
                        "filters": [{"column": column, "operator": "gt", "value": 0}]
                    },
                }
            ]
        )
        chains.append(
            [{"tool": "sort_rows", "args": {"sort_by": [{"column": column, "ascending": False}]}}]
        )
        chains.append(
            [
                {
                    "tool": "top_n_rows",
                    "args": {
                        "sort_by": [{"column": column, "ascending": False}],
                        "limit": 10,
                    },
                }
            ]
        )
        chains.append(
            [
                {
                    "tool": "aggregate_data",
                    "args": {
                        "metrics": [{"column": column, "function": "sum"}],
                    },
                }
            ]
        )
        chains.append(
            [
                {
                    "tool": "describe_data",
                    "args": {},
                }
            ]
        )

    if numeric and text:
        group = text[0]
        chains.append(
            [
                {
                    "tool": "aggregate_data",
                    "args": {
                        "group_by": [group],
                        "metrics": [{"column": numeric[0], "function": "sum"}],
                    },
                }
            ]
        )
        chains.append(
            [
                {
                    "tool": "create_chart",
                    "args": {
                        "chart_type": "bar",
                        "chart_x": group,
                        "chart_y": numeric[0],
                    },
                }
            ]
        )
        chains.append(
            [
                {
                    "tool": "sort_rows",
                    "args": {"sort_by": [{"column": group, "ascending": True}]},
                },
                {
                    "tool": "top_n_rows",
                    "args": {
                        "sort_by": [{"column": numeric[0], "ascending": False}],
                        "limit": 5,
                    },
                },
            ]
        )

    if len(numeric) >= 2:
        chains.append(
            [
                {
                    "tool": "add_computed_column",
                    "args": {
                        "name": "计算列",
                        "operator": "multiply",
                        "left": numeric[0],
                        "right_column": numeric[1],
                        "round_digits": 2,
                    },
                }
            ]
        )

    if text:
        chains.append(
            [
                {
                    "tool": "deduplicate_rows",
                    "args": {"subset": [text[0]]},
                }
            ]
        )
        chains.append(
            [
                {
                    "tool": "fill_missing_values",
                    "args": {"fill_strategy": "ffill", "subset": [text[0]]},
                }
            ]
        )

    if numeric:
        chains.append(
            [
                {
                    "tool": "fill_missing_values",
                    "args": {"fill_strategy": "value", "subset": [numeric[0]], "fill_value": 0},
                }
            ]
        )

    if datetimes and numeric:
        chains.append(
            [
                {
                    "tool": "sort_rows",
                    "args": {
                        "sort_by": [
                            {"column": datetimes[0], "ascending": True},
                            {"column": numeric[0], "ascending": False},
                        ]
                    },
                }
            ]
        )

    if numeric and text:
        chains.append(
            [
                {
                    "tool": "filter_rows",
                    "args": {
                        "filters": [
                            {"column": text[0], "operator": "not_empty", "value": None}
                        ]
                    },
                },
                {
                    "tool": "aggregate_data",
                    "args": {
                        "group_by": [text[0]],
                        "metrics": [
                            {"column": numeric[0], "function": "sum"},
                            {"column": numeric[0], "function": "count"},
                        ],
                    },
                },
                {
                    "tool": "sort_rows",
                    "args": {
                        "sort_by": [
                            {"column": f"{numeric[0]}_sum", "ascending": False}
                        ]
                    },
                },
            ]
        )

    return [chain for chain in chains if len(chain) <= max_steps]
