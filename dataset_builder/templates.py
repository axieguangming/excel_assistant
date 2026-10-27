"""确定性模板生成器：离线生成操作链与自然语言问题。

模板生成器用于自测、无网络环境和流水线冒烟验证。正式的语义多样数据
应使用 `dataset_builder.llm.LLMChainProposer` 等大模型实现。
"""

from __future__ import annotations

import json
import random
from typing import Any, Iterable, Mapping

import pandas as pd

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
    """由操作链直接生成自然语言问题与预期答复。

    同一操作链会生成多条不同措辞的问题（``variants``），因为固定句式重复上千次
    会让训练数据退化成记忆模板，而不是学习理解需求。
    """

    # 每个模板用 {sheet} 与 {summary} 占位。
    QUESTION_TEMPLATES: tuple[str, ...] = (
        "请对{sheet}执行以下处理：{summary}。",
        "在{sheet}里，{summary}。",
        "{summary}，请直接在{sheet}上操作。",
        "帮我处理一下{sheet}：{summary}。",
        "基于{sheet}的数据，{summary}。",
        "{sheet}需要这样处理：{summary}。",
        "对{sheet}做这些操作——{summary}。",
    )

    def __init__(self, variants: int = 1, *, seed: int = 0):
        self.variants = max(1, variants)
        self.seed = seed

    def write(
        self,
        table: SyntheticTable,
        steps: list[dict[str, Any]],
        *,
        rows: int | None = None,
        columns: list[str] | None = None,
    ) -> dict[str, str]:
        summary = summarize_steps(steps)
        answer = f"已按要求完成操作：{summary}。"
        if rows is not None and columns is not None:
            answer += f"结果为 {rows} 行 × {len(columns)} 列。"
        return {
            "question": self._question(table, summary, index=0),
            "answer": answer,
        }

    def write_variants(
        self,
        table: SyntheticTable,
        steps: list[dict[str, Any]],
        *,
        rows: int | None = None,
        columns: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """返回 ``variants`` 条同义问题，共享同一份预期答复。"""
        summary = summarize_steps(steps)
        answer = f"已按要求完成操作：{summary}。"
        if rows is not None and columns is not None:
            answer += f"结果为 {rows} 行 × {len(columns)} 列。"

        seen: set[str] = set()
        questions: list[str] = []
        for index in range(self.variants):
            question = self._question(table, summary, index=index)
            if question in seen:
                continue
            seen.add(question)
            questions.append(question)
        return [{"question": q, "answer": answer} for q in questions]

    def _question(self, table: SyntheticTable, summary: str, *, index: int) -> str:
        # 用表名与序号选择模板，保证同一批数据里措辞分布稳定可复现。
        offset = sum(ord(char) for char in table.table_id) + self.seed
        template = self.QUESTION_TEMPLATES[(offset + index) % len(self.QUESTION_TEMPLATES)]
        question = template.format(sheet=table.sheet, summary=summary)
        # 模板拼接后可能出现"。。"或"，。"，统一收尾。
        while "。。" in question:
            question = question.replace("。。", "。")
        return question.replace("，。", "。")


def _is_populated(table: SyntheticTable, column: str) -> bool:
    """全空列（notna 为 0）不应参与任务生成。

    这类列在 pandas 里常被推断成 float64，从而混进"数值列"里；用它做加减或汇总
    只会得到整列空值，是退化任务，用户也不会提出这种需求。
    """
    return bool(table.frame[column].notna().any())


def _numeric_columns(table: SyntheticTable) -> list[str]:
    return [
        str(name)
        for name in table.frame.columns
        if column_kind(table.frame[name]) == "numeric" and _is_populated(table, name)
    ]


def _columns_by_kind(table: SyntheticTable, kind: str) -> list[str]:
    return [
        str(name)
        for name in table.frame.columns
        if column_kind(table.frame[name]) == kind and _is_populated(table, name)
    ]


def _first_value(table: SyntheticTable, column: str) -> Any:
    values = table.frame[column].dropna()
    return values.iloc[0] if not values.empty else None

def _distinct_values(table: SyntheticTable, column: str, limit: int = 3) -> list[str]:
    values = table.frame[column].dropna()
    seen: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
        if len(seen) >= limit:
            break
    return seen


def _numeric_thresholds(table: SyntheticTable, column: str) -> list[float]:
    values = pd.to_numeric(table.frame[column], errors="coerce").dropna()
    if values.empty:
        return []
    quantiles = [0.25, 0.5, 0.75]
    return sorted({round(float(values.quantile(q)), 2) for q in quantiles})


def _numeric_pairs(columns: list[str], limit: int = 2) -> list[tuple[str, str]]:
    """取前几对不同的数值列组合，避免组合数爆炸。"""
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            pairs.append((left, right))
            if len(pairs) >= limit:
                return pairs
    return pairs


def _candidates(table: SyntheticTable, *, max_steps: int) -> list[list[dict[str, Any]]]:
    """系统地枚举候选操作链。

    旧实现每张表只产出十几种链，导致上千条样本里唯一操作链不到百条。
    这里按"列 × 操作符 × 参数 × 步数"组合展开，让规则生成也能有足够多样性。
    """
    numeric = _numeric_columns(table)
    text = _columns_by_kind(table, "text")
    datetimes = _columns_by_kind(table, "datetime")

    chains: list[list[dict[str, Any]]] = []
    seen: set[str] = set()

    def add(chain: list[dict[str, Any]]) -> None:
        if not chain or len(chain) > max_steps:
            return
        key = json.dumps(chain, ensure_ascii=False, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        chains.append(chain)

    # ---- 单步：筛选 ----
    for column in text:
        for value in _distinct_values(table, column):
            add([{"tool": "filter_rows", "args": {"filters": [
                {"column": column, "operator": "eq", "value": value}]}}])
            add([{"tool": "filter_rows", "args": {"filters": [
                {"column": column, "operator": "ne", "value": value}]}}])
        for operator in ("is_empty", "not_empty"):
            add([{"tool": "filter_rows", "args": {"filters": [
                {"column": column, "operator": operator}]}}])

    for column in numeric:
        for threshold in _numeric_thresholds(table, column):
            for operator in ("gt", "gte", "lt", "lte"):
                add([{"tool": "filter_rows", "args": {"filters": [
                    {"column": column, "operator": operator, "value": threshold}]}}])

    # 多条件筛选（AND）
    if len(numeric) >= 2:
        left, right = numeric[0], numeric[1]
        left_threshold = (_numeric_thresholds(table, left) or [0])[0]
        right_threshold = (_numeric_thresholds(table, right) or [0])[0]
        add([{"tool": "filter_rows", "args": {"filters": [
            {"column": left, "operator": "gte", "value": left_threshold},
            {"column": right, "operator": "lte", "value": right_threshold}]}}])
    if text and numeric:
        value = (_distinct_values(table, text[0]) or [None])[0]
        if value is not None:
            threshold = (_numeric_thresholds(table, numeric[0]) or [0])[0]
            add([{"tool": "filter_rows", "args": {"filters": [
                {"column": text[0], "operator": "eq", "value": value},
                {"column": numeric[0], "operator": "gt", "value": threshold}]}}])

    # ---- 单步：排序 / TopN ----
    for column in numeric + datetimes:
        for ascending in (True, False):
            add([{"tool": "sort_rows", "args": {"sort_by": [
                {"column": column, "ascending": ascending}]}}])
    if len(numeric) >= 2:
        add([{"tool": "sort_rows", "args": {"sort_by": [
            {"column": numeric[0], "ascending": False},
            {"column": numeric[1], "ascending": True}]}}])
    if datetimes and numeric:
        add([{"tool": "sort_rows", "args": {"sort_by": [
            {"column": datetimes[0], "ascending": True},
            {"column": numeric[0], "ascending": False}]}}])

    for column in numeric:
        for limit in (5, 10, 20):
            for ascending in (False, True):
                add([{"tool": "top_n_rows", "args": {
                    "sort_by": [{"column": column, "ascending": ascending}],
                    "limit": limit}}])

    # ---- 单步：汇总 ----
    for column in numeric:
        for function in ("sum", "mean", "min", "max", "count", "nunique"):
            add([{"tool": "aggregate_data", "args": {
                "metrics": [{"column": column, "function": function}]}}])
    for group in text:
        for column in numeric:
            for function in ("sum", "mean", "count"):
                add([{"tool": "aggregate_data", "args": {
                    "group_by": [group],
                    "metrics": [{"column": column, "function": function}]}}])
    if len(text) >= 2 and numeric:
        add([{"tool": "aggregate_data", "args": {
            "group_by": text[:2],
            "metrics": [{"column": numeric[0], "function": "sum"}]}}])

    # ---- 单步：去重 / 填补缺失 / 描述统计 ----
    for column in text:
        add([{"tool": "deduplicate_rows", "args": {"subset": [column]}}])
    add([{"tool": "deduplicate_rows", "args": {}}])

    for column in text + numeric:
        add([{"tool": "fill_missing_values", "args": {
            "fill_strategy": "ffill", "subset": [column]}}])
        add([{"tool": "fill_missing_values", "args": {
            "fill_strategy": "bfill", "subset": [column]}}])
    for column in text:
        add([{"tool": "fill_missing_values", "args": {
            "fill_strategy": "value", "subset": [column], "fill_value": "未知"}}])
    for column in numeric:
        add([{"tool": "fill_missing_values", "args": {
            "fill_strategy": "value", "subset": [column], "fill_value": 0}}])

    add([{"tool": "describe_data", "args": {}}])

    # ---- 单步：计算列 ----
    for left in numeric:
        for right in numeric:
            if left == right:
                continue
            for operator in ("add", "subtract", "multiply", "divide"):
                add([{"tool": "add_computed_column", "args": {
                    "name": f"{left}_{operator}_{right}",
                    "operator": operator,
                    "left": left,
                    "right_column": right,
                    "round_digits": 2,
                }}])
        for operator, constant in (("multiply", 1.13), ("add", 100)):
            add([{"tool": "add_computed_column", "args": {
                "name": f"{left}_{operator}_{constant}",
                "operator": operator,
                "left": left,
                "right_value": constant,
                "round_digits": 2,
            }}])

    # ---- 单步：图表 ----
    # 单步绘图只在"行粒度本身就是合适的横轴"时成立：表行数不多，或类别数接近行数。
    # 若一行代表一条明细、类别却只有几个，直接按行画图会产生大量重复堆叠，
    # 正确做法是先汇总（见下面的多步组合）。
    row_count = len(table.frame)

    for column in numeric:
        for chart_type in ("bar", "line", "pie"):
            add([{"tool": "create_chart", "args": {
                "chart_type": chart_type, "chart_y": column}}])
    for group in text:
        distinct = int(table.frame[group].nunique())
        if distinct == 0:
            continue
        # 一行一条明细、类别却只有几个：直接绘图会把大量重复堆叠在一起，应先汇总。
        if row_count > max(distinct * 3, 12):
            continue
        # 每行都不同的列是订单号/编号这类标识符，不适合做分类轴。
        if distinct == row_count and row_count > 12:
            continue
        for column in numeric:
            for chart_type in ("bar", "line", "pie"):
                add([{"tool": "create_chart", "args": {
                    "chart_type": chart_type,
                    "chart_x": group,
                    "chart_y": column}}])
    if len(numeric) >= 2:
        add([{"tool": "create_chart", "args": {
            "chart_type": "scatter", "chart_x": numeric[0], "chart_y": numeric[1]}}])

    # ---- 多步组合 ----
    for group in text:
        for column in numeric:
            for function in ("sum", "mean"):
                metric_name = f"{column}_{function}"
                add([
                    {"tool": "aggregate_data", "args": {
                        "group_by": [group],
                        "metrics": [{"column": column, "function": function,
                                     "name": metric_name}]}},
                    {"tool": "sort_rows", "args": {
                        "sort_by": [{"column": metric_name, "ascending": False}]}},
                ])

                # 先按类别汇总，再用汇总结果绘图：原始行粒度与图表语义不匹配时，
                # 这是唯一合理的做法，也是最常见的真实需求。
                for chart_type in ("bar", "line", "pie"):
                    add([
                        {"tool": "aggregate_data", "args": {
                            "group_by": [group],
                            "metrics": [{"column": column, "function": function,
                                         "name": metric_name}]}},
                        {"tool": "create_chart", "args": {
                            "chart_type": chart_type,
                            "chart_x": group,
                            "chart_y": metric_name}},
                    ])

                # 汇总 → 排序 → 绘图：柱状图按数值排列，是报表里的常见形态。
                add([
                    {"tool": "aggregate_data", "args": {
                        "group_by": [group],
                        "metrics": [{"column": column, "function": function,
                                     "name": metric_name}]}},
                    {"tool": "sort_rows", "args": {
                        "sort_by": [{"column": metric_name, "ascending": False}]}},
                    {"tool": "create_chart", "args": {
                        "chart_type": "bar",
                        "chart_x": group,
                        "chart_y": metric_name}},
                ])

                add([
                    {"tool": "aggregate_data", "args": {
                        "group_by": [group],
                        "metrics": [{"column": column, "function": function,
                                     "name": metric_name}]}},
                    {"tool": "top_n_rows", "args": {
                        "sort_by": [{"column": metric_name, "ascending": False}],
                        "limit": 5}},
                ])

    # 汇总两个指标后再绘图：横轴分组、纵轴其一，另一个留作对照。
    for group in text:
        for left, right in _numeric_pairs(numeric):
            for function in ("sum", "mean"):
                left_name = f"{left}_{function}"
                right_name = f"{right}_{function}"
                add([
                    {"tool": "aggregate_data", "args": {
                        "group_by": [group],
                        "metrics": [
                            {"column": left, "function": function, "name": left_name},
                            {"column": right, "function": function, "name": right_name},
                        ]}},
                    {"tool": "create_chart", "args": {
                        "chart_type": "bar",
                        "chart_x": group,
                        "chart_y": left_name}},
                ])

    # 筛选后汇总再绘图：先限定范围再展示，是分析类需求的典型链条。
    if text and numeric:
        for value in _distinct_values(table, text[0], limit=2):
            for chart_type in ("bar", "line"):
                add([
                    {"tool": "filter_rows", "args": {"filters": [
                        {"column": text[0], "operator": "eq", "value": value}]}},
                    {"tool": "aggregate_data", "args": {
                        "group_by": [text[0]],
                        "metrics": [{"column": numeric[0], "function": "sum",
                                     "name": f"{numeric[0]}_sum"}]}},
                    {"tool": "create_chart", "args": {
                        "chart_type": chart_type,
                        "chart_x": text[0],
                        "chart_y": f"{numeric[0]}_sum"}},
                ])

    for column in numeric:
        add([
            {"tool": "filter_rows", "args": {"filters": [
                {"column": column, "operator": "not_empty"}]}},
            {"tool": "top_n_rows", "args": {
                "sort_by": [{"column": column, "ascending": False}], "limit": 10}},
        ])
        # 注意：不要再生成 sort_rows → top_n_rows。top_n_rows 内部已按同样字段排序，
        # 前一步的排序对结果没有任何影响，属于冗余标签，会让"序列一致"失去意义。

    for left in numeric:
        for right in numeric:
            if left == right:
                continue
            name = f"{left}_x_{right}"
            add([
                {"tool": "add_computed_column", "args": {
                    "name": name, "operator": "multiply",
                    "left": left, "right_column": right, "round_digits": 2}},
                {"tool": "top_n_rows", "args": {
                    "sort_by": [{"column": name, "ascending": False}], "limit": 10}},
            ])
            if text:
                add([
                    {"tool": "add_computed_column", "args": {
                        "name": name, "operator": "multiply",
                        "left": left, "right_column": right, "round_digits": 2}},
                    {"tool": "aggregate_data", "args": {
                        "group_by": [text[0]],
                        "metrics": [{"column": name, "function": "sum"}]}},
                ])

    if text and numeric:
        value = (_distinct_values(table, text[0]) or [None])[0]
        if value is not None:
            add([
                {"tool": "filter_rows", "args": {"filters": [
                    {"column": text[0], "operator": "eq", "value": value}]}},
                {"tool": "aggregate_data", "args": {
                    "group_by": [text[0]],
                    "metrics": [{"column": numeric[0], "function": "sum"}]}},
                {"tool": "sort_rows", "args": {"sort_by": [
                    {"column": f"{numeric[0]}_sum", "ascending": False}]}},
            ])
            add([
                {"tool": "filter_rows", "args": {"filters": [
                    {"column": text[0], "operator": "eq", "value": value}]}},
                {"tool": "top_n_rows", "args": {
                    "sort_by": [{"column": numeric[0], "ascending": False}],
                    "limit": 10}},
            ])

    for group in text:
        add([
            {"tool": "fill_missing_values", "args": {
                "fill_strategy": "ffill", "subset": [group]}},
            {"tool": "aggregate_data", "args": {
                "group_by": [group],
                "metrics": [{"column": numeric[0], "function": "count"}]}},
        ] if numeric else [])

    return chains
