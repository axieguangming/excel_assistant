from __future__ import annotations

from collections import Counter
import json
import os
import sys
from typing import Annotated, Any, Callable
import uuid

import matplotlib
import pandas as pd
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app_paths import DATA_DIR
from schemas import (
    AggregateDataInput,
    ComputedColumnInput,
    CreateChartInput,
    DeduplicateRowsInput,
    FilterRowsInput,
    FillMissingInput,
    Metric,
    SortKey,
    SortRowsInput,
    TopNRowsInput,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = str(DATA_DIR / "outputs")


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    duplicates = [column for column, count in Counter(df.columns).items() if count > 1]
    if duplicates:
        raise ValueError(f"数据中存在重复字段名：{duplicates}。请先整理表头后再分析。")

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"找不到字段：{missing}。可用字段：{list(df.columns)}")


def _apply_filters(df: pd.DataFrame, filters: list[Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for condition in filters:
        _validate_columns(df, [condition.column])
        values = df[condition.column]
        if condition.operator == "eq":
            current = values.eq(condition.value)
        elif condition.operator == "ne":
            current = values.ne(condition.value)
        elif condition.operator in {"gt", "gte", "lt", "lte"}:
            if pd.api.types.is_datetime64_any_dtype(values.dtype):
                comparable = pd.to_datetime(values, errors="coerce")
                target = pd.to_datetime(condition.value, errors="coerce")
            else:
                comparable = pd.to_numeric(values, errors="coerce")
                try:
                    target = float(condition.value)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"字段“{condition.column}”的数值筛选条件无效：{condition.value!r}"
                    ) from error
            if pd.isna(target):
                raise ValueError(
                    f"字段“{condition.column}”的筛选条件无效：{condition.value!r}"
                )
            current = {
                "gt": comparable.gt,
                "gte": comparable.ge,
                "lt": comparable.lt,
                "lte": comparable.le,
            }[condition.operator](target)
        elif condition.operator == "contains":
            current = values.astype("string").str.contains(
                str(condition.value), regex=False, na=False,
            )
        elif condition.operator == "not_contains":
            current = ~values.astype("string").str.contains(
                str(condition.value), regex=False, na=False,
            )
        elif condition.operator == "is_empty":
            current = values.isna() | values.astype("string").str.strip().eq("")
        else:
            current = values.notna() & values.astype("string").str.strip().ne("")
        mask &= current.fillna(False)
    return df.loc[mask].copy()


def _aggregate(
    df: pd.DataFrame,
    group_by: list[str],
    metrics: list[Metric],
) -> pd.DataFrame:
    columns = [*group_by, *(metric.column for metric in metrics)]
    _validate_columns(df, columns)
    for metric in metrics:
        if metric.function not in {"sum", "mean"}:
            continue
        numeric = pd.to_numeric(df[metric.column], errors="coerce")
        if df[metric.column].notna().any() and numeric.notna().sum() == 0:
            raise ValueError(
                f"字段“{metric.column}”不包含可用于 {metric.function} 的数值。"
            )
        df[metric.column] = numeric

    named_aggregations: dict[str, tuple[str, str]] = {}
    for index, metric in enumerate(metrics, start=1):
        name = metric.name or f"{metric.column}_{metric.function}"
        while name in group_by or name in named_aggregations:
            name = f"{name}_{index}"
        # count 的语义是「这一组有多少条记录」，与数的是哪一列无关。
        # pandas 的 "size" 正好表示组内行数，且不跳过空值。
        # 若沿用 "count"（即 Series.count），含空值的列会少算，同一句
        # 「有多少条反馈」会因模型选了哪一列而得到不同结果。
        agg_function = "size" if metric.function == "count" else metric.function
        named_aggregations[name] = (metric.column, agg_function)

    if group_by:
        return (
            df.groupby(group_by, dropna=False, sort=False)
            .agg(**named_aggregations)
            .reset_index()
        )

    values: dict[str, list[Any]] = {}
    for name, (column, function) in named_aggregations.items():
        series = df[column]
        if function == "sum":
            value = series.sum()
        elif function == "mean":
            value = series.mean()
        elif function == "min":
            value = series.min()
        elif function == "max":
            value = series.max()
        elif function == "size":
            value = len(df)
        else:
            value = series.nunique(dropna=True)
        values[name] = [value]
    return pd.DataFrame(values)


def _sort_rows(df: pd.DataFrame, sort_by: list[SortKey]) -> pd.DataFrame:
    columns = [sort_key.column for sort_key in sort_by]
    _validate_columns(df, columns)
    return df.sort_values(
        by=columns,
        ascending=[sort_key.ascending for sort_key in sort_by],
        na_position="last",
    ).reset_index(drop=True)


def _top_n_rows(
    df: pd.DataFrame,
    sort_by: list[SortKey],
    limit: int,
) -> pd.DataFrame:
    return _sort_rows(df, sort_by).head(limit).reset_index(drop=True)


def _deduplicate_rows(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    columns = subset or list(df.columns)
    _validate_columns(df, columns)
    return df.drop_duplicates(subset=columns, keep="first").reset_index(drop=True)


def _fill_missing_values(
    df: pd.DataFrame,
    strategy: str,
    subset: list[str],
    fill_value: Any,
) -> pd.DataFrame:
    columns = subset or list(df.columns)
    _validate_columns(df, columns)
    if strategy == "ffill":
        df[columns] = df[columns].ffill()
    elif strategy == "bfill":
        df[columns] = df[columns].bfill()
    else:
        df[columns] = df[columns].fillna(fill_value)
    return df


def _describe_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").transpose().reset_index(names="字段")


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    values = df[column]
    if (
        pd.api.types.is_datetime64_any_dtype(values)
        or pd.api.types.is_timedelta64_dtype(values)
    ):
        raise ValueError(f"字段“{column}”是日期或时间类型，不能参与数值计算。")
    numeric = pd.to_numeric(values, errors="coerce")
    if values.notna().any() and numeric.notna().sum() == 0:
        raise ValueError(f"字段“{column}”不包含可用于计算的数值。")
    return numeric


def _computed_column(df: pd.DataFrame, computed: ComputedColumnInput) -> pd.DataFrame:
    name = computed.name
    if name in df.columns:
        raise ValueError(f"字段“{name}”已存在，请为新列使用其他名称。")

    columns = [computed.left, *([computed.right_column] if computed.right_column else [])]
    _validate_columns(df, columns)

    left_values = _numeric_series(df, computed.left)
    if computed.right_column:
        right: pd.Series | float = _numeric_series(df, computed.right_column)
    else:
        right = float(computed.right_value)

    if computed.operator == "add":
        result = left_values + right
    elif computed.operator == "subtract":
        result = left_values - right
    elif computed.operator == "multiply":
        result = left_values * right
    else:
        if isinstance(right, pd.Series):
            if right.eq(0).any():
                raise ValueError(
                    f"字段“{computed.right_column}”包含 0，不能作为除数。"
                )
        elif right == 0:
            raise ValueError("除数不能为 0。")
        result = left_values / right

    result = result.astype(float).replace([float("inf"), float("-inf")], pd.NA)
    if computed.round_digits is not None:
        result = result.round(computed.round_digits)
    if left_values.notna().any() and result.notna().sum() == 0:
        raise ValueError(f"字段“{computed.left}”的计算结果全部为空，请检查参与计算的字段。")

    df[name] = result
    return df


def _frame_observation(df: pd.DataFrame, *, read_only: bool = False) -> str:
    preview = df.head(8).to_string(index=False, max_colwidth=40)
    if not preview:
        preview = "（无数据行）"
    if read_only:
        # 只读工具的产出不是新的工作表。若照旧写成「当前结果为 N 行」
        # 模型会误以为原始数据被替换掉了，从而放弃后续步骤。
        return (
            f"查看成功（只读，当前工作表数据未改变，后续步骤仍在原数据上继续）。\n"
            f"本次查看得到 {len(df)} 行 × {len(df.columns)} 列：\n"
            f"字段：{list(df.columns)}\n"
            f"内容：\n{preview}"
        )
    return (
        f"操作成功。当前结果为 {len(df)} 行 × {len(df.columns)} 列。\n"
        f"字段：{list(df.columns)}\n"
        f"样例：\n{preview}"
    )


def chart_observation(chart: "CreateChartInput", chart_url: str) -> str:
    """`create_chart` 成功时返回给模型的文本。

    单独抽成函数是为了让训练数据构造与线上工具共用同一份实现——否则会退化成
    「训练时返回数据预览、推理时返回图表地址」，模型学不到画图这一步的存在。
    """
    return f"图表生成成功：{chart_url}\n图表设置：{chart.model_dump()}"


def observation_for_step(
    tool_name: str,
    df: pd.DataFrame,
    *,
    arguments: dict[str, Any] | None = None,
    chart_url: str | None = None,
) -> str:
    """按工具语义生成「成功」时的观察文本。训练数据构造使用此函数。

    - 只读工具（`describe_data`）走只读措辞
    - `create_chart` 返回图表地址与设置，不像数据框工具那样返回行列表
    - 其余数据框工具返回结果预览
    """
    if tool_name == "describe_data":
        return _frame_observation(df, read_only=True)
    if tool_name == "create_chart":
        chart = CreateChartInput.model_validate(arguments or {})
        return chart_observation(chart, chart_url or "")
    return _frame_observation(df)


def _record_plan(
    state: dict[str, Any],
    tool_name: str,
    arguments: dict[str, Any],
    ok: bool,
    error: str = "",
) -> list[str]:
    entry = json.dumps(
        {
            "tool": tool_name,
            "arguments": arguments,
            "ok": ok,
            **({"error": error} if error else {}),
        },
        ensure_ascii=False,
    )
    _log_tool_sample(state, entry, ok, error)
    return [
        *state.get("plans", []),
        entry,
    ]


def _log_tool_sample(
    state: dict[str, Any],
    tool_call: str,
    ok: bool,
    error: str,
) -> None:
    if getattr(sys, "frozen", False):
        return
    if os.getenv("TRAIN_DATA_LOGGING", "").strip().lower() not in {"1", "true", "yes"}:
        return
    data_dir = os.getenv("TRAIN_DATA_DIR", "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "codegen_log.jsonl"), "a", encoding="utf-8") as file:
        file.write(json.dumps({
            "prompt": state.get("user_prompt", ""),
            "plan": tool_call,
            "ok": ok,
            "error": error,
        }, ensure_ascii=False) + "\n")


def _tool_command(
    state: dict[str, Any],
    tool_call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    content: str,
    *,
    ok: bool,
    result: pd.DataFrame | None = None,
    chart_file: str | None = None,
    read_only: bool = False,
) -> Command:
    update: dict[str, Any] = {
        "plans": _record_plan(
            state,
            tool_name,
            arguments,
            ok,
            error="" if ok else content,
        ),
        "tool_steps": state.get("tool_steps", 0) + 1,
        "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
    }
    if result is not None:
        update["last_result_rows"] = len(result)
        # result_df 是"最近一次产出"，用于展示/导出；只读工具也会更新它。
        update["result_df"] = result
        # current_df 是后续步骤的工作副本，只读工具（如 describe_data）不改动它。
        if not read_only:
            update["current_df"] = result
    if chart_file is not None:
        update["chart_file"] = chart_file
    return Command(update=update)


def _execute_dataframe_tool(
    state: dict[str, Any],
    tool_call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    operation: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    read_only: bool = False,
) -> Command:
    try:
        result = operation(state["current_df"].copy())
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        return _tool_command(
            state,
            tool_call_id,
            tool_name,
            arguments,
            f"工具执行失败：{error}",
            ok=False,
            read_only=read_only,
        )

    return _tool_command(
        state,
        tool_call_id,
        tool_name,
        arguments,
        _frame_observation(result, read_only=read_only),
        ok=True,
        result=result,
        read_only=read_only,
    )


@tool(args_schema=FilterRowsInput)
def filter_rows(
    filters: list,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """按一个或多个条件筛选当前工作表数据。多个条件必须同时满足。"""
    parsed_filters = FilterRowsInput(filters=filters).filters
    return _execute_dataframe_tool(
        state,
        tool_call_id,
        "filter_rows",
        {"filters": [condition.model_dump(exclude_none=True) for condition in parsed_filters]},
        lambda df: _apply_filters(df, parsed_filters),
    )


@tool(args_schema=SortRowsInput)
def sort_rows(
    sort_by: list,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """按一个或多个字段对当前数据排序。"""
    parsed_sort_by = SortRowsInput(sort_by=sort_by).sort_by
    return _execute_dataframe_tool(
        state,
        tool_call_id,
        "sort_rows",
        {"sort_by": [sort_key.model_dump() for sort_key in parsed_sort_by]},
        lambda df: _sort_rows(df, parsed_sort_by),
    )


@tool(args_schema=AggregateDataInput)
def aggregate_data(
    metrics: list,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    group_by: list[str] | None = None,
) -> Command:
    """对当前数据按字段分组，并计算总和、均值、最大值、最小值、行数或去重计数。

    count 表示该组有多少行记录（与指定列是否为空无关）；若需要「某列有多少个
    非空值」，请先用 filter_rows 过滤掉空值再 count。
    """
    parsed = AggregateDataInput(group_by=group_by or [], metrics=metrics)
    return _execute_dataframe_tool(
        state,
        tool_call_id,
        "aggregate_data",
        {
            "group_by": parsed.group_by,
            "metrics": [metric.model_dump(exclude_none=True) for metric in parsed.metrics],
        },
        lambda df: _aggregate(df, parsed.group_by, parsed.metrics),
    )


@tool(args_schema=TopNRowsInput)
def top_n_rows(
    sort_by: list,
    limit: int,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """按指定字段取当前数据的前 N 行。"""
    parsed = TopNRowsInput(sort_by=sort_by, limit=limit)
    return _execute_dataframe_tool(
        state,
        tool_call_id,
        "top_n_rows",
        {
            "sort_by": [sort_key.model_dump() for sort_key in parsed.sort_by],
            "limit": parsed.limit,
        },
        lambda df: _top_n_rows(df, parsed.sort_by, parsed.limit),
    )


@tool(args_schema=DeduplicateRowsInput)
def deduplicate_rows(
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    subset: list[str] | None = None,
) -> Command:
    """按指定字段去重；未指定字段时按整行去重，保留首次出现的行。"""
    parsed = DeduplicateRowsInput(subset=subset or [])
    return _execute_dataframe_tool(
        state,
        tool_call_id,
        "deduplicate_rows",
        {"subset": parsed.subset},
        lambda df: _deduplicate_rows(df, parsed.subset),
    )


@tool(args_schema=FillMissingInput)
def fill_missing_values(
    fill_strategy: str,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    subset: list[str] | None = None,
    fill_value: Any = None,
) -> Command:
    """填补当前数据中的空值；可向前填、向后填或填入指定值。"""
    parsed = FillMissingInput(
        fill_strategy=fill_strategy,
        subset=subset or [],
        fill_value=fill_value,
    )
    return _execute_dataframe_tool(
        state,
        tool_call_id,
        "fill_missing_values",
        parsed.model_dump(exclude_none=True),
        lambda df: _fill_missing_values(
            df,
            parsed.fill_strategy,
            parsed.subset,
            parsed.fill_value,
        ),
    )


@tool
def describe_data(
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """查看当前数据的描述统计（只读，不会改变后续步骤的数据）。"""
    return _execute_dataframe_tool(
        state,
        tool_call_id,
        "describe_data",
        {},
        _describe_data,
        read_only=True,
    )


def _category_labels(values: pd.Series) -> pd.Series:
    """转成纯字符串标签。

    matplotlib 的类别轴拒绝把 NaN（float）注册为类别，因此缺失值必须先换成
    占位字符串，否则"类别列有空值"这类很常见的数据会让所有柱状/折线/饼图崩溃。
    """
    return values.astype("string").fillna("(空)").astype(str)


def _create_chart(df: pd.DataFrame, chart: CreateChartInput) -> str:
    columns = [chart.chart_y, *([chart.chart_x] if chart.chart_x else [])]
    _validate_columns(df, columns)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(9, 5))
    try:
        labels = (
            _category_labels(df[chart.chart_x])
            if chart.chart_x
            else pd.Series(df.index.astype(str), index=df.index)
        )
        values = pd.to_numeric(df[chart.chart_y], errors="coerce")
        if values.notna().sum() == 0:
            raise ValueError(f"图表字段“{chart.chart_y}”没有可用数值。")

        if chart.chart_type == "bar":
            axis.bar(labels, values)
        elif chart.chart_type == "line":
            axis.plot(labels, values, marker="o")
        elif chart.chart_type == "scatter":
            x_values = pd.to_numeric(df[chart.chart_x], errors="coerce")
            valid = x_values.notna() & values.notna()
            axis.scatter(x_values[valid], values[valid])
            axis.set_xlabel(chart.chart_x)
        else:
            valid = values.notna()
            if (values[valid] < 0).any() or values[valid].sum() <= 0:
                raise ValueError("饼图需要非负且总和大于零的数据。")
            axis.pie(values[valid], labels=labels[valid], autopct="%1.1f%%")

        if chart.chart_type != "pie":
            axis.set_xlabel(chart.chart_x or "类别")
            axis.set_ylabel(chart.chart_y)
            axis.tick_params(axis="x", labelrotation=30)
        axis.set_title(chart.chart_title or chart.chart_y)
        figure.tight_layout()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"chart_{uuid.uuid4().hex[:12]}.png"
        figure.savefig(
            os.path.join(OUTPUT_DIR, filename),
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
    finally:
        plt.close(figure)
    return f"/api/chart/{filename}"


@tool(args_schema=ComputedColumnInput)
def add_computed_column(
    name: str,
    operator: str,
    left: str,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    right_column: str | None = None,
    right_value: float | None = None,
    round_digits: int | None = None,
) -> Command:
    """新增一个计算列，用加、减、乘、除组合两个数值字段或一个字段与常数。"""
    computed = ComputedColumnInput(
        name=name,
        operator=operator,
        left=left,
        right_column=right_column,
        right_value=right_value,
        round_digits=round_digits,
    )
    return _execute_dataframe_tool(
        state,
        tool_call_id,
        "add_computed_column",
        computed.model_dump(exclude_none=True),
        lambda df: _computed_column(df, computed),
    )


@tool(args_schema=CreateChartInput)
def create_chart(
    chart_type: str,
    chart_y: str,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    chart_x: str | None = None,
    chart_title: str | None = None,
) -> Command:
    """根据当前数据生成柱状图、折线图、饼图或散点图。"""
    chart = CreateChartInput(
        chart_type=chart_type,
        chart_y=chart_y,
        chart_x=chart_x,
        chart_title=chart_title,
    )
    current_df = state["current_df"]
    try:
        chart_url = _create_chart(current_df, chart)
        return _tool_command(
            state,
            tool_call_id,
            "create_chart",
            chart.model_dump(exclude_none=True),
            chart_observation(chart, chart_url),
            ok=True,
            chart_file=chart_url,
        )
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        return _tool_command(
            state,
            tool_call_id,
            "create_chart",
            chart.model_dump(exclude_none=True),
            f"图表生成失败：{error}",
            ok=False,
        )


EXCEL_TOOLS = [
    filter_rows,
    sort_rows,
    aggregate_data,
    top_n_rows,
    deduplicate_rows,
    fill_missing_values,
    add_computed_column,
    describe_data,
    create_chart,
]
