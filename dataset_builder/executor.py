"""用项目自身的工具算法执行操作链，产出可验证的真值。

执行过程与 Agent 调用工具时走同一套算法（`excel_ops` 中的私有函数），
因此标签与线上行为一致，同时不必构造 LangGraph 状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

import excel_ops
from excel_ops import (
    _aggregate,
    _apply_filters,
    _computed_column,
    _deduplicate_rows,
    _describe_data,
    _fill_missing_values,
    _sort_rows,
    _top_n_rows,
    _validate_columns,
)
from schemas import (
    AggregateDataInput,
    ComputedColumnInput,
    CreateChartInput,
    DeduplicateRowsInput,
    FillMissingInput,
    FilterRowsInput,
    SortRowsInput,
    TopNRowsInput,
)

RUNTIME_ERRORS = (ValueError, TypeError, KeyError, OverflowError)


@dataclass
class StepOutcome:
    tool: str
    args: dict[str, Any]
    ok: bool
    rows: int | None = None
    columns: list[str] | None = None
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"tool": self.tool, "ok": self.ok}
        if self.ok:
            payload["rows"] = self.rows
            payload["columns"] = self.columns
        else:
            payload["error"] = self.error
        return payload


@dataclass
class ExecutionOutcome:
    frame: pd.DataFrame | None
    steps: list[StepOutcome] = field(default_factory=list)
    chart: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.frame is not None

    def step_dicts(self) -> list[dict[str, Any]]:
        return [step.as_dict() for step in self.steps]


def _validate_chart(df: pd.DataFrame, chart: CreateChartInput) -> None:
    columns = [chart.chart_y, *([chart.chart_x] if chart.chart_x else [])]
    _validate_columns(df, columns)

    values = pd.to_numeric(df[chart.chart_y], errors="coerce")
    if values.notna().sum() == 0:
        raise ValueError(f"图表字段“{chart.chart_y}”没有可用数值。")

    if chart.chart_type == "scatter" and chart.chart_x:
        x_values = pd.to_numeric(df[chart.chart_x], errors="coerce")
        if x_values.notna().sum() == 0:
            raise ValueError(f"图表字段“{chart.chart_x}”没有可用数值。")

    if chart.chart_type == "pie":
        valid = values.notna()
        if (values[valid] < 0).any() or float(values[valid].sum()) <= 0:
            raise ValueError("饼图需要非负且总和大于零的数据。")


def _render_chart(df: pd.DataFrame, chart: CreateChartInput, chart_dir: Path) -> str:
    chart_dir.mkdir(parents=True, exist_ok=True)
    previous = excel_ops.OUTPUT_DIR
    excel_ops.OUTPUT_DIR = str(chart_dir)
    try:
        return excel_ops._create_chart(df, chart)
    finally:
        excel_ops.OUTPUT_DIR = previous


def _apply_step(
    df: pd.DataFrame,
    step: Mapping[str, Any],
    *,
    render_charts: bool,
    chart_dir: Path | None,
) -> tuple[pd.DataFrame, str | None]:
    tool = step["tool"]
    args = step["args"]

    if tool == "filter_rows":
        parsed = FilterRowsInput.model_validate(args)
        return _apply_filters(df, parsed.filters), None
    if tool == "sort_rows":
        parsed = SortRowsInput.model_validate(args)
        return _sort_rows(df, parsed.sort_by), None
    if tool == "aggregate_data":
        parsed = AggregateDataInput.model_validate(args)
        return _aggregate(df, parsed.group_by, parsed.metrics), None
    if tool == "top_n_rows":
        parsed = TopNRowsInput.model_validate(args)
        return _top_n_rows(df, parsed.sort_by, parsed.limit), None
    if tool == "deduplicate_rows":
        parsed = DeduplicateRowsInput.model_validate(args)
        return _deduplicate_rows(df, parsed.subset), None
    if tool == "fill_missing_values":
        parsed = FillMissingInput.model_validate(args)
        return (
            _fill_missing_values(
                df,
                parsed.fill_strategy,
                parsed.subset,
                parsed.fill_value,
            ),
            None,
        )
    if tool == "add_computed_column":
        parsed = ComputedColumnInput.model_validate(args)
        return _computed_column(df, parsed), None
    if tool == "describe_data":
        return _describe_data(df), None
    if tool == "create_chart":
        parsed = CreateChartInput.model_validate(args)
        _validate_chart(df, parsed)
        if render_charts and chart_dir is not None:
            return df, _render_chart(df, parsed, chart_dir)
        return df, None

    raise ValueError(f"未知工具：{tool}")


def execute_chain(
    frame: pd.DataFrame,
    steps: list[Mapping[str, Any]],
    *,
    render_charts: bool = False,
    chart_dir: str | Path | None = None,
) -> ExecutionOutcome:
    """逐步执行操作链；失败时返回已完成的步骤与错误原因。"""
    current = frame.copy()
    outcomes: list[StepOutcome] = []
    resolved_chart_dir = Path(chart_dir) if chart_dir is not None else None
    chart_url: str | None = None

    for index, step in enumerate(steps, start=1):
        tool = str(step["tool"])
        args = dict(step.get("args", {}))
        try:
            current, step_chart = _apply_step(
                current,
                step,
                render_charts=render_charts,
                chart_dir=resolved_chart_dir,
            )
        except RUNTIME_ERRORS as error:
            message = str(error) or error.__class__.__name__
            outcomes.append(
                StepOutcome(tool=tool, args=args, ok=False, error=message)
            )
            return ExecutionOutcome(
                frame=None,
                steps=outcomes,
                chart=chart_url,
                error=f"第 {index} 步（{tool}）失败：{message}",
            )

        if step_chart:
            chart_url = step_chart
        outcomes.append(
            StepOutcome(
                tool=tool,
                args=args,
                ok=True,
                rows=int(len(current)),
                columns=[str(column) for column in current.columns],
            )
        )

    return ExecutionOutcome(frame=current, steps=outcomes, chart=chart_url)
