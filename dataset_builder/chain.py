"""操作链的规范化与结构校验。

操作链是与 Agent 工具一一对应的结构化步骤序列。这里的校验只检查结构
（工具名、参数 schema、步数上限）；列是否存在、类型是否适用等可达性
问题由执行阶段发现，见 `executor.py`。
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ValidationError

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

# 与 excel_ops.EXCEL_TOOLS 保持一致；describe_data 不接受参数。
TOOL_SCHEMAS: dict[str, type[BaseModel] | None] = {
    "filter_rows": FilterRowsInput,
    "sort_rows": SortRowsInput,
    "aggregate_data": AggregateDataInput,
    "top_n_rows": TopNRowsInput,
    "deduplicate_rows": DeduplicateRowsInput,
    "fill_missing_values": FillMissingInput,
    "add_computed_column": ComputedColumnInput,
    "describe_data": None,
    "create_chart": CreateChartInput,
}

TOOL_NAMES: tuple[str, ...] = tuple(TOOL_SCHEMAS)
MAX_CHAIN_STEPS = 10


class ChainValidationError(ValueError):
    """操作链不符合工具契约。"""


def _as_step_list(payload: Any) -> list[Any]:
    if isinstance(payload, Mapping):
        steps = payload.get("steps")
    else:
        steps = payload
    if not isinstance(steps, list):
        raise ChainValidationError("操作链必须是步骤数组，或包含 steps 数组的对象。")
    return steps


def _normalize_step(step: Any, index: int, allowed_tools: Iterable[str]) -> dict[str, Any]:
    if not isinstance(step, Mapping):
        raise ChainValidationError(f"第 {index} 步必须是对象。")

    tool = step.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        raise ChainValidationError(f"第 {index} 步缺少 tool 字段。")
    tool = tool.strip()
    if tool not in TOOL_SCHEMAS:
        raise ChainValidationError(f"第 {index} 步使用了未知工具：{tool}")
    if tool not in allowed_tools:
        raise ChainValidationError(f"第 {index} 步使用了不允许的工具：{tool}")

    raw_args = step.get("args", {})
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, Mapping):
        raise ChainValidationError(f"第 {index} 步的 args 必须是对象。")

    schema = TOOL_SCHEMAS[tool]
    if schema is None:
        if raw_args:
            raise ChainValidationError(f"工具 {tool} 不接受参数。")
        return {"tool": tool, "args": {}}

    try:
        parsed = schema.model_validate(dict(raw_args))
    except ValidationError as error:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'args'}: {item['msg']}"
            for item in error.errors()
        )
        raise ChainValidationError(f"第 {index} 步 {tool} 参数无效：{detail}") from error

    return {"tool": tool, "args": parsed.model_dump(exclude_none=True)}


def normalize_chain(
    payload: Any,
    *,
    max_steps: int = MAX_CHAIN_STEPS,
    allowed_tools: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """把外部输入（模型输出）规范化为可执行、可比较的步骤列表。"""
    steps = _as_step_list(payload)
    if not steps:
        raise ChainValidationError("操作链不能为空。")
    if max_steps < 1:
        raise ValueError("max_steps 必须大于 0。")
    if len(steps) > max_steps:
        raise ChainValidationError(
            f"操作链包含 {len(steps)} 步，超过上限 {max_steps} 步。"
        )

    allowed = tuple(allowed_tools) if allowed_tools is not None else TOOL_NAMES
    return [
        _normalize_step(step, index, allowed)
        for index, step in enumerate(steps, start=1)
    ]


def _values_equivalent(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _values_equivalent(item_left, item_right, tolerance)
            for item_left, item_right in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(
            _values_equivalent(left[key], right[key], tolerance) for key in left
        )
    return left == right


def chains_equivalent(
    left: Iterable[Mapping[str, Any]],
    right: Iterable[Mapping[str, Any]],
    *,
    tolerance: float = 1e-6,
) -> bool:
    """比较两条已规范化的操作链，数值允许浮点误差。"""
    left_steps = list(left)
    right_steps = list(right)
    if len(left_steps) != len(right_steps):
        return False
    return all(
        _values_equivalent(step_left, step_right, tolerance)
        for step_left, step_right in zip(left_steps, right_steps)
    )


def canonical_chain(steps: Iterable[Mapping[str, Any]]) -> str:
    """用于去重的稳定字符串表示。"""
    return json.dumps(list(steps), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
