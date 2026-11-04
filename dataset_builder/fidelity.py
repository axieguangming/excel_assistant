"""语义等价比对：区分真正的语义漂移与等价的命名/写法差异。

严格相等会把大量**结果完全相同**的链判成错误，导致忠实度被严重低估。已确认的
等价情况：

1. 解析器省略可选的 ``metrics[].name``，代码自动按「字段_函数」命名，
   而期望链显式写了业务列名（如「预算合计」）。两者结果数据完全相同。
2. 计算列的新列名不同（``推荐意愿_add_100`` vs ``推荐意愿_加100``）。
3. ``create_chart.chart_title`` 只是图上文字。
4. ``top_n_rows`` 之前多写一步键相同的 ``sort_rows``（``top_n_rows`` 内部本就排序）。
5. 解析器补/省略 ``add_computed_column.round_digits``（问题里往往不提取整）。

做法：把两条链都规范化成「数据语义」表示——聚合列与计算列统一成与列名文字
无关的内部标记，后续步骤对列的引用同步替换成同一标记。这样列名差异被抹平，
而聚合函数、分组、筛选、排序等真正影响数据的参数仍会被严格比较。
"""

from __future__ import annotations

import json
from typing import Any


def _default_metric_name(metric: dict[str, Any]) -> str:
    return f"{metric.get('column')}_{metric.get('function')}"


def _drop_redundant_sorts(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """删除紧邻 top_n_rows 之前、排序键完全相同的 sort_rows。"""
    result: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if step.get("tool") == "sort_rows":
            following = steps[index + 1] if index + 1 < len(steps) else None
            if following and following.get("tool") == "top_n_rows":
                left = (step.get("args") or {}).get("sort_by")
                right = (following.get("args") or {}).get("sort_by")
                if left == right:
                    continue
        result.append(step)
    return result


def _rename(value: Any, mapping: dict[str, str]) -> Any:
    """把列引用替换成规范化标记；未命中的保持原样。"""
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_rename(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _rename(item, mapping) for key, item in value.items()}
    return value


def normalize_for_comparison(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把一条链规范到「结果数据语义」层面。"""
    cleaned: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}  # 旧列名 -> 规范标记
    computed_index = 0
    aggregate_index = 0

    for step in steps:
        tool = step.get("tool")
        args = dict(step.get("args") or {})

        # describe_data 只读，不改变数据。问题里说"告诉我多少行多少列"时，
        # 解析器常额外补一步，这不应算语义漂移。
        if tool == "describe_data":
            continue

        if tool == "aggregate_data":
            metrics = [m for m in (args.get("metrics") or []) if isinstance(m, dict)]
            group_by = [str(name) for name in args.get("group_by") or []]
            renamed_metrics = []
            for metric in metrics:
                column = _rename(str(metric.get("column")), mapping)
                function = str(metric.get("function"))
                # count 是「组内行数」，与数的是哪一列无关（见 excel_ops._aggregate），
                # 因此 count(反馈) 与 count(受访者编号) 结果完全相同，比对时抹平列名。
                if function == "count":
                    column = "*"
                # 标记只依赖「第几个聚合指标 + 字段 + 函数」，与列名文字无关。
                token = f"#agg{aggregate_index}:{column}:{function}"
                aliases = {
                    metric.get("name"),
                    _default_metric_name(metric),
                }
                for alias in aliases:
                    if isinstance(alias, str) and alias:
                        mapping[alias] = token
                renamed_metrics.append(
                    {"token": token, "column": column, "function": function}
                )
                aggregate_index += 1
            cleaned.append(
                {
                    "tool": tool,
                    "args": {
                        "group_by": [_rename(name, mapping) for name in group_by],
                        "metrics": renamed_metrics,
                    },
                }
            )
            continue

        if tool == "add_computed_column":
            computed_index += 1
            token = f"#col{computed_index}"
            name = args.pop("name", None)
            if isinstance(name, str) and name:
                mapping[name] = token
            # 问题通常不提取整要求，它不改变「做了什么运算」。
            args.pop("round_digits", None)
            for key in ("left", "right_column"):
                if args.get(key):
                    args[key] = _rename(str(args[key]), mapping)
            cleaned.append({"tool": tool, "args": args})
            continue

        if tool == "create_chart":
            args.pop("chart_title", None)

        cleaned.append({"tool": tool, "args": _rename(args, mapping)})

    return _drop_redundant_sorts(cleaned)


def canonical_json(steps: list[dict[str, Any]]) -> str:
    return json.dumps(
        normalize_for_comparison(steps),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def is_semantically_equivalent(
    expected: list[dict[str, Any]],
    parsed: list[dict[str, Any]],
) -> bool:
    return canonical_json(expected) == canonical_json(parsed)
