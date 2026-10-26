"""在冻结数据集上评测 Agent 与候选模型。

评测驱动的是**应用真实的 Agent 图**（`agent.invoke_agent`），而不是简化的
直接调用，因此结果能代表线上行为。

指标口径见 [docs/dataset-plan.md](../docs/dataset-plan.md) 第 8 节：
任务成功率是主指标，工具选择与参数准确率用于定位失败原因。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd

import agent
from app import build_df_info
from dataset_builder.chain import chains_equivalent, normalize_chain
from dataset_builder.pipeline import read_jsonl
from dataset_builder.tables import SyntheticTable, load_tables


@dataclass
class EvalConfig:
    split: str = "test"
    limit: int | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0


@dataclass
class SampleResult:
    sample_id: str
    workbook_id: str
    categories: list[str]
    difficulty: str
    task_success: bool = False
    task_success_unordered: bool = False
    tool_sequence_match: bool = False
    first_tool_match: bool = False
    args_match: bool = False
    planned_steps: int = 0
    tool_steps: int = 0
    turns: int = 0
    hit_turn_limit: bool = False
    failure: str = ""
    planned_tools: list[str] = field(default_factory=list)
    expects_chart: bool = False
    chart_produced: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "workbook_id": self.workbook_id,
            "categories": self.categories,
            "difficulty": self.difficulty,
            "task_success": self.task_success,
            "task_success_unordered": self.task_success_unordered,
            "tool_sequence_match": self.tool_sequence_match,
            "first_tool_match": self.first_tool_match,
            "args_match": self.args_match,
            "planned_steps": self.planned_steps,
            "tool_steps": self.tool_steps,
            "turns": self.turns,
            "hit_turn_limit": self.hit_turn_limit,
            "planned_tools": self.planned_tools,
            "expects_chart": self.expects_chart,
            "chart_produced": self.chart_produced,
            "failure": self.failure,
        }


@dataclass
class EvalReport:
    config: dict[str, Any]
    samples: list[SampleResult] = field(default_factory=list)

    @property
    def metrics(self) -> dict[str, Any]:
        total = len(self.samples)
        if not total:
            return {"samples": 0}

        def rate(predicate) -> float:
            return round(sum(1 for sample in self.samples if predicate(sample)) / total, 4)

        categories: dict[str, dict[str, Any]] = {}
        for sample in self.samples:
            for category in sample.categories:
                bucket = categories.setdefault(category, {"samples": 0, "success": 0})
                bucket["samples"] += 1
                bucket["success"] += int(sample.task_success)
        for bucket in categories.values():
            bucket["success_rate"] = round(bucket["success"] / bucket["samples"], 4)

        chart_samples = [sample for sample in self.samples if sample.expects_chart]
        return {
            "samples": total,
            "task_success_rate": rate(lambda s: s.task_success),
            "task_success_unordered_rate": rate(lambda s: s.task_success_unordered),
            "tool_sequence_match_rate": rate(lambda s: s.tool_sequence_match),
            "first_tool_match_rate": rate(lambda s: s.first_tool_match),
            "args_match_rate": rate(lambda s: s.args_match),
            "hit_turn_limit_rate": rate(lambda s: s.hit_turn_limit),
            "avg_tool_steps": round(
                sum(sample.tool_steps for sample in self.samples) / total, 2
            ),
            "avg_turns": round(sum(sample.turns for sample in self.samples) / total, 2),
            "chart_samples": len(chart_samples),
            "chart_produced_rate": (
                round(
                    sum(1 for sample in chart_samples if sample.chart_produced)
                    / len(chart_samples),
                    4,
                )
                if chart_samples
                else 0.0
            ),
            "by_category": dict(sorted(categories.items())),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "metrics": self.metrics,
            "failures": [
                {
                    "sample_id": sample.sample_id,
                    "failure": sample.failure,
                    "planned_tools": sample.planned_tools,
                }
                for sample in self.samples
                if not sample.task_success
            ],
            "samples": [sample.as_dict() for sample in self.samples],
        }


_DATE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?$"
)


def _format_timestamp(value: pd.Timestamp) -> str:
    if value.time() == pd.Timestamp("00:00:00").time():
        return value.date().isoformat()
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _cell_key(value: Any) -> str:
    """把单元格规范化成可比较的字符串，抹平 CSV 往返带来的表示差异。"""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return _format_timestamp(pd.Timestamp(value))
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value)
    # 日期列在 CSV 与 DataFrame 中的写法不同（2025-11-17 vs 2025-11-17T00:00:00）。
    if _DATE_PATTERN.match(text):
        parsed = pd.to_datetime(text, errors="coerce")
        if parsed is not None and not pd.isna(parsed):
            return _format_timestamp(pd.Timestamp(parsed))
    try:
        return f"{float(text):.6f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return text


def _rows(frame: pd.DataFrame, *, ordered: bool) -> tuple[tuple[str, ...], ...]:
    rows = [
        tuple(_cell_key(value) for value in row)
        for row in frame.itertuples(index=False, name=None)
    ]
    return tuple(rows) if ordered else tuple(sorted(rows))


def compare_frames(expected: pd.DataFrame, actual: pd.DataFrame) -> tuple[bool, str]:
    """返回 (有序一致, 不一致原因)。"""
    if expected.shape != actual.shape:
        return False, (
            f"形状不同：期望 {expected.shape[0]}×{expected.shape[1]}，"
            f"实际 {actual.shape[0]}×{actual.shape[1]}"
        )
    if _rows(expected, ordered=True) == _rows(actual, ordered=True):
        return True, ""
    if _rows(expected, ordered=False) == _rows(actual, ordered=False):
        return False, "行内容一致但顺序不同"
    return False, "数据内容不同"


def _plan_entries(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for raw in result.get("plans", []):
        try:
            entry = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _expected_steps(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        return normalize_chain(sample.get("expected_actions"), max_steps=32)
    except (ValueError, TypeError):
        return []


def _args_match(
    planned: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> bool:
    """参数准确率：只比较成功的工具调用，且要求逐位置一致。"""
    succeeded = [
        {"tool": entry.get("tool"), "args": entry.get("arguments", {})}
        for entry in planned
        if entry.get("ok")
    ]
    if len(succeeded) != len(expected):
        return False
    return all(
        chains_equivalent([left], [right])
        for left, right in zip(succeeded, expected)
    )


def _initial_state(df: pd.DataFrame, question: str) -> dict[str, Any]:
    return {
        "df": df,
        "df_info": build_df_info(df),
        "question": question,
        "history": [],
        "messages": [],
        "current_df": df.copy(),
        "plans": [],
        "tool_steps": 0,
        "turns": 0,
        "result_df": None,
        "final_answer": "",
        "chart_file": "",
        "code": "",
        "user_prompt": "",
    }


def _expects_chart(steps: list[dict[str, Any]]) -> bool:
    return any(step.get("tool") == "create_chart" for step in steps)


def _chart_ready(url: object) -> bool:
    """确认工具确实产出了一张已落盘的图表。"""
    if not isinstance(url, str) or not url.strip():
        return False
    filename = url.rstrip("/").rsplit("/", 1)[-1]
    if not filename:
        return False
    import excel_ops

    return (Path(excel_ops.OUTPUT_DIR) / filename).is_file()


def evaluate_sample(
    sample: Mapping[str, Any],
    tables: Mapping[str, SyntheticTable],
    *,
    dataset_root: Path,
) -> SampleResult:
    table = tables.get(str(sample.get("workbook_id")))
    result = SampleResult(
        sample_id=str(sample.get("sample_id")),
        workbook_id=str(sample.get("workbook_id")),
        categories=list(sample.get("task_categories") or []),
        difficulty=str(sample.get("difficulty") or ""),
    )
    if table is None:
        result.failure = "missing_table"
        return result

    expected_steps = _expected_steps(sample)
    result.expects_chart = _expects_chart(expected_steps)
    expected_artifact = (sample.get("expected_result") or {}).get("artifact")
    if not expected_artifact:
        result.failure = "missing_expected_result"
        return result
    artifact_path = dataset_root / str(expected_artifact)
    if not artifact_path.is_file():
        result.failure = f"missing_artifact: {artifact_path.name}"
        return result
    expected_frame = pd.read_csv(artifact_path, encoding="utf-8-sig")

    outcome = agent.invoke_agent(
        _initial_state(table.frame, str(sample.get("user_request", "")))
    )

    planned = _plan_entries(outcome)
    planned_tools = [str(entry.get("tool")) for entry in planned]
    result.planned_tools = planned_tools
    result.planned_steps = len(planned)
    result.tool_steps = int(outcome.get("tool_steps", 0))
    result.turns = int(outcome.get("turns", 0))
    result.hit_turn_limit = result.turns >= agent.MAX_AGENT_TURNS
    result.first_tool_match = bool(
        planned_tools and expected_steps and planned_tools[0] == expected_steps[0]["tool"]
    )
    result.tool_sequence_match = planned_tools == [step["tool"] for step in expected_steps]
    result.args_match = _args_match(planned, expected_steps)
    result.chart_produced = _chart_ready(outcome.get("chart_file"))

    actual_frame = outcome.get("result_df")

    # 图表类任务：create_chart 不写 result_df，若仍按"必须有结果表"判定，
    # 失败原因会被误报为 no_result，掩盖"没画图"或"前置步骤缺失"。
    if result.expects_chart and not result.chart_produced:
        result.failure = "no_chart" if planned_tools else "no_tool_call"
        return result

    if actual_frame is None and result.expects_chart:
        # create_chart 不写 result_df；图表类任务用当前数据状态当结果表，
        # 这样"只画图不改数据"的任务也能按数据是否正确判定。
        actual_frame = outcome.get("current_df")

    if actual_frame is None:
        result.failure = "no_result"
        return result

    ordered_match, reason = compare_frames(expected_frame, actual_frame)
    result.task_success = ordered_match
    result.task_success_unordered = ordered_match or reason == "行内容一致但顺序不同"
    if not ordered_match:
        result.failure = (
            "chart_ok_but_data_mismatch" if result.expects_chart else reason
        )
    return result


def evaluate(
    *,
    data_dir: str | Path,
    llm: Any,
    config: EvalConfig | None = None,
) -> EvalReport:
    resolved = config or EvalConfig()
    root = Path(data_dir)

    split_path = root / "splits" / f"{resolved.split}.jsonl"
    if not split_path.is_file():
        raise FileNotFoundError(
            f"找不到分区文件：{split_path}。请先运行 build 生成数据集。"
        )
    samples = read_jsonl(split_path)
    if resolved.limit is not None:
        samples = samples[: max(0, resolved.limit)]
    if not samples:
        raise ValueError(f"分区 {resolved.split} 中没有样本。")

    tables = {table.table_id: table for table in load_tables(root / "tables")}

    report = EvalReport(
        config={
            "split": resolved.split,
            "limit": resolved.limit,
            "model": resolved.model,
            "base_url": resolved.base_url,
            "temperature": resolved.temperature,
        }
    )

    previous_llm = agent.llm
    agent.llm = llm

    # 评测会产生图表文件；重定向到临时目录，避免污染用户的 outputs 目录。
    import excel_ops

    previous_output_dir = excel_ops.OUTPUT_DIR
    chart_dir = root / "eval_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    excel_ops.OUTPUT_DIR = str(chart_dir)
    try:
        for sample in samples:
            report.samples.append(
                evaluate_sample(sample, tables, dataset_root=root)
            )
    finally:
        excel_ops.OUTPUT_DIR = previous_output_dir
        agent.llm = previous_llm

    return report


def make_chat_model(config: EvalConfig) -> Any:
    """构造评测用模型：显式 base_url 时走本地 OpenAI 兼容服务，否则用项目配置。"""
    if config.base_url:
        from langchain.chat_models import init_chat_model

        return init_chat_model(
            model=config.model or "qwen2.5-1.5b-instruct",
            model_provider="openai",
            temperature=config.temperature,
            api_key=config.api_key or "none",
            base_url=config.base_url,
        )

    from load_model import init_llm

    return init_llm()


def write_report(report: EvalReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def format_summary(report: EvalReport) -> str:
    metrics = report.metrics
    lines = [
        f"样本数            : {metrics.get('samples', 0)}",
        f"任务成功率        : {metrics.get('task_success_rate', 0):.2%}",
        f"任务成功(无序)    : {metrics.get('task_success_unordered_rate', 0):.2%}",
        f"工具序列一致      : {metrics.get('tool_sequence_match_rate', 0):.2%}",
        f"首个工具正确      : {metrics.get('first_tool_match_rate', 0):.2%}",
        f"参数完全一致      : {metrics.get('args_match_rate', 0):.2%}",
        f"触顶轮数上限      : {metrics.get('hit_turn_limit_rate', 0):.2%}",
        f"平均工具调用次数  : {metrics.get('avg_tool_steps', 0)}",
        f"平均模型轮数      : {metrics.get('avg_turns', 0)}",
    ]
    if metrics.get("chart_samples"):
        lines.append(
            f"图表产出率        : {metrics.get('chart_produced_rate', 0):.2%}"
            f"（{metrics['chart_samples']} 条图表任务）"
        )
    by_category = metrics.get("by_category") or {}
    if by_category:
        lines.append("")
        lines.append("按类别：")
        for name, bucket in by_category.items():
            lines.append(
                f"  {name:<18} {bucket['success']}/{bucket['samples']}"
                f"  ({bucket['success_rate']:.0%})"
            )
    return "\n".join(lines)


def failure_breakdown(report: EvalReport) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for sample in report.samples:
        if sample.task_success:
            continue
        counts[sample.failure] = counts.get(sample.failure, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))
