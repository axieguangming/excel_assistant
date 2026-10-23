"""数据集流水线：候选操作链 → 结构校验 → 执行真值 → 问题生成 → 一致性校验 → 样本落盘。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import pandas as pd

from dataset_builder.chain import (
    ChainValidationError,
    TOOL_NAMES,
    canonical_chain,
    normalize_chain,
)
from dataset_builder.executor import execute_chain
from dataset_builder.split import grouped_split
from dataset_builder.tables import SOURCE, SyntheticTable
from dataset_builder.templates import categories_for


class ChainProposer(Protocol):
    def propose(
        self,
        table: SyntheticTable,
        count: int,
        *,
        max_steps: int = 4,
        allowed_tools: Iterable[str] | None = None,
    ) -> list[Any]: ...


class QuestionWriter(Protocol):
    def write(
        self,
        table: SyntheticTable,
        steps: list[dict[str, Any]],
        *,
        rows: int | None = None,
        columns: list[str] | None = None,
    ) -> dict[str, str]: ...


QuestionParser = Callable[..., Any]


@dataclass
class BuildConfig:
    samples_per_table: int = 6
    max_steps: int = 4
    max_chain_attempts: int = 12
    render_charts: bool = False
    verify_questions: bool = True
    parse_attempts: int = 2
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 0


@dataclass
class BuildResult:
    samples: list[dict[str, Any]] = field(default_factory=list)
    rejects: list[dict[str, Any]] = field(default_factory=list)
    splits: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _difficulty(steps: list[dict[str, Any]]) -> str:
    if len(steps) == 1:
        return "simple"
    if len(steps) <= 3:
        return "multi_step"
    return "complex"


def _reject(
    table: SyntheticTable,
    reason: str,
    detail: str,
    steps: Any = None,
) -> dict[str, Any]:
    return {
        "table_id": table.table_id,
        "reason": reason,
        "detail": detail,
        "steps": steps,
        "rejected_at": _now(),
    }


def _write_result_artifact(
    frame: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "artifact": f"results/{path.name}",
        "row_count": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "sha256": digest,
    }


def _diverse_order(
    chains: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """先按类别签名取一条，再补其余候选，避免单张表被同一类任务占满。"""
    selected: list[list[dict[str, Any]]] = []
    remaining: list[list[dict[str, Any]]] = []
    signatures: set[tuple[str, ...]] = set()

    for chain in chains:
        signature = tuple(categories_for(chain))
        if signature in signatures:
            remaining.append(chain)
        else:
            signatures.add(signature)
            selected.append(chain)

    return [*selected, *remaining]


def _parse_with_repair(
    parser: QuestionParser,
    table: SyntheticTable,
    question: str,
    *,
    max_steps: int,
    attempts: int,
) -> tuple[list[dict[str, Any]] | None, str]:
    """解析问题并把校验错误反馈给模型，最多尝试 attempts 次。"""
    feedback = ""
    last_error = ""
    for _ in range(max(1, attempts)):
        try:
            payload = parser(table, question, feedback)
        except TypeError:
            # 兼容只接受两个参数的解析器；此时无法反馈错误。
            if feedback:
                break
            payload = parser(table, question)
        except (ValueError, RuntimeError) as error:
            last_error = str(error)
            feedback = last_error
            continue

        try:
            return normalize_chain(payload, max_steps=max_steps), ""
        except ChainValidationError as error:
            last_error = str(error)
            feedback = last_error

    return None, last_error or "解析失败"


def _result_signature(frame: pd.DataFrame) -> tuple[str, ...]:
    """把结果表转换成可比较的稳定字符串，忽略列名重命名等表现差异。"""
    normalized = frame.reset_index(drop=True).astype(str)
    return tuple(normalized.itertuples(index=False, name=None))


def _results_mismatch(expected: pd.DataFrame, actual: pd.DataFrame) -> str:
    """比较两张结果表；返回空字符串表示一致。"""
    if expected.shape != actual.shape:
        return (
            f"结果形状不同：期望 {expected.shape}，解析后为 {actual.shape}"
        )
    expected_signature = _result_signature(expected)
    actual_signature = _result_signature(actual)
    if expected_signature != actual_signature:
        return "结果内容不同：问题解析出的操作产生了不同的数据"
    return ""


def build_dataset(
    tables: Iterable[SyntheticTable],
    *,
    proposer: ChainProposer,
    writer: QuestionWriter,
    parser: QuestionParser | None = None,
    config: BuildConfig | None = None,
    out_dir: str | Path,
    allowed_tools: Iterable[str] | None = None,
) -> BuildResult:
    resolved = config or BuildConfig()
    root = Path(out_dir)
    results_dir = root / "results"
    charts_dir = root / "charts"
    result = BuildResult()
    table_list = list(tables)

    for table in table_list:
        collected = 0
        seen: set[str] = set()
        # 至少请求与工具数相当的候选，才能覆盖不同能力类别。
        requested = max(resolved.samples_per_table + 2, len(TOOL_NAMES))
        candidates = proposer.propose(
            table,
            requested,
            max_steps=resolved.max_steps,
            allowed_tools=allowed_tools,
        )

        normalized: list[list[dict[str, Any]]] = []
        for candidate in list(candidates)[: resolved.max_chain_attempts]:
            try:
                steps = normalize_chain(
                    candidate,
                    max_steps=resolved.max_steps,
                    allowed_tools=allowed_tools,
                )
            except ChainValidationError as error:
                result.rejects.append(_reject(table, "invalid_chain", str(error), candidate))
                continue

            fingerprint = canonical_chain(steps)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            normalized.append(steps)

        # 优先让每张表覆盖不同的任务类别，再补充同类候选。
        for steps in _diverse_order(normalized):
            if collected >= resolved.samples_per_table:
                break

            outcome = execute_chain(
                table.frame,
                steps,
                render_charts=resolved.render_charts,
                chart_dir=charts_dir,
            )
            if not outcome.ok or outcome.frame is None:
                result.rejects.append(
                    _reject(table, "execution_failed", outcome.error or "", steps)
                )
                continue

            try:
                written = writer.write(
                    table,
                    steps,
                    rows=int(len(outcome.frame)),
                    columns=[str(column) for column in outcome.frame.columns],
                )
            except (ValueError, RuntimeError, KeyError) as error:
                result.rejects.append(
                    _reject(table, "question_failed", str(error), steps)
                )
                continue

            question = str(written.get("question", "")).strip()
            answer = str(written.get("answer", "")).strip()
            if not question or not answer:
                result.rejects.append(
                    _reject(table, "question_failed", "问题或预期答复为空", steps)
                )
                continue

            consistency = "unverified"
            if parser is not None and resolved.verify_questions:
                parsed_steps, parse_error = _parse_with_repair(
                    parser,
                    table,
                    question,
                    max_steps=resolved.max_steps,
                    attempts=resolved.parse_attempts,
                )
                if parsed_steps is None:
                    result.rejects.append(
                        _reject(
                            table,
                            "question_mismatch",
                            f"反向解析失败：{parse_error}",
                            steps,
                        )
                    )
                    continue

                # 主判据是执行结果是否一致：重命名等表现差异不影响判定，
                # 但解析出不同的操作会产生不同结果，从而被拒绝。
                replayed = execute_chain(table.frame, parsed_steps)
                if not replayed.ok or replayed.frame is None:
                    result.rejects.append(
                        _reject(
                            table,
                            "question_mismatch",
                            f"反向解析结果无法执行：{replayed.error}",
                            steps,
                        )
                    )
                    continue
                difference = _results_mismatch(outcome.frame, replayed.frame)
                if difference:
                    result.rejects.append(
                        _reject(
                            table,
                            "question_mismatch",
                            difference,
                            steps,
                        )
                    )
                    continue
                consistency = "llm_verified"
            elif resolved.verify_questions:
                consistency = "structural"

            sample_id = f"{table.table_id}-{collected + 1:03d}"
            artifact = _write_result_artifact(
                outcome.frame, results_dir / f"{sample_id}.csv"
            )
            result.samples.append(
                {
                    "sample_id": sample_id,
                    "source": dict(SOURCE),
                    "workbook_id": table.table_id,
                    "sheet": table.sheet,
                    "user_request": question,
                    "task_categories": categories_for(steps),
                    "difficulty": _difficulty(steps),
                    "expected_actions": steps,
                    "step_results": outcome.step_dicts(),
                    "expected_result": artifact,
                    "expected_response": answer,
                    "verification": {
                        "state": (
                            "auto_passed"
                            if consistency in {"llm_verified", "structural"}
                            else "needs_review"
                        ),
                        "execution": "passed",
                        "question_consistency": consistency,
                        "checked_at": _now(),
                    },
                    "provenance": {
                        "proposer": proposer.__class__.__name__,
                        "writer": writer.__class__.__name__,
                        "prompt_version": "dataset-v1",
                        "created_at": _now(),
                    },
                }
            )
            collected += 1

    result.splits = grouped_split(
        result.samples,
        ratios=resolved.split_ratios,
        seed=resolved.split_seed,
    )
    result.report = _build_report(result, tables=table_list)
    return result


def _build_report(result: BuildResult, *, tables: list[SyntheticTable]) -> dict[str, Any]:
    categories: dict[str, int] = {}
    difficulties: dict[str, int] = {}
    for sample in result.samples:
        for category in sample["task_categories"]:
            categories[category] = categories.get(category, 0) + 1
        difficulties[sample["difficulty"]] = difficulties.get(sample["difficulty"], 0) + 1

    reasons: dict[str, int] = {}
    for reject in result.rejects:
        reasons[reject["reason"]] = reasons.get(reject["reason"], 0) + 1

    return {
        "tables": len(tables),
        "samples": len(result.samples),
        "rejects": len(result.rejects),
        "reject_reasons": reasons,
        "categories": dict(sorted(categories.items())),
        "difficulties": dict(sorted(difficulties.items())),
        "splits": {name: len(items) for name, items in result.splits.items()},
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_build_result(result: BuildResult, out_dir: str | Path) -> dict[str, Path]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "samples": root / "samples.jsonl",
        "rejects": root / "rejects.jsonl",
        "report": root / "report.json",
    }
    _write_jsonl(paths["samples"], result.samples)
    _write_jsonl(paths["rejects"], result.rejects)
    paths["report"].write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for split_name, samples in result.splits.items():
        split_path = root / "splits" / f"{split_name}.jsonl"
        _write_jsonl(split_path, samples)
        paths[split_name] = split_path

    return paths


def replay_samples(
    samples: Iterable[dict[str, Any]],
    tables_by_id: dict[str, SyntheticTable],
    *,
    max_steps: int = 10,
) -> dict[str, Any]:
    """重新执行期望操作链，核对记录的行数与列名是否仍然一致。"""
    checked = 0
    mismatches: list[dict[str, Any]] = []

    for sample in samples:
        table = tables_by_id.get(str(sample.get("workbook_id")))
        if table is None:
            mismatches.append(
                {"sample_id": sample.get("sample_id"), "reason": "missing_table"}
            )
            continue

        try:
            steps = normalize_chain(sample.get("expected_actions"), max_steps=max_steps)
        except (ChainValidationError, ValueError) as error:
            mismatches.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "reason": "invalid_chain",
                    "detail": str(error),
                }
            )
            continue

        outcome = execute_chain(table.frame, steps)
        if not outcome.ok or outcome.frame is None:
            mismatches.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "reason": "execution_failed",
                    "detail": outcome.error,
                }
            )
            continue

        recorded = sample.get("step_results", [])
        actual = outcome.step_dicts()
        if len(recorded) != len(actual):
            mismatches.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "reason": "step_count_mismatch",
                    "detail": f"{len(recorded)} != {len(actual)}",
                }
            )
            continue

        difference = next(
            (
                (index, expected, found)
                for index, (expected, found) in enumerate(zip(recorded, actual), start=1)
                if expected.get("rows") != found.get("rows")
                or expected.get("columns") != found.get("columns")
            ),
            None,
        )
        if difference is not None:
            index, expected, found = difference
            mismatches.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "reason": "step_result_mismatch",
                    "detail": (
                        f"第 {index} 步记录为 {expected.get('rows')} 行/"
                        f"{len(expected.get('columns') or [])} 列，实际为 "
                        f"{found.get('rows')} 行/{len(found.get('columns') or [])} 列"
                    ),
                }
            )
            continue

        expected_result = sample.get("expected_result") or {}
        if expected_result.get("row_count") not in (None, len(outcome.frame)):
            mismatches.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "reason": "row_count_mismatch",
                    "detail": (
                        f"记录 {expected_result.get('row_count')}，"
                        f"实际 {len(outcome.frame)}"
                    ),
                }
            )
            continue

        checked += 1

    return {"checked": checked, "mismatches": mismatches}
