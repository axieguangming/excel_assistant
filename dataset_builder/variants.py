"""为已验证的操作链生成口语化问法与多样化答复。

为什么需要单独实现而不是复用 `LLMQuestionWriter`：

- `LLMQuestionWriter` 每条链只产出一条「完整、规范」的问题，句式仍贴近模板；
  真实用户会说「各城市满意度加起来画个饼图」，不会说「按 城市 分组统计
  满意度_sum 的合计，然后绘制饼图」。训练集只有前者，模型就学不到后者。
- 预期答复在原流程里是 `已按要求完成操作：{summary}。` 的固定模板，
  微调后模型会在分布外输入上出现重复输出退化（实测 3/32 条）。

本模块因此同时改写问法与答复，并做三类质量校验：

1. **不得泄漏内部列名**：`满意度_sum`、`推荐意愿_add_100` 这类由工具生成的
   列名属于实现细节，真实用户不会知道。出现即判定不合格。
2. **不得出现工具名/参数名**：`aggregate_data`、`group_by`、`chart_y` 等。
3. **不得增加链中没有的条件**：由 `LLMQuestionParser` 的反向解析在批量阶段
   二次校验；本模块只做廉价的文本级过滤，避免明显污染进入训练集。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import re
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from dataset_builder.llm import DatasetLLM, _content_text
from dataset_builder.tables import SyntheticTable

# 工具名与参数名，出现在提问里说明模型没有「转写」而是在复述操作链。
FORBIDDEN_WORDS = (
    "aggregate_data",
    "filter_rows",
    "sort_rows",
    "top_n_rows",
    "deduplicate_rows",
    "fill_missing_values",
    "add_computed_column",
    "describe_data",
    "create_chart",
    "group_by",
    "chart_y",
    "chart_x",
    "chart_type",
    "sort_by",
    "ascending",
    "fill_strategy",
    "round_digits",
    "metrics",
    "subset",
)

# 计数口径歧义：「X数」既可理解为行数，也可理解为去重个数。
# 出现这类说法时执行者只能反问，无法直接执行。
AMBIGUOUS_COUNT_TERM = re.compile(
    r"(运单数|订单数|客户数|员工数|受访者数|商品数|记录数|条数|个数)"
)


def looks_internal_name(name: str) -> bool:
    """判断列名是否是"工具内部命名"而不是用户会说的业务用语。

    区分标准是下划线：代码自动拼出的列名一定是 ``<列>_<函数>`` 或
    ``<列>_x_<列>``（如 ``销售额_sum``、``预算金额_x_差异``），
    而 LLM 生成的业务列名是纯中文短语（如 ``销售额合计``、``平均满意度``）。

    这个区分很重要：用户完全可以说「按产品线看销售额合计」，
    这不算泄漏；但「销售额_sum」这种名字一旦出现在提问里，
    模型就是在被喂入一个它必须逐字复现的内部标识符。
    """
    return "_" in str(name)


def generated_column_names(steps: Sequence[dict[str, Any]]) -> list[str]:
    """收集由工具生成、真实用户不可能预先知道的列名。"""
    names: list[str] = []
    for step in steps:
        args = step.get("args") or {}
        if step.get("tool") == "add_computed_column" and args.get("name"):
            names.append(str(args["name"]))
            continue
        # 聚合产生的命名列（未显式命名时由代码拼成 "<列>_<函数>"）。
        for metric in args.get("metrics") or []:
            if isinstance(metric, dict) and metric.get("name"):
                names.append(str(metric["name"]))
    return names


def internal_column_names(steps: Sequence[dict[str, Any]]) -> list[str]:
    """只返回可能泄漏内部命名约定、需要在提问中回避的列名。"""
    return [name for name in generated_column_names(steps) if looks_internal_name(name)]


def _counting_columns(
    steps: Sequence[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """返回 (count 指标涉及列, nunique 指标涉及列)。

    这两类指标的措辞最容易产生口径歧义，需要单独校验收紧。
    """
    counting: list[str] = []
    distinct: list[str] = []
    for step in steps:
        if step.get("tool") != "aggregate_data":
            continue
        for metric in (step.get("args") or {}).get("metrics") or []:
            if not isinstance(metric, dict):
                continue
            name = str(metric.get("name") or metric.get("column") or "")
            function = str(metric.get("function"))
            if function == "count":
                counting.append(name)
            elif function == "nunique":
                distinct.append(name)
    return counting, distinct


@dataclass
class Variant:
    question: str
    answer: str
    style: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "question": self.question,
            "answer": self.answer,
            "style": self.style,
            "note": self.note,
        }


@dataclass
class VariantReport:
    accepted: list[Variant] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)

    @property
    def rejection_rate(self) -> float:
        total = len(self.accepted) + len(self.rejected)
        return round(len(self.rejected) / total, 4) if total else 0.0


class ColloquialVariantWriter:
    """把操作链改写成多条真实用户口吻的提问与答复。"""

    system = (
        "你是中文 Excel 用户的代言人。给定工作表结构、真实执行的操作链、"
        "执行后的行数与列名，请写多条**真实用户会说的话**。\n"
        "\n"
        "严格要求：\n"
        "1. 提问必须忠实表达给定的操作链：动作、分组字段、统计方式、排序方向、"
        "筛选条件、条数、图表类型都不能变。**任何一条条件都不能省略**，"
        "包括「不为空」「不等于某值」这类过滤。\n"
        "2. **汇总必须写清口径**：sum 要写成「合计/总和/加起来/加总」，"
        "mean 要写成「平均/平均值/均值」，"
        "min/max 要写成「最小/最大」，"
        "**绝不能只说「汇总」「统计一下」**——那会同时指向 sum 和 mean，产生歧义。\n"
        "   计数类必须写到不可能误解为止：\n"
        "   - count（组内行数）→ 只能写「多少条记录」「记录条数」「一共几行」；"
        "**禁止写成「订单数」「运单数」「客户数」「商品数」这类「X数」说法**，"
        "因为它们也可理解为去重个数。\n"
        "   - nunique → 只能写「多少个不同的 X」「X 有几种」；"
        "同样禁止「X数」说法。\n"
        "3. **不要出现带下划线的内部列名**（例如 xxx_sum、xxx_add_100、"
        "xxx_x_yyy）——那是程序拼出来的标识符，真实用户不知道。"
        "汇总结果用自然口语表达即可（如「销售额合计」「平均满意度」）。"
        "对于计算列，不要以「新增列 某某」的方式描述，而要说「算一下 A 减 B」"
        "「把两列相乘」这类自然说法。\n"
        "4. 不要出现任何工具名、英文参数名或 JSON。\n"
        "5. 措辞要有真实感：可以说「帮我」「我看看」「大概」，可以省略表名、"
        "省略主语、用业务口语（如「花超了」「最多的几个」「占多少」）。\n"
        "   **「省略」只针对语气和主语，不针对任何操作条件。**\n"
        "6. 每条变体风格不同，不要只是同义词替换：\n"
        "   - plain: 简洁直白的一句话\n"
        "   - minimal: 极简口语，去掉一切多余的字，但条件必须齐全\n"
        "   - business: 业务口吻，带业务目的（如「要跟领导汇报各城市满意度占比」）\n"
        "   - verbose: 啰嗦一点，带口语填充词\n"
        "   - no_sheet: 不提工作表名的说法\n"
        "7. answer 要像助手如实汇报，**不要用固定开头**。可以变化说法："
        "「已经处理好了」「结果如下」「按你的要求做完了」等；必须提到行数与列数。"
        "answer 里同样不能出现工具名。\n"
        "\n"
        "只输出 JSON 对象：{\"variants\": [{\"question\": \"...\", "
        "\"answer\": \"...\", \"style\": \"plain\"}]}"
    )

    def __init__(self, llm: DatasetLLM | None = None):
        self.llm = llm or DatasetLLM()

    def write(
        self,
        table: SyntheticTable,
        steps: list[dict[str, Any]],
        *,
        count: int = 5,
        rows: int | None = None,
        columns: list[str] | None = None,
        existing_questions: Sequence[str] = (),
    ) -> VariantReport:
        leaked = internal_column_names(steps)
        counting, distinct = _counting_columns(steps)
        user_payload = {
            "worksheet": {
                "sheet": table.sheet,
                "columns": [str(name) for name in table.frame.columns],
                "rows": int(len(table.frame)),
            },
            "steps": steps,
            "expected_rows": rows,
            "expected_columns": columns,
            "do_not_mention_columns": leaked,
            # 告知模型哪些指标是计数/去重计数，必须用无歧义措辞
            "count_metrics": counting,
            "nunique_metrics": distinct,
            "wanted_variants": count,
        }
        response = self.llm.model.invoke(
            [
                SystemMessage(content=self.system),
                HumanMessage(content=json.dumps(user_payload, ensure_ascii=False)),
            ]
        )
        payload = _extract_variants(_content_text(getattr(response, "content", "")))

        report = VariantReport()
        seen = {question.strip() for question in existing_questions}
        for item in payload:
            if not isinstance(item, dict):
                report.rejected.append({"reason": "非对象", "raw": json.dumps(item, ensure_ascii=False)})
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            reason = _check(
                question,
                answer,
                leaked,
                count_columns=counting,
                nunique_columns=distinct,
            )
            if reason:
                report.rejected.append({"reason": reason, "question": question})
                continue
            if question in seen:
                report.rejected.append({"reason": "与已有问题重复", "question": question})
                continue
            seen.add(question)
            report.accepted.append(
                Variant(
                    question=question,
                    answer=answer,
                    style=str(item.get("style", "")).strip(),
                )
            )
        return report


def _extract_variants(text: str) -> list[Any]:
    """从回复中取出 variants 数组；解析失败时退回通用 JSON 提取。"""
    from dataset_builder.llm import extract_json

    start = text.find("{")
    if start >= 0:
        try:
            payload = extract_json(text[start:])
            if isinstance(payload, dict) and isinstance(payload.get("variants"), list):
                return payload["variants"]
            if isinstance(payload, list):
                return payload
        except ValueError:
            pass
    raise ValueError("模型回复中找不到 variants 数组。")


def _check(
    question: str,
    answer: str,
    leaked: Sequence[str],
    *,
    count_columns: Sequence[str] = (),
    nunique_columns: Sequence[str] = (),
) -> str:
    """返回拒绝原因；空字符串表示通过。

    ``count_columns`` / ``nunique_columns`` 用于拦住「订单数」这类计数口径歧义：
    模型说「X数」时，执行者无法判断要行数还是去重个数，只能反问，
    这与评测期望的「直接执行」冲突。
    """
    if not question or not answer:
        return "问题或答复为空"
    if len(question) < 4:
        return "问题过短"
    if len(question) > 200:
        return "问题过长"

    combined = question + "\n" + answer
    lowered = combined.lower()
    for word in FORBIDDEN_WORDS:
        if word.lower() in lowered:
            return f"含工具名/参数名：{word}"
    for name in leaked:
        if name and name in combined:
            return f"泄漏生成列名：{name}"

    # 模板痕迹：出现「分组统计」「（降序）」「并保留」之类就是复述操作链。
    # 注意不要用「横轴/纵轴」判定：真实用户指定图表轴时也会这么说，
    # 之前用它做过滤误杀了 3 条合格变体。
    for marker in ("分组统计", "（降序）", "（升序）", "并保留 "):
        if marker in question:
            return f"仍是模板句式：{marker}"

    # 计数口径歧义：「<标识名词>数」既可指行数也可指去重个数。
    if count_columns or nunique_columns:
        ambiguous = AMBIGUOUS_COUNT_TERM.search(question)
        if ambiguous:
            return f"计数口径歧义：{ambiguous.group(0)}"

    if not re.search(r"\d", answer):
        return "答复缺少行数/列数"
    return ""
