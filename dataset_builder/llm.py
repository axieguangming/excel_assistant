"""大模型适配层：生成候选操作链、自然语言问题，并做反向解析校验。

所有输出都要求是 JSON，并经过 `extract_json` 解析。模型输出只作为候选，
是否可用仍由 `chain.normalize_chain` 和执行阶段决定。
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from langchain_core.messages import HumanMessage, SystemMessage

from dataset_builder.chain import TOOL_NAMES, TOOL_SCHEMAS
from dataset_builder.tables import SyntheticTable

PROMPT_VERSION = "dataset-v1"


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return str(content)


def _extract_balanced(text: str, start: int) -> str | None:
    opener = text[start]
    closer = "]" if opener == "[" else "}"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(text: str) -> Any:
    """从模型回复中提取第一个可解析的 JSON 数组或对象。"""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("模型回复为空，无法解析 JSON。")

    for start, char in enumerate(text):
        if char not in "[{":
            continue
        chunk = _extract_balanced(text, start)
        if chunk is None:
            continue
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue
    raise ValueError("模型回复中找不到合法 JSON。")


def tool_catalog(allowed_tools: Iterable[str] = TOOL_NAMES) -> str:
    """把工具契约转成提示词可读的文本。"""
    entries: list[dict[str, Any]] = []
    for name in allowed_tools:
        schema = TOOL_SCHEMAS.get(name)
        entries.append(
            {
                "tool": name,
                "args_schema": schema.model_json_schema() if schema else {},
            }
        )
    return json.dumps(entries, ensure_ascii=False, indent=2)


class DatasetLLM:
    def __init__(self, chat_model: Any = None):
        self._chat_model = chat_model

    @property
    def model(self) -> Any:
        if self._chat_model is None:
            from load_model import init_llm

            self._chat_model = init_llm()
        return self._chat_model

    def _invoke_json(self, system: str, user: str) -> Any:
        response = self.model.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return extract_json(_content_text(getattr(response, "content", "")))


class LLMChainProposer:
    """让强模型根据工作表概况提出候选操作链。"""

    system = (
        "你是 Excel 数据分析任务设计器。给定工作表结构和可用工具，"
        "生成贴近真实用户需求的候选操作链。只能使用给定工具与真实字段名，"
        "参数必须符合对应工具的 JSON Schema。不要输出解释文字。"
        "输出格式：JSON 数组，每个元素是 {\"steps\": [{\"tool\": ..., \"args\": {...}}]}。"
    )

    def __init__(self, llm: DatasetLLM | None = None):
        self.llm = llm or DatasetLLM()

    def propose(
        self,
        table: SyntheticTable,
        count: int,
        *,
        max_steps: int = 4,
        allowed_tools: Iterable[str] | None = None,
    ) -> list[Any]:
        allowed = tuple(allowed_tools) if allowed_tools is not None else TOOL_NAMES
        user = json.dumps(
            {
                "worksheet": table.profile(),
                "available_tools": json.loads(tool_catalog(allowed)),
                "requirements": {
                    "count": count,
                    "max_steps": max_steps,
                    "min_steps": 1,
                    "unique": True,
                    "notes": [
                        "每一步的输入是上一步的输出；汇总后只能引用汇总结果中的字段。",
                        "多步任务要真实可执行，不要引用不存在的字段。",
                    ],
                },
            },
            ensure_ascii=False,
        )
        payload = self.llm._invoke_json(self.system, user)
        if isinstance(payload, dict):
            payload = payload.get("chains", payload.get("steps"))
        if not isinstance(payload, list):
            raise ValueError("操作链生成结果必须是 JSON 数组。")
        return payload


class LLMQuestionWriter:
    """由已验证的操作链生成自然语言问题与预期答复要点。"""

    system = (
        "你是 Excel 用户诉求转写员。根据工作表概况和操作链，写一条自然、完整的"
        "中文用户提问，并给出一句预期答复要点。提问必须忠实表达操作链，"
        "不得增加链中没有的条件，也不要出现工具名、参数名或 JSON。"
        "只输出 {\"question\": \"...\", \"answer\": \"...\"}。"
    )

    def __init__(self, llm: DatasetLLM | None = None):
        self.llm = llm or DatasetLLM()

    def write(
        self,
        table: SyntheticTable,
        steps: list[dict[str, Any]],
        *,
        rows: int | None = None,
        columns: list[str] | None = None,
    ) -> dict[str, str]:
        user = json.dumps(
            {
                "worksheet": table.profile(),
                "steps": steps,
                "expected_rows": rows,
                "expected_columns": columns,
            },
            ensure_ascii=False,
        )
        payload = self.llm._invoke_json(self.system, user)
        if not isinstance(payload, dict):
            raise ValueError("问题生成结果必须是 JSON 对象。")
        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError("问题或预期答复为空。")
        return {"question": question, "answer": answer}


class LLMQuestionParser:
    """把自然语言问题反解析为操作链，用于一致性校验。"""

    system = (
        "你是 Excel 操作解析器。把用户问题解析为等价的工具调用链，"
        "只能使用给定工具与真实字段名，参数必须符合对应工具的 JSON Schema。"
        "如果无法解析，输出 {\"steps\": []}。只输出 JSON。"
    )

    def __init__(self, llm: DatasetLLM | None = None):
        self.llm = llm or DatasetLLM()

    def parse(
        self,
        table: SyntheticTable,
        question: str,
        *,
        allowed_tools: Iterable[str] | None = None,
    ) -> Any:
        allowed = tuple(allowed_tools) if allowed_tools is not None else TOOL_NAMES
        user = json.dumps(
            {
                "worksheet": table.profile(),
                "available_tools": json.loads(tool_catalog(allowed)),
                "question": question,
            },
            ensure_ascii=False,
        )
        return self.llm._invoke_json(self.system, user)
