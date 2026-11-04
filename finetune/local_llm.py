"""把本地 HF 模型（基座 + LoRA 适配器）包装成 LangChain 聊天模型。

这样 `dataset_builder.eval` 就能驱动**应用真实的 Agent 图**（多步工具调用、
真实工具执行、轮数上限、强制总结），而不是简化的单次生成对比。

工具定义与消息序列化刻意复刻 `finetune.dataset`，保证训练/推理格式一致。
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def _tool_definition(tool: Any) -> dict[str, Any]:
    """与 `finetune.dataset.tool_definitions()` 保持完全一致的 schema。"""
    if getattr(tool, "args_schema", None) is not None:
        schema = tool.args_schema.model_json_schema()
        schema.pop("title", None)
    else:
        schema = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
        },
    }


def _to_chat_messages(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """把 LangChain 消息转成 Qwen 模板接受的字典序列。"""
    converted: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            converted.append({"role": "system", "content": str(message.content)})
        elif isinstance(message, HumanMessage):
            converted.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, ToolMessage):
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "name": message.name or "",
                    "content": str(message.content),
                }
            )
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                converted.append(
                    {
                        "role": "assistant",
                        "content": str(message.content or ""),
                        "tool_calls": [
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": json.dumps(
                                        call["args"], ensure_ascii=False
                                    ),
                                },
                            }
                            for call in message.tool_calls
                        ],
                    }
                )
            else:
                converted.append(
                    {"role": "assistant", "content": str(message.content or "")}
                )
        else:
            converted.append({"role": "user", "content": str(message.content)})
    return converted


def _extract_tool_call(text: str) -> tuple[dict[str, Any] | None, str]:
    """从生成文本中解析第一个工具调用，返回 (调用, 之前的正文)。

    用标签切分而不是正则匹配花括号，避免参数里出现嵌套 JSON 时截断。
    """
    start = text.find(_TOOL_CALL_OPEN)
    if start < 0:
        return None, text.strip()
    end = text.find(_TOOL_CALL_CLOSE, start)
    if end < 0:
        payload = text[start + len(_TOOL_CALL_OPEN):]
    else:
        payload = text[start + len(_TOOL_CALL_OPEN):end]

    payload = payload.strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None, text.strip()
    if not isinstance(parsed, dict) or "name" not in parsed:
        return None, text.strip()
    return parsed, text[:start].strip()


class LocalChatModel:
    """最小可用的 LangChain 聊天模型：`bind_tools` + `invoke`。

    只实现 Agent 用到的这两个方法，避免引入额外的 provider 依赖。
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 256,
        model_name: str = "local-lora",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.tools: list[dict[str, Any]] = []

    # ---- LangChain 兼容接口 -------------------------------------------
    def bind_tools(self, tools: Sequence[Any], **_: Any) -> "LocalChatModel":
        # 直接在此实例上保存：force_finalize 节点会复用同一个 llm 对象，
        # 未绑定的实例会让最终总结缺少工具上下文。
        self.tools = [_tool_definition(tool) for tool in tools]
        return self

    def invoke(self, messages: Sequence[BaseMessage], **_: Any) -> AIMessage:
        import torch

        payload = _to_chat_messages(messages)
        prompt = self.tokenizer.apply_chat_template(
            payload,
            tools=self.tools or None,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        call, preface = _extract_tool_call(text)
        if call is None:
            return AIMessage(content=text.strip())

        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if not isinstance(arguments, dict):
            # 参数无法解析时不要伪造调用，让 Agent 把原始文本当作最终答复，
            # 评测侧会记为参数错误而不是误报成功。
            return AIMessage(content=text.strip())

        return AIMessage(
            content=preface,
            tool_calls=[
                {
                    "name": str(call["name"]),
                    "args": arguments,
                    "id": f"call_{abs(hash(text)) % 10**8}",
                    "type": "tool_call",
                }
            ],
        )
