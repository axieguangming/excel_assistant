"""让推理使用与训练一致的 tools schema 与历史渲染方式。

问题一：tools schema 不一致
---------------------------
训练数据里的 schema 由 `tool_schema.tool_definitions()` 生成；推理时如果走
LangChain 的 `bind_tools()`，它会先用 `convert_to_openai_tool()` 重新转换一遍，
结果**完全不一样**：

    tool_definitions() 原始输出   5,704 字符
    LangChain 转换后               4,558 字符
    9 个工具里 0 个的 JSON 相同

其中 `describe_data` 差得最多（380 vs 158 字符）：它没有显式 `args_schema`，
`model_json_schema()` 会把框架注入的 `InjectedState` / `InjectedToolCallId`
当成 `state`、`tool_call_id` 两个**必填**参数暴露出来。

直觉上应该把它清理掉，但**训练时模型见到的就是这个形态**，所以推理侧要做的
不是"修正"，而是**忠实复现**。

问题二：历史工具调用的编码不一致（这才是 GGUF 掉分的主因）
----------------------------------------------------------
Qwen 模板用 `{{ tool_call.arguments | tojson }}` 渲染历史里的工具调用：

    训练数据（`finetune/dataset.py`）里 `arguments` 是**字符串**
      → tojson 再编码一次 → 双重编码
        "arguments": "{\"group_by\": \"部门\"}"
    llama-server 先把字符串解析回对象再渲染
      → 单重编码
        "arguments": {"group_by": "部门"}

于是第二轮起 GGUF 收到的 prompt 与训练不同，退化成**重复上一步工具调用**
（`aggregate_data` ×9 直到轮数上限）。实测同权重、同 schema、同 20 条：

    transformers（HF 模板，与训练一致）  85%  (17/20)
    llama-server（参数规范化后）          70%  (14/20)

`fold_tool_history=True` 时把历史工具调用**预先渲染成文本**放进 `content`，
并去掉 `tool_calls` 字段。模板对"没有 tool_calls 的 assistant"是原样输出
content，服务端没有可规范化的东西，两边就必然逐字节相同（已用
`build/diag_fold_history.py` 验证：6506 == 6506）。

**但这个方案实测无效，默认关闭。** 20 条样本：未折叠 70%，折叠后 60%。
prompt 字节相同却没提升，说明差异不在历史参数编码，而在更下游的采样/数值
环节（已排除分词：llama.cpp 与 HF 的 token id 完全相同；也已排除模板：两份
模板 2507 字符逐字节相同）。真正的原因是 llama-server 的默认采样器——
它自带 `repeat_penalty 1.1`、`top_k 20`、`top_p 0.8`、`min_p 0.05`，
这些在 argmax 之前就修改了分布，而 transformers 侧是纯贪心。修复办法是
启动时中和采样器，见 `LOCAL_MODEL.md`。

注意：这个折叠硬编码了 Qwen 的 `<tool_call>` / `<tool_response>` 文本格式，
只适用于本项目的微调 Qwen 模型；远程 DeepSeek 等按 OpenAI 标准格式训练，
**不要**开启。

共同的教训
----------
本项目已出现三例训练/推理格式漂移（工具调用标签、观察文本措辞、tools schema，
现在是第四例：历史参数编码）。凡是有两处代码各自生成同一份协议，就必须按
字节比对，而不能按意图相信。`tests/test_tool_schema_alignment.py` 与
`build/diag_fold_history.py` 负责挡住前两例，prompt 逐字节对比负责挡住这一类。

做法
----
`ChatOpenAI.bind_tools()` 内部就是 `super().bind(tools=<转换后的>)`，而 bind 的
kwargs 会原样进入请求体。所以只要绕过一次 `bind_tools`、直接传 `tools=`，
原始 schema 就能逐字节上链，不必重写 HTTP 与消息序列化——认证、重试、
`extra_body`（DeepSeek 的 thinking 开关）、异常类型都保持原样。

    llm = TrainingSchemaChatModel(init_chat_model(...))
    llm.bind_tools(EXCEL_TOOLS).invoke(messages)   # 发送训练原始 schema
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

# 由框架注入、模型无从填写的参数名。仅用于说明与实验开关。
INJECTED_PARAMETERS: tuple[str, ...] = ("state", "tool_call_id")


def _default_definitions() -> list[dict[str, Any]]:
    from tool_schema import tool_definitions

    return tool_definitions()


def strip_injected_params(schema: dict[str, Any]) -> dict[str, Any]:
    """移除注入参数与其 required 项。

    仅供实验：让 schema 看起来"正确"。实测会让 GGUF 从 6/8 掉到 3/8，
    因为它偏离了训练分布。要真正清理，必须在**重新训练**时改
    `tool_schema.describe_tool()`，让训练与推理同时换成干净版本。
    """
    cleaned = dict(schema)
    properties = cleaned.get("properties")
    if isinstance(properties, dict):
        cleaned["properties"] = {
            key: value
            for key, value in properties.items()
            if key not in INJECTED_PARAMETERS
        }
    required = cleaned.get("required")
    if isinstance(required, list):
        kept = [item for item in required if item not in INJECTED_PARAMETERS]
        if kept:
            cleaned["required"] = kept
        else:
            cleaned.pop("required", None)
    return cleaned


def fold_tool_history(messages: Any) -> Any:
    """把历史里的工具调用折叠成 Qwen 模板的纯文本形式。

    输入与输出都是 LangChain 消息列表。做两件事：

    - `AIMessage(tool_calls=[...])` → `AIMessage(content="<tool_call>…</tool_call>")`
      且不带 `tool_calls` 字段，走模板"原样输出 content"的分支。
    - `ToolMessage` → `HumanMessage("<tool_response>…</tool_response>")`，
      与模板对 `role == "tool"` 的渲染字节等价，同时避开"tool 消息必须紧跟在
      tool_calls 之后"的校验。

    `arguments` 刻意再 `json.dumps` 一次：训练数据里它本来就是字符串，
    模板的 `tojson` 会二次编码。保持双重编码才与训练一致。
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    folded: list[Any] = []
    for message in messages:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            parts: list[str] = []
            if message.content:
                parts.append(str(message.content))
            for call in message.tool_calls:
                name = call.get("name") or ""
                args = call.get("args")
                arguments = args if isinstance(args, str) else json.dumps(
                    args if args is not None else {}, ensure_ascii=False
                )
                parts.append(
                    '<tool_call>\n{"name": "'
                    + name
                    + '", "arguments": '
                    + json.dumps(arguments, ensure_ascii=False)
                    + "}\n</tool_call>"
                )
            folded.append(AIMessage(content="\n".join(parts)))
        elif isinstance(message, ToolMessage):
            folded.append(
                HumanMessage(
                    content="<tool_response>\n"
                    + str(message.content or "")
                    + "\n</tool_response>"
                )
            )
        else:
            folded.append(message)
    return folded


class TrainingSchemaChatModel:
    """透明包装：`bind_tools()` 固定发送训练格式的 schema。

    `fold_tool_history=True` 时还会把历史里的工具调用折叠成纯文本，
    让 llama.cpp 这类会规范化 `arguments` 的服务端渲染出与训练相同的
    prompt（详见模块文档）。该选项硬编码 Qwen 的文本格式，
    只适用于本项目微调的 Qwen 模型。

    **`fold_tool_history` 实测无效（70% → 60%），默认关闭。**
    保留实现是因为它把 prompt 差异压到了零，是定位"剩余差异不在这里"
    的关键实验。

    其余能力全部委托给内层模型，因此 `load_model.describe_model_error()`
    依据 `openai` 异常做的分类、以及 DeepSeek 的 `extra_body` 都仍然有效。
    """

    def __init__(
        self,
        inner: Any,
        *,
        definitions_factory: Callable[[], list[dict[str, Any]]] | None = None,
        strip_injected: bool = False,
        fold_tool_history: bool = False,
    ):
        self._inner = inner
        self._definitions_factory = definitions_factory or _default_definitions
        self._strip_injected = strip_injected
        self._fold_tool_history = fold_tool_history

    def definitions(self) -> list[dict[str, Any]]:
        """当前会真正发送出去的 tools 定义。"""
        definitions = self._definitions_factory()
        if self._strip_injected:
            for item in definitions:
                function = item.get("function") or {}
                if isinstance(function.get("parameters"), dict):
                    function["parameters"] = strip_injected_params(
                        function["parameters"]
                    )
        return definitions

    def _rebuild(self, inner: Any) -> "TrainingSchemaChatModel":
        return TrainingSchemaChatModel(
            inner,
            definitions_factory=self._definitions_factory,
            strip_injected=self._strip_injected,
            fold_tool_history=self._fold_tool_history,
        )

    def bind_tools(
        self,
        _tools: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> "TrainingSchemaChatModel":
        """忽略调用方传入的工具对象，改用训练格式的 schema。

        `agent.think()` 传的是 `EXCEL_TOOLS`（LangChain 工具对象），但模型在
        训练时见到的是 `tool_definitions()` 的字典格式，这里以后者为准。
        """
        kwargs.pop("strict", None)  # 训练时没有 strict，别额外加
        return self._rebuild(self._inner.bind(tools=self.definitions(), **kwargs))

    def bind(self, **kwargs: Any) -> "TrainingSchemaChatModel":
        return self._rebuild(self._inner.bind(**kwargs))

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        if self._fold_tool_history:
            messages = fold_tool_history(messages)
        return self._inner.invoke(messages, **kwargs)

    def stream(self, messages: Any, **kwargs: Any) -> Any:
        if self._fold_tool_history:
            messages = fold_tool_history(messages)
        return self._inner.stream(messages, **kwargs)

    @property
    def unwrapped(self) -> Any:
        return self._inner

    def __getattr__(self, name: str) -> Any:
        # 只在自身属性缺失时触发，其余能力透传给内层模型。
        return getattr(object.__getattribute__(self, "_inner"), name)
