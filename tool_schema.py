"""工具 schema 的**唯一**生成入口。

为什么单独一个模块
------------------
训练侧（`finetune/dataset.py`、`finetune/local_llm.py`）与推理侧
（`finetune/eval_sft.py`、`load_model.py`）必须发同一份 schema。
之前训练侧自己生成、推理侧交给 LangChain `convert_to_openai_tool()`，
结果两边不一致，模型在 GGUF 部署上从 85% 掉到 55%。

具体差多少
----------
    tool_definitions() 原始输出   5,704 字符
    LangChain 转换后               4,558 字符
    9 个工具里 0 个完全相同

其中 `describe_data` 差异最大（380 vs 158 字符）：它没有显式 `args_schema`，
`model_json_schema()` 会把 `InjectedState` / `InjectedToolCallId` 当成
`state`、`tool_call_id` 两个必填参数暴露出来。

看着不合理，但**这就是模型在训练时见到的东西**，因此推理侧必须原样复现。
实测同服务同 8 条样本：干净 schema 3/8，训练原始 schema 6/8。

注意
----
本模块只依赖 `excel_ops`，刻意不引入 pandas / dataset_builder 等重依赖，
这样打包后的应用（`load_model.py`）也能低成本复用。
"""

from __future__ import annotations

from typing import Any

import excel_ops


def describe_tool(tool: Any) -> dict[str, Any]:
    """把一个 LangChain 工具转成 OpenAI tools 定义。"""
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


def tool_definitions() -> list[dict[str, Any]]:
    """返回与训练数据完全一致的 OpenAI tools 定义。"""
    return [describe_tool(tool) for tool in excel_ops.EXCEL_TOOLS]
