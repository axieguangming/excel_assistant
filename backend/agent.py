"""LangGraph Excel Agent 核心逻辑"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from backend.state import ExcelAgentState
from backend.tools import ALL_TOOLS
from dotenv import load_dotenv
import os
import json
from langchain.chat_models import init_chat_model
import logging
logger = logging.getLogger(__name__)

# ============ LLM 配置 ============
def load_api_config():
    load_dotenv()
    # 加载api参数
    model_name= "deepseek-v4-flash"
    api_name= "deepseek_api"
    api_json= os.getenv(api_name)
    api= json.loads(api_json)
    api_key= api['api_key']
    base_url= api['base_url']
    return model_name, api_key, base_url

def init_llm():
    model_name, api_key, base_url= load_api_config()
    # 加载llm
    model= init_chat_model(
        model=model_name,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        extra_body={"thinking": {"type": "disabled"}}
    )
    return model

llm = init_llm()

# 绑定工具
tools_map = {t.name: t for t in ALL_TOOLS}
llm_with_tools = llm.bind_tools(ALL_TOOLS)

# 定义每个工具需要的自动注入参数（修复 #2 原版问题）
TOOL_AUTO_INJECT = {
    "read_excel_structure": ["file_path"],
    "query_excel_with_sql": ["file_path"],
    "filter_and_transform": ["file_path"],
    "get_column_stats": ["file_path"],
}


# ============ 节点定义 ============

def analyze_intent_node(state: ExcelAgentState) -> dict:
    """意图解析 + 决策节点"""

    file_context = ""
    if state.get("df_summary"):
        file_context = f"\n\n当前已上传文件信息:\n{state['df_summary']}"

    # backend/agent.py - analyze_intent_node 中的 system_prompt 修改

    system_prompt = f"""你是一个专业的Excel数据分析助手。你可以帮助用户：
1. 查看和理解Excel文件结构
2. 用SQL查询和分析数据
3. 筛选、排序、聚合数据
4. 生成统计报告
5. 生成可视化图表

⚠️⚠️⚠️ 极其重要的规则 ⚠️⚠️⚠️
- 调用任何工具时，**绝对不要**填写 file_path 参数！系统会自动注入正确的文件路径。
- 如果你填写了 file_path（包括"已上传文件"、"当前文件"等任何值），工具将会失败！
- 只需要填写除 file_path 以外的其他参数。

使用规则：
- 如果用户问的是文件结构，调用 read_excel_structure（不传 file_path）
- 如果用户需要数据分析/查询，优先使用 query_excel_with_sql（用SQL，不传 file_path）
- 如果用户需要统计信息，使用 get_column_stats（不传 file_path）
- 如果用户需要图表，先查询数据，再调用 generate_chart_config
- 回答要简洁专业，使用中文
- 如果文件未上传，提醒用户先上传文件
- SQL中列名的空格和横线已替换为下划线
{file_context}"""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    logger.info(
        "[analyze_intent] tool_calls: %s",
        getattr(response, "tool_calls", None),
    )
    return {"messages": [response]}

# backend/agent.py - execute_tools_node 完整替换

import os

# 定义每个工具需要的自动注入参数
TOOL_AUTO_INJECT = {
    "read_excel_structure": ["file_path"],
    "query_excel_with_sql": ["file_path"],
    "filter_and_transform": ["file_path"],
    "get_column_stats": ["file_path"],
}

# 这些值被视为"无效占位符"，应被覆盖
INVALID_FILE_PATH_VALUES = {
    "已上传文件", "上传的文件", "当前文件", "the uploaded file",
    "uploaded file", "current file", "", "none", "null",
}


def _is_invalid_file_path(value: str, state: ExcelAgentState) -> bool:
    """判断 file_path 是否是 LLM 编造的无效占位符"""
    if not value or not isinstance(value, str):
        return True
    stripped = value.strip().lower()
    # 检查是否在黑名单中
    if stripped in {v.lower() for v in INVALID_FILE_PATH_VALUES}:
        return True
    # 检查是否是真实存在的路径
    real_path = state.get("file_path")
    if real_path and value != real_path and not os.path.exists(value):
        return True
    return False


def execute_tools_node(state: ExcelAgentState) -> dict:
    """工具执行节点：执行LLM决定的工具调用"""
    last_message = state["messages"][-1]
    results = []

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = dict(tool_call["args"])

        # ✅ 智能注入：不仅检查 key 是否存在，还检查值是否有效
        inject_fields = TOOL_AUTO_INJECT.get(tool_name, [])
        for field in inject_fields:
            real_value = state.get(field)
            if not real_value:
                continue

            current_value = tool_args.get(field)
            # 情况1：key 不存在 → 注入
            # 情况2：key 存在但值是无效占位符 → 强制覆盖
            if field not in tool_args or _is_invalid_file_path(current_value, state):
                tool_args[field] = real_value
                logger.warning(
                    "[execute_tools] 覆盖无效 %s: '%s' → '%s'",
                    field, current_value, real_value,
                )

        logger.info("[execute_tools] 调用: %s, 参数: %s", tool_name, tool_args)

        try:
            if tool_name in tools_map:
                result = tools_map[tool_name].invoke(tool_args)
            else:
                result = f"未知工具: {tool_name}"
        except Exception as e:
            result = f"工具执行错误 [{tool_name}]: {str(e)}"
            logger.error("[execute_tools] 错误: %s", e)

        results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": results}


def summarize_node(state: ExcelAgentState) -> dict:
    """总结节点（修复 #6：用 isinstance 做健壮判断）"""

    # 从历史消息中提取图表配置
    chart_config = None
    for msg in reversed(state["messages"]):
        content = getattr(msg, "content", "")
        if not content:
            continue
        try:
            parsed = json.loads(str(content))
            if isinstance(parsed, dict) and "series" in parsed:
                chart_config = json.dumps(parsed, ensure_ascii=False)
                break
        except (json.JSONDecodeError, TypeError):
            continue

    last_message = state["messages"][-1]

    # 判断最后一条是否是 AI 的直接回复（非工具调用）
    is_direct_reply = (
        isinstance(last_message, AIMessage)
        and not getattr(last_message, "tool_calls", None)
    )

    if is_direct_reply:
        return {
            "analysis_result": last_message.content,
            "chart_config": chart_config,
        }

    # 有工具结果需要总结
    summarize_prompt = HumanMessage(
        content=(
            "请根据以上工具执行结果，用简洁的中文给出分析总结。"
            "如果有数据表格，保留关键数据。"
        )
    )
    response = llm.invoke(state["messages"] + [summarize_prompt])
    return {
        "analysis_result": response.content,
        "chart_config": chart_config,
        "messages": [response],
    }


# ============ 路由逻辑 ============

def route_after_analysis(state: ExcelAgentState) -> str:
    """判断是否需要执行工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"
    return "summarize"


# ============ 构建 Graph ============

def build_excel_agent():
    """构建并编译 Excel Agent 图"""

    workflow = StateGraph(ExcelAgentState)

    # 添加节点
    workflow.add_node("analyze_intent", analyze_intent_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("summarize", summarize_node)

    # 设置入口
    workflow.set_entry_point("analyze_intent")

    # 添加边
    workflow.add_conditional_edges(
        "analyze_intent",
        route_after_analysis,
        {
            "execute_tools": "execute_tools",
            "summarize": "summarize",
        },
    )
    workflow.add_edge("execute_tools", "analyze_intent")
    workflow.add_edge("summarize", END)

    # 编译（带记忆）
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# 全局 Agent 实例
excel_agent = build_excel_agent()