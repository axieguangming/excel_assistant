"""
基于 LangGraph 的 Excel 处理 Agent
"""
import os
import json
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.redis import RedisSaver
from langchain_openai import ChatOpenAI
from agent.state import AgentState
from agent.tools import EXCEL_TOOLS
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

""" 初始化llm等配置"""
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
    model= model.bind_tools(EXCEL_TOOLS)
    return model

load_dotenv()

llm = init_llm()
llm = llm.bind_tools(EXCEL_TOOLS)

def agent_node(state: AgentState):
    system_prompt = """
你是Excel分析助手。用户上传Excel文件后，根据用户自然语言，调用工具完成：
1.先查看sheet列表，再查看表头样例；
2.使用工具完成统计、筛选、修改表格；
3.修改文件必须调用write_modified_excel输出新文件；
4.不要编造数据，全部依赖工具返回结果。
当前文件路径:{file_path}
""".format(file_path=state.get("file_path"))
    msgs = [{"role":"system", "content":system_prompt}] + state["messages"]
    resp = llm.invoke(msgs)
    return {"messages":[resp]}

def should_continue(state: AgentState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

# 图定义
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(EXCEL_TOOLS))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools":"tools", END:END})
builder.add_edge("tools", "agent")

# checkpoint：开发内存，生产换RedisSaver
checkpointer = MemorySaver()
# checkpointer = RedisSaver.from_conn_string("redis://localhost:6379")

graph = builder.compile(checkpointer=checkpointer)