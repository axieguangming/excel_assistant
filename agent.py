# agent.py
import io
import os
import re

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from load_model import init_llm
from state import AgentState

# ---------- LLM（OpenAI 兼容接口，可换 DeepSeek / 通义千问 / Ollama 等） ----------
# llm = ChatOpenAI(
#     model=os.getenv("MODEL_NAME", "qwen-plus"),
#     api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
#     base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
#     temperature=0,
# )
llm= init_llm()

MAX_RETRY = 3

# ---------- 沙箱：限制代码执行权限（Demo 级） ----------
import re as _re
import datetime as _dt
import math as _math

_ALLOWED_MODULES = {"pandas": pd, "numpy": np, "re": _re, "datetime": _dt, "math": _math}

def _safe_import(name, *args, **kwargs):
    if name in _ALLOWED_MODULES:
        return _ALLOWED_MODULES[name]
    raise ImportError(f"禁止导入模块: {name}")


SAFE_BUILTINS = {
    "__import__": _safe_import,
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "round": round, "sum": sum, "min": min, "max": max, "abs": abs,
    "sorted": sorted, "list": list, "dict": dict, "tuple": tuple, "set": set,
    "range": range, "enumerate": enumerate, "zip": zip, "map": map,
    "filter": filter, "print": print, "isinstance": isinstance,
}


def build_df_info(df: pd.DataFrame) -> str:
    """生成给 LLM 看的数据描述"""
    buf = io.StringIO()
    df.info(buf=buf)
    try:
        preview = df.head(8).to_markdown(index=False)
    except Exception:
        preview = df.head(8).to_string(index=False)
    return (f"数据形状: {df.shape[0]} 行 x {df.shape[1]} 列\n"
            f"列名: {list(df.columns)}\n\n"
            f"字段类型:\n{buf.getvalue()}\n"
            f"数据预览(前8行):\n{preview}")


def _fmt_history(history: list) -> str:
    if not history:
        return "（无）"
    return "\n".join(f"{h['role']}: {h['content'][:200]}" for h in history[-6:])


# 提取代码内容
def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


CODE_SYSTEM = """资深数据分析师，用户上传的 Excel 已加载为 pandas DataFrame，变量名为 df。
请根据用户问题编写 Python(pandas) 代码完成分析任务。

严格要求:
1. 只输出一段 ```python ``` 代码，不要任何解释。
2. 数据分析结果必须赋值给变量 result，且 result 必须是 pandas.DataFrame（即使只有一个汇总值，也要构造成 DataFrame，列名用中文便于阅读）。
3. 如果用户问题是闲聊或不需要数据处理，不要设置 result，改为给变量 answer_text 赋值一段文本回答。
4. 禁止读写文件、网络请求；只允许使用 pandas/numpy/re/datetime/math。
5. 注意列名的大小写与空格，必要时先清洗数据。环境中已导入 pd、np。
"""

# ---------- 节点 1：生成代码 ----------
def generate_code(state: AgentState):
    user = (f"数据信息:\n{state['df_info']}\n\n"
            f"历史对话:\n{_fmt_history(state['history'])}\n\n"
            f"当前问题: {state['question']}")
    if state.get("error"):  # 带着报错信息重试
        user += (f"\n\n⚠️ 上一次生成的代码执行报错:\n{state['error']}\n"
                 f"上一次的代码:\n{state.get('code', '')}\n"
                 f"请定位并修正错误，重新输出完整代码。")
    resp = llm.invoke([SystemMessage(content=CODE_SYSTEM), HumanMessage(content=user)])
    return {"code": _extract_code(resp.content), "error": ""}


# ---------- 节点 2：执行代码 ----------
def execute_code(state: AgentState):
    namespace = {"df": state["df"].copy(), "pd": pd, "np": np,
                 "result": None, "answer_text": None}
    try:
        exec(state["code"], {"__builtins__": SAFE_BUILTINS}, namespace)
        result, answer_text = namespace.get("result"), namespace.get("answer_text") or ""
        if result is not None and not isinstance(result, pd.DataFrame):
            result = pd.DataFrame([result]) if isinstance(result, (int, float, str)) \
                     else pd.DataFrame(result)
        if result is None and not answer_text:
            return {"error": "代码执行成功但未产生 result 或 answer_text，请补充。",
                    "retries": state["retries"] + 1}
        return {"result_df": result, "answer_text": answer_text, "error": ""}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "retries": state["retries"] + 1}


# ---------- 条件路由：报错则重试，否则总结 ----------
def route_after_exec(state: AgentState):
    if state.get("error") and state["retries"] < MAX_RETRY:
        return "generate_code"
    return "summarize"


# ---------- 节点 3：总结结论 ----------
def summarize(state: AgentState):
    rdf = state.get("result_df")
    if rdf is not None:
        preview = rdf.head(20).to_string(index=False, max_colwidth=30)
        prompt = (f"用户问题: {state['question']}\n"
                  f"代码执行成功，结果形状 {rdf.shape}，预览:\n{preview}\n\n"
                  f"请用简洁的中文总结分析结论（3~5 句，可适当引用关键数字），不要输出表格。")
        return {"final_answer": llm.invoke([HumanMessage(content=prompt)]).content}
    if state.get("answer_text"):
        return {"final_answer": state["answer_text"]}
    return {"final_answer": f"抱歉，经过 {MAX_RETRY} 次尝试仍未完成分析。最后错误：{state.get('error')}"}


# ---------- 组装 Graph ----------
graph = StateGraph(AgentState)
graph.add_node("generate_code", generate_code)
graph.add_node("execute_code", execute_code)
graph.add_node("summarize", summarize)
graph.set_entry_point("generate_code")
graph.add_edge("generate_code", "execute_code")
graph.add_conditional_edges("execute_code", route_after_exec,
                            {"generate_code": "generate_code", "summarize": "summarize"})
graph.add_edge("summarize", END)
agent_app = graph.compile()


def run_agent(df: pd.DataFrame, question: str, history: list) -> dict:
    """对外入口：返回最终状态"""
    init_state = {
        "df": df, "df_info": build_df_info(df), "question": question,
        "history": history, "code": "", "error": "", "retries": 0,
        "result_df": None, "answer_text": "", "final_answer": "",
    }
    return agent_app.invoke(init_state, config={"recursion_limit": 30})