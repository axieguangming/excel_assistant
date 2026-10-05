# agent.py
import os
import re
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _plt

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from load_model import init_llm
from state import AgentState

_HAS_MPL = True


llm= init_llm()

MAX_RETRY = 3

# ---------- 沙箱：限制代码执行权限（Demo 级） ----------
import re as _re
import datetime as _dt
import math as _math

_real_import = __import__

_ALLOWED_ROOTS = {"pandas", "numpy", "re", "datetime", "math"}
if _HAS_MPL:
    _ALLOWED_ROOTS.add("matplotlib")

def _safe_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root in _ALLOWED_ROOTS:
        return _real_import(name, *args, **kwargs)
    raise ImportError(f"禁止导入模块: {name}")


SAFE_BUILTINS = {
    "__import__": _safe_import,
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "round": round, "sum": sum, "min": min, "max": max, "abs": abs,
    "sorted": sorted, "list": list, "dict": dict, "tuple": tuple, "set": set,
    "range": range, "enumerate": enumerate, "zip": zip, "map": map,
    "filter": filter, "print": print, "isinstance": isinstance,
}

OUTPUT_DIR = "outputs"
_chart_counter = 0

def _make_save_chart():
    """创建 save_chart 闭包，用于在沙箱中保存 matplotlib 图表并返回其访问URL"""
    def save_chart(fig=None, name=None):
        global _chart_counter
        if not _HAS_MPL:
            raise RuntimeError("matplotlib 未安装，无法绘图")
        if name is None:
            _chart_counter += 1
            name = f"chart_{_chart_counter}"
        import uuid
        safe_name = re.sub(r'[^\w\u4e00-\u9fff-]', '_', str(name))
        fname = f"chart_{safe_name}_{uuid.uuid4().hex[:6]}.png"
        fpath = os.path.join(OUTPUT_DIR, fname)
        target = fig if hasattr(fig, 'savefig') else _plt.gcf()
        target.savefig(fpath, dpi=150, bbox_inches='tight', facecolor='white')
        _plt.close(target)
        return f"/api/chart/{fname}"
    return save_chart


def _fmt_history(history: list) -> str:
    if not history:
        return "（无）"
    return "\n".join(f"{h['role']}: {h['content'][:200]}" for h in history[-6:])


# 提取代码内容
def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


CODE_SYSTEM = """资深的数据分析师，用户上传的 Excel 已加载为 pandas DataFrame，变量名为 df。
请根据用户问题编写 Python(pandas) 代码完成分析任务。

严格要求:
1. 只输出一段 ```python ``` 代码，不要任何解释。
2. 数据分析结果必须赋值给变量 result，且 result 必须是 pandas.DataFrame（即使只有一个汇总值，也要构造成 DataFrame，列名用中文便于阅读）。
3. 如果用户问题是闲聊或不需要数据处理，不要设置 result，改为给变量 answer_text 赋值一段文本回答。
4. 禁止读写文件、网络请求；只允许使用 pandas/numpy/re/datetime/math/matplotlib。
5. 注意列名的大小写与空格，必要时先清洗数据。环境中已导入 pd、np。
6. 当用户希望可视化（柱状图/折线图/饼图/散点图等），用 matplotlib 绘图：
   - 环境已注入 plt（即 matplotlib.pyplot）和 save_chart 函数。
   - 绘图前设置中文字体：plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
   - 图表需有标题和坐标轴标签。
   - 保存图表：chart_url = save_chart(plt.gcf(), "图表名称")  或直接 chart_url = save_chart(name="图表名称")
   - 即使绘图，也必须同时给出 result DataFrame（图表对应的数据），让结果表可展示。
"""

# ---------- 训练数据采集：记录 (prompt → code) 对，供后续微调 ----------
DATA_DIR = os.getenv("TRAIN_DATA_DIR", "data")

def _log_sample(prompt: str, code: str, ok: bool, err: str = ""):
    """把一次代码生成结果追加写入 JSONL，成功样本是主要微调语料。"""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        record = {"prompt": prompt, "code": code, "ok": ok, "err": err}
        with open(os.path.join(DATA_DIR, "codegen_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 记录失败不影响主流程

# ---------- 节点 1：生成代码 ----------
def _build_prompt(state: AgentState) -> str:
    user = (f"数据信息:\n{state['df_info']}\n\n"
            f"历史对话:\n{_fmt_history(state['history'])}\n\n"
            f"当前问题: {state['question']}")
    if state.get("error"):  # 带着报错信息重试
        user += (f"\n\n⚠️ 上一次生成的代码执行报错:\n{state['error']}\n"
                 f"上一次的代码:\n{state.get('code', '')}\n"
                 f"请定位并修正错误，重新输出完整代码。")
    return user

def generate_code(state: AgentState):
    user = _build_prompt(state)
    resp = llm.invoke([SystemMessage(content=CODE_SYSTEM), HumanMessage(content=user)])
    return {"code": _extract_code(resp.content), "error": "", "user_prompt": user}


# ---------- 节点 2：执行代码 ----------
def execute_code(state: AgentState):
    namespace = {"df": state["df"].copy(), "pd": pd, "np": np,
                 "result": None, "answer_text": None, "chart_url": None}
    if _HAS_MPL:
        namespace["plt"] = _plt
        namespace["matplotlib"] = matplotlib
        namespace["save_chart"] = _make_save_chart()
    try:
        exec(state["code"], {"__builtins__": SAFE_BUILTINS}, namespace)
        result, answer_text = namespace.get("result"), namespace.get("answer_text") or ""
        chart_file = namespace.get("chart_url") or ""
        if result is not None and not isinstance(result, pd.DataFrame):
            result = pd.DataFrame([result]) if isinstance(result, (int, float, str)) \
                     else pd.DataFrame(result)
        if result is None and not answer_text:
            err = "代码执行成功但未产生 result 或 answer_text，请补充。"
            _log_sample(state.get("user_prompt", ""), state["code"], False, err)
            return {"error": err, "retries": state["retries"] + 1}
        _log_sample(state.get("user_prompt", ""), state["code"], True)
        return {"result_df": result, "answer_text": answer_text, "error": "",
                "chart_file": chart_file}
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        _log_sample(state.get("user_prompt", ""), state["code"], False, err)
        return {"error": err, "retries": state["retries"] + 1}


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


