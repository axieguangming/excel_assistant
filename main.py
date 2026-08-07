import os
import json
import uuid
from pathlib import Path
from typing import TypedDict, Annotated, Sequence
import operator

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

# ========= 配置 =========
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", api_key)
MODEL_NAME = "deepseek-v4-flash"

app = FastAPI()
file_store = dict()

HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>LangGraph Excel Agent</title>
    <style>
        body {font-family:system-ui;max-width:900px;margin:20px auto;padding:0 20px;}
        .box{border:1px solid #ccc;padding:16px;border-radius:8px;margin:12px 0;}
        textarea{width:100%;height:80px;padding:8px;box-sizing:border-box;}
        button{padding:8px 16px;cursor:pointer;}
        pre{background:#f5f5f5;padding:10px;overflow:auto;white-space:pre-wrap;}
    </style>
</head>
<body>
<h2>Excel Agent | LangGraph Demo</h2>
<div class="box">
    <h4>1.上传Excel</h4>
    <input type="file" id="fileInput" accept=".xlsx,.xls">
    <div id="fileMsg"></div>
</div>

<div class="box">
    <h4>2.提问关于表格数据</h4>
    <textarea id="queryInput" placeholder="例如：统计销售额总和，找出最大的一行"></textarea>
    <br>
    <button id="sendBtn">执行Agent</button>
</div>

<div class="box">
    <h4>Agent输出</h4>
    <pre id="result"></pre>
</div>

<script>
let uploadFileId = null;
const fileInput = document.getElementById('fileInput');
const sendBtn = document.getElementById('sendBtn');
const queryInput = document.getElementById('queryInput');
const resultDom = document.getElementById('result');
const fileMsg = document.getElementById('fileMsg');

fileInput.addEventListener('change', async (e)=>{
    const f = e.target.files[0];
    if(!f) return;
    const fd = new FormData();
    fd.append('file', f);
    const resp = await fetch("/upload", {method:"POST", body:fd});
    const data = await resp.json();
    if(data.ok){
        uploadFileId = data.file_id;
        fileMsg.innerText = `✅已上传: ${data.filename}, file_id=${uploadFileId}`;
    }else{
        fileMsg.innerText = `❌${data.msg}`;
    }
});

sendBtn.onclick = async ()=>{
    if(!uploadFileId){
        alert("请先上传excel");
        return;
    }
    const query = queryInput.value.trim();
    if(!query) return;
    resultDom.innerText = "Agent思考中...";
    const resp = await fetch("/agent/run",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({file_id:uploadFileId, query:query})
    });
    const out = await resp.json();
    resultDom.innerText = JSON.stringify(out, null, 2);
}
</script>
</body>
</html>
"""

# ========= 1.定义Agent状态 =========
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    file_path: str


# ========= 2.定义工具 =========
@tool
def read_excel(file_path: str) -> str:
    """读取excel文件，返回表头与前几行样例数据，用于了解表格结构
    Args:
        file_path: excel本地路径
    """
    df = pd.read_excel(file_path)
    info = f"""
columns: {list(df.columns)}
shape: {df.shape}
head(5):
{df.head(5).to_csv(sep="|")}
    """
    return info


@tool
def pandas_run(file_path: str, code: str) -> str:
    """执行pandas分析代码，对excel做统计、筛选、计算。
    只能使用变量df，df = pd.read_excel(file_path)
    返回执行结果字符串，不要返回大段原始df。
    Args:
        file_path: excel路径
        code: pandas代码片段，例如 "res = df['sales'].sum()\nreturn str(res)"
    """
    df = pd.read_excel(file_path)
    local_vars = {"df": df}
    try:
        exec(code, {}, local_vars)
        ret = local_vars.get("res", "代码执行完成无返回")
        return str(ret)
    except Exception as e:
        return f"执行异常:{str(e)}"

tools = [read_excel, pandas_run]

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

llm = init_llm().bind_tools(tools)


# ========= 3.构建LangGraph 工作流 =========
def agent_node(state: AgentState):
    messages = state["messages"]
    resp = llm.invoke(messages)
    return {"messages": [resp]}


def should_continue(state: AgentState):
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return END


tool_node = ToolNode(tools)

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()

graph_app = build_graph()


# ========= 4.FastAPI接口 =========
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    fid = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{fid}_{file.filename}"
    with open(save_path, "wb") as f:
        f.write(await file.read())
    file_store[fid] = str(save_path)
    return {"ok": True, "file_id": fid, "filename": file.filename}


@app.post("/agent/run")
async def agent_run(payload: dict):
    fid = payload["file_id"]
    user_query = payload["query"]
    if fid not in file_store:
        return {"error": "file_id不存在"}
    fp = file_store[fid]

    input_state = {
        "messages": [HumanMessage(content=f"你是Excel数据分析Agent，使用工具完成用户需求。用户问题：{user_query}")],
        "file_path": fp
    }
    result = graph_app.invoke(input_state)
    msgs = [m.model_dump() for m in result["messages"]]
    return {"messages": msgs, "final_answer": msgs[-1]["content"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)