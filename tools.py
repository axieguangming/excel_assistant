import pandas as pd
import json
from typing import Optional
from langchain.tools import tool, InjectedToolArg
from typing_extensions import Annotated

_workbook_store: dict[str, pd.DataFrame]= {}

def load_excel_to_store(file_path: str, session_id: str):
    df= pd.read_excel(file_path, engine="openpyxl")
    _workbook_store[session_id]= df
    return df

def save_df_to_excel(session_id: str, output_path: str):
    df= _workbook_store.get(session_id)
    if df:
        df.to_excel(output_path, index=False, engine="openpyxl")

def get_df(session_id: str):
    return _workbook_store.get(session_id, None)

# ========== 工具定义 ==========
# 注意：session_id 使用 InjectedToolArg 标记
# 这样 LLM 不会看到也不需要填写这个参数
# 它由框架在运行时自动注入

@tool
def get_excel_info(
    session_id: Annotated[str, InjectedToolArg]
):
    """获取当前 Excel 文件的基本信息：列名、行数、数据类型、前5行预览。
    在处理任何需求之前，务必先调用此工具了解数据结构。"""
    df= get_df(session_id)
    if df is None:
        return "当前没有加载任何excel文件"
    info= {
        "columns": df.columns.tolist(),
        "shape": list(df.shape),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "head": df.head().to_dict(orient="records"),
        "missing_values": {col: int(cnt) for col, cnt in df.isnull().sum().items() if cnt > 0},
    }
    return json.dumps(info, ensure_ascii=False, default=str)

ALL_TOOLS= [get_excel_info]
TOOL_BY_NAME= {tool.name: tool for tool in ALL_TOOLS}

# 诊断：检查注入参数是否被识别
for t in ALL_TOOLS:
    print(f"{t.name}:")
    print(f"  injected_arg_names: {getattr(t, 'injected_arg_names', 'NOT_FOUND')}")
    print(f"  args_schema fields: {list(t.args_schema.model_fields.keys()) if t.args_schema else 'None'}")
    # 检查 session_id 是否在 schema 中被标记为 excluded
    if t.args_schema and 'session_id' in t.args_schema.model_fields:
        field = t.args_schema.model_fields['session_id']
        print(f"  session_id exclude: {field.exclude}")
        print(f"  session_id metadata: {field.metadata}")