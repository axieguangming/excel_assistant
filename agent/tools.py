from langchain_core.tools import tool
import pandas as pd
import os

@tool
def list_excel_sheets(file_path: str) -> str:
    """获取excel所有sheet名称，用户需要知道有哪些工作表时调用"""
    excel_file = pd.ExcelFile(file_path)
    return str(excel_file.sheet_names)

@tool
def read_sheet_head(file_path: str, sheet_name: str, nrows:int=10) -> str:
    """读取excel指定sheet前N行，看列名与样例数据"""
    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=nrows)
    return f"columns:{list(df.columns)}\n{df.to_string()}"

@tool
def run_pandas_query(file_path: str, sheet_name: str, query_code: str) -> str:
    """
    执行pandas分析查询，不要print，return结果。
    query_code示例：df.groupby("部门")["销售额"].sum()
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    local_vars = {"df":df, "pd":pd}
    try:
        res = eval(query_code, {}, local_vars)
        return str(res)
    except Exception as e:
        return f"执行错误：{str(e)}"

@tool
def write_modified_excel(file_path: str, sheet_name: str, output_path: str, python_modify_code: str) -> str:
    """
    修改excel并另存为output_path。
    python_modify_code操作df对象，修改完成后df.to_excel(output_path, index=False)
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    local_vars = {"df":df, "pd":pd, "output_path":output_path}
    exec(python_modify_code, {}, local_vars)
    return f"文件已输出到 {output_path}"

EXCEL_TOOLS = [list_excel_sheets, read_sheet_head, run_pandas_query, write_modified_excel]