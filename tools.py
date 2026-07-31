import pandas as pd

_workbook_store: dict[str, pd.DataFrame]= {}

def load_excel_to_store(file_path: str, session_id):
    df= pd.read_excel(file_path, engine="openpyxl")
    _workbook_store[session_id]= df
    return df

def save_df_to_excel(session_id, output_path: str):
    df= _workbook_store.get(session_id)
    if df:
        df.to_excel(output_path, index=False, engine="openpyxl")

def get_df(session_id):
    return _workbook_store.get(session_id)

def get_excel_info(session_id: str):
    df= get_df(session_id)
    if df is None:
        return "当前没有加载任何excel文件"
    info= {
        "columns": df.columns.tolist(),
        "shape": list(df.shape),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "head": df.head().to_dict(orient="records")
    }

ALL_TOOLS= [get_excel_info]