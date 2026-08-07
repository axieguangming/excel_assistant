"""Excel 操作工具集"""
import json
import pandas as pd
import duckdb
from langchain_core.tools import tool


# ============ 列名清洗（修复 #2：冲突去重） ============

def sanitize_columns(df: pd.DataFrame) -> tuple:
    """统一清洗列名：去除首尾空格，空格/横线替换为下划线。
    重复列名自动加后缀 _1, _2 ...
    返回 (清洗后的 df, {清洗后列名: 原始列名} 映射)"""
    mapping = {}
    new_cols = []
    seen: dict[str, int] = {}

    for col in df.columns:
        original = str(col)
        clean = original.strip().replace(" ", "_").replace("-", "_")

        # 处理冲突
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0

        mapping[clean] = original
        new_cols.append(clean)

    df = df.copy()
    df.columns = new_cols
    return df, mapping


# ============ LangChain Tools ============

@tool
def read_excel_structure(file_path: str) -> str:
    """读取Excel文件的结构信息，包括所有sheet名称、每个sheet的列名、数据类型和前5行数据。
    用于了解文件内容，帮助用户提问。"""
    try:
        xls = pd.ExcelFile(file_path)
        info_parts = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            df_clean, col_map = sanitize_columns(df)
            col_info = "\n".join(
                f"  - {col} ({dtype})"
                for col, dtype in zip(df_clean.columns, df_clean.dtypes)
            )
            info_parts.append(
                f"📋 Sheet: [{sheet}] | 行数: {len(df)} | 列数: {len(df.columns)}\n"
                f"⚠️ SQL查询时列名中的空格和横线已替换为下划线\n"
                f"列信息:\n{col_info}\n"
                f"前5行预览:\n{df_clean.head().to_string(index=False)}"
            )
        return "\n\n".join(info_parts)
    except Exception as e:
        return f"读取失败: {str(e)}"


@tool
def query_excel_with_sql(file_path: str, sql: str) -> str:
    """使用SQL查询Excel数据。每个sheet名就是表名，可以直接SELECT/WHERE/GROUP BY/JOIN等。
    列名中的空格和横线已替换为下划线。
    适用于：筛选、聚合、排序、多表关联等分析场景。"""
    try:
        con = duckdb.connect()
        xls = pd.ExcelFile(file_path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            df_clean, _ = sanitize_columns(df)
            con.register(sheet, df_clean)

        result = con.execute(sql).fetchdf()
        con.close()

        if len(result) == 0:
            return "查询结果为空，请检查SQL条件。"
        if len(result) > 50:
            return (
                f"结果共{len(result)}行，展示前50行:\n"
                f"{result.head(50).to_string(index=False)}"
            )
        return result.to_string(index=False)
    except Exception as e:
        return f"SQL执行错误: {str(e)}\n请检查SQL语法，确保表名(sheet名)和列名正确。"


@tool
def filter_and_transform(file_path: str, sheet_name: str, operation: str) -> str:
    """对指定sheet执行筛选或变换操作。
    operation格式示例:
    - filter:column_name==value
    - sort:column_name:asc/desc
    - group_sum:group_col:sum_col
    - top_n:column_name:n
    注意：列名中的空格和横线已替换为下划线。"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df, _ = sanitize_columns(df)

        if operation.startswith("filter:"):
            parts = operation[7:].split("==")
            if len(parts) != 2:
                return "filter 格式错误，应为 filter:column_name==value"
            col, val = parts[0].strip(), parts[1].strip()
            if col not in df.columns:
                return f"列 '{col}' 不存在。可用列: {list(df.columns)}"
            result = df[df[col].astype(str) == val]
        elif operation.startswith("sort:"):
            parts = operation[5:].split(":")
            col = parts[0]
            order = parts[1] if len(parts) > 1 else "asc"
            if col not in df.columns:
                return f"列 '{col}' 不存在。可用列: {list(df.columns)}"
            result = df.sort_values(col, ascending=(order == "asc"))
        elif operation.startswith("top_n:"):
            parts = operation[6:].split(":")
            if len(parts) != 2:
                return "top_n 格式错误，应为 top_n:column_name:n"
            col, n = parts[0], int(parts[1])
            if col not in df.columns:
                return f"列 '{col}' 不存在。可用列: {list(df.columns)}"
            result = df.nlargest(n, col)
        elif operation.startswith("group_sum:"):
            parts = operation[10:].split(":")
            if len(parts) != 2:
                return "group_sum 格式错误，应为 group_sum:group_col:sum_col"
            group_col, sum_col = parts[0], parts[1]
            if group_col not in df.columns:
                return f"列 '{group_col}' 不存在。可用列: {list(df.columns)}"
            if sum_col not in df.columns:
                return f"列 '{sum_col}' 不存在。可用列: {list(df.columns)}"
            result = df.groupby(group_col)[sum_col].sum().reset_index()
        else:
            return f"不支持的操作: {operation}"

        return f"操作完成，结果{len(result)}行:\n{result.to_string(index=False)}"
    except Exception as e:
        return f"操作失败: {str(e)}"


@tool
def generate_chart_config(chart_type: str, title: str, data_json: str) -> str:
    """生成ECharts图表配置。
    chart_type: bar/line/pie/scatter
    title: 图表标题
    data_json: JSON格式的数据，例如 {"categories": ["A","B"], "values": [10, 20]}
    返回ECharts option配置JSON字符串。"""
    try:
        data = json.loads(data_json)
        categories = data.get("categories", [])
        values = data.get("values", [])

        if chart_type == "pie":
            series_data = [
                {"name": c, "value": v} for c, v in zip(categories, values)
            ]
            option = {
                "title": {"text": title, "left": "center"},
                "tooltip": {"trigger": "item"},
                "series": [
                    {"type": "pie", "radius": "60%", "data": series_data}
                ],
            }
        else:
            option = {
                "title": {"text": title, "left": "center"},
                "tooltip": {"trigger": "axis"},
                "xAxis": {"type": "category", "data": categories},
                "yAxis": {"type": "value"},
                "series": [
                    {"type": chart_type, "data": values, "name": title}
                ],
            }

        return json.dumps(option, ensure_ascii=False)
    except Exception as e:
        return f"图表生成失败: {str(e)}"


@tool
def get_column_stats(file_path: str, sheet_name: str, column_name: str) -> str:
    """获取指定列的统计信息：均值、中位数、标准差、最大最小值、缺失值数量等。
    注意：列名中的空格和横线已替换为下划线。"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df, _ = sanitize_columns(df)

        if column_name not in df.columns:
            return f"列 '{column_name}' 不存在。可用列: {list(df.columns)}"

        col = df[column_name]
        if col.dtype in ("int64", "float64", "int32", "float32"):
            stats = (
                f"列 [{column_name}] 统计信息:\n"
                f"  数据类型: {col.dtype}\n"
                f"  非空数量: {col.count()}\n"
                f"  缺失数量: {col.isna().sum()}\n"
                f"  均值: {col.mean():.4f}\n"
                f"  中位数: {col.median():.4f}\n"
                f"  标准差: {col.std():.4f}\n"
                f"  最小值: {col.min()}\n"
                f"  最大值: {col.max()}\n"
                f"  25%分位: {col.quantile(0.25):.4f}\n"
                f"  75%分位: {col.quantile(0.75):.4f}"
            )
        else:
            stats = (
                f"列 [{column_name}] 统计信息:\n"
                f"  数据类型: {col.dtype}\n"
                f"  非空数量: {col.count()}\n"
                f"  唯一值数量: {col.nunique()}\n"
                f"  最常见值:\n{col.value_counts().head(5).to_string()}"
            )
        return stats
    except Exception as e:
        return f"统计失败: {str(e)}"


# 工具列表
ALL_TOOLS = [
    read_excel_structure,
    query_excel_with_sql,
    filter_and_transform,
    generate_chart_config,
    get_column_stats,
]