"""Agent 状态定义"""
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import add_messages


class ExcelAgentState(TypedDict):
    """Excel Agent 的全局状态"""
    messages: Annotated[List, add_messages]   # 对话历史（自动追加）
    file_path: Optional[str]                  # 当前上传的 Excel 路径
    file_name: Optional[str]                  # 文件名
    sheet_names: Optional[List[str]]          # Sheet 列表
    df_summary: Optional[str]                 # 数据结构摘要（给 LLM 看的）
    analysis_result: Optional[str]            # 分析结果文本
    chart_config: Optional[str]               # ECharts 配置 JSON 字符串
    error: Optional[str]                      # 错误信息