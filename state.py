from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


# ---------- 状态定义 ----------
class AgentState(TypedDict):
    df: Any                      # 上传的 DataFrame
    df_info: str                 # 数据结构描述（喂给 LLM）
    question: str                # 用户问题
    history: list                # 多轮对话历史
    messages: Annotated[list[Any], add_messages]  # LangChain Agent / tool 对话消息
    current_df: Any               # 工具链当前处理中的 DataFrame
    plans: list[str]              # 本轮实际调用的工具计划
    tool_steps: int               # 已执行的工具数
    turns: int                    # 模型决策轮数
    result_df: Any                # 处理后的结果 DataFrame
    final_answer: str             # 最终文字结论
    chart_file: str               # 生成的图表文件 URL
    code: str                     # 兼容旧 API 的工具调用记录
    user_prompt: str              # 用于可选训练日志的提示词