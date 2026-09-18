from typing import TypedDict, Any, Optional


# ---------- 状态定义 ----------
class AgentState(TypedDict):
    df: Any                      # 上传的 DataFrame
    df_info: str                 # 数据结构描述（喂给 LLM）
    question: str                # 用户问题
    history: list                # 多轮对话历史
    code: str                    # LLM 生成的 pandas 代码
    error: str                   # 执行报错信息
    retries: int                 # 已重试次数
    result_df: Optional[Any]     # 结果 DataFrame
    answer_text: str             # 纯文本回答（闲聊类问题）
    final_answer: str            # 最终文字结论
    chart_file: str              # 生成的图表文件URL
    user_prompt: str             # 本轮喂给 LLM 的完整 prompt（用于采集微调数据）