from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    file_path: str | None       # 上传后本地excel路径
    sheet_names: list[str] | None
    file_meta: dict | None      # 列名、行数信息
    output_file_path: str | None# 修改后输出excel路径