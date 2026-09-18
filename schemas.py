from pydantic import BaseModel
from typing import List, Any


# 定义聊天请求类，设定需要的内容
class ChatRequest(BaseModel):
    session_id: str
    question: str


# Excel 插件：把当前工作表数据（表头 + 数据行）上传以建立会话
class SessionRequest(BaseModel):
    columns: List[str]
    rows: List[List[Any]]
