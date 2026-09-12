from pydantic import BaseModel


# 定义聊天请求类，设定需要的内容
class ChatRequest(BaseModel):
    session_id: str
    question: str