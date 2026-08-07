# from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
# from fastapi.staticfiles import StaticFiles
# import uuid, os

# from agent.graph import graph
# from agent.state import AgentState
# from langchain_core.messages import HumanMessage

# app = FastAPI()
# UPLOAD_DIR="./static/upload"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.post("/upload")
# async def upload_file(file:UploadFile=File(...), thread_id:str=""):
#     if not thread_id:
#         thread_id = str(uuid.uuid4())
#     save_path = os.path.join(UPLOAD_DIR, f"{thread_id}_{file.filename}")
#     with open(save_path, "wb") as f:
#         f.write(await file.read())
#     # 将文件路径写入该thread的state
#     config = {"configurable":{"thread_id":thread_id}}
#     graph.update_state(config, {"file_path":save_path})
#     return {"thread_id":thread_id, "file_path":save_path}

# @app.websocket("/ws/{thread_id}")
# async def websocket_endpoint(websocket:WebSocket, thread_id:str):
#     await websocket.accept()
#     config = {"configurable":{"thread_id":thread_id}}
#     try:
#         while True:
#             user_query = await websocket.receive_text()
#             input_state:AgentState = {
#                 "messages":[HumanMessage(content=user_query)]
#             }
#             # aream流式返回每一步节点事件
#             async for chunk in graph.astream(input_state, config, stream_mode="updates"):
#                 await websocket.send_json(chunk)
#     except WebSocketDisconnect:
#         pass

# if __name__=="__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)