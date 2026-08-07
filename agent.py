import os
from typing import Annotated, TypedDict, Sequence, Literal   # 类型注解
from langchain.chat_models import init_chat_model   # 加载llm
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage, RemoveMessage    # message
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, BaseMessage
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import json
from tools import ALL_TOOLS, TOOL_BY_NAME
from pydantic import BaseModel, Field
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

""" 初始化llm等配置"""
def load_api_config():
    load_dotenv()
    # 加载api参数
    model_name= "deepseek-v4-flash"
    api_name= "deepseek_api"
    api_json= os.getenv(api_name)
    api= json.loads(api_json)
    api_key= api['api_key']
    base_url= api['base_url']
    return model_name, api_key, base_url

def init_llm():
    model_name, api_key, base_url= load_api_config()
    # 加载llm
    model= init_chat_model(
        model=model_name,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        extra_body={"thinking": {"type": "disabled"}}
    )
    model= model.bind_tools(ALL_TOOLS)
    return model

""" 定义状态 """
# 反思结果信息
class ReflectionResult(BaseModel):
    score: int= Field(description="打分，0-100分")
    reason: str= Field(description="反思的结果和理由")

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    history_messages_summary: str
    llm_calls: int
    reflection_result: ReflectionResult | None
    round_msg_ids: list[list[str]]

""" 定义图节点 """
# thought model node
system_message= SystemMessage(
"""
（1）你正在思考阶段(thought_node)，你需要根据前面的内容进行思考并生成回复。
（2）有一些工具可以使用，每个工具有不同的功能，你可以通过调用相关工具(tool_node)得到相关的结果,如果可以调用工具完成，应该优先调用工具，因为调用工具的结果可靠性高。
（3）如果你不确定你的想法是否正确，或者你认为你已经得到了答案，那么可以进入反思阶段。
""")

reflection_message= SystemMessage(
""" 
（1）你正在反思阶段(reflection_node)，你需要综合前面的所有信息，对最新的思考内容进行反思，判断是否合理，并给出原因，reason填写反思结果和原因。并且进行打分(0-100分，分数越高表示最新的思考内容越正确越好)，score填写分数。
"""
)
def get_nodes(system_message, reflection_message, model):
    window_size= 2
    max_size= 5

    # ========== 全局工具：生成唯一消息ID ==========
    def generate_msg_id() -> str:
        """生成全局唯一消息UUID，保证msg.id不为None"""
        return str(uuid.uuid4())

    def concatenate_summary_messages(state: dict, pre_prompt, sub_prompt):
        summary= state.get("history_messages_summary", "")
        summary= SystemMessage(f"历史摘要：{summary}")
        pre_prompt_msg= SystemMessage(pre_prompt) if not isinstance(pre_prompt, SystemMessage) else pre_prompt
        sub_prompt_msg= SystemMessage(sub_prompt) if not isinstance(sub_prompt, SystemMessage) else sub_prompt
        messages= state['messages']
        context= []
        context= context+ [pre_prompt_msg] if pre_prompt_msg.content != "" else context
        context= context+ [summary] if summary.content != "" else context
        context= context+ messages
        context= context+ [sub_prompt_msg] if sub_prompt_msg.content != "" else context
        return context

    def summarize_node(state: dict):
        messages= state['messages']
        round_msg_ids= state.get('round_msg_ids', [])

        if len(round_msg_ids) >= max_size:  # 当窗口轮次数超过限制时，更新摘要，并清理窗口信息
            summary= state.get('history_messages_summary', "")

            # 获取需要摘要的msg ids
            need_smrz_msg_ids= []
            for i in range(0, len(round_msg_ids)-window_size):
                need_smrz_msg_ids.extend(round_msg_ids[i])
            # 获取对应的msg
            last_history= [msg for msg in messages if msg.id in need_smrz_msg_ids]

            # 构造摘要请求prompt
            if summary == "":
                request= f"对以下内容进行摘要，提取和保留重要的内容：{last_history}"
            else:
                request= f"这是现在的历史摘要：{summary}, 这是最新的历史记录：{last_history}，请根据这两部分生成新的历史摘要，提取和保留重要的内容。"
            # 生成摘要
            resp_summary= model.invoke(request).content
            print(f"摘要：\n{resp_summary}")

            # 要删除的历史记录
            delete_obj= [RemoveMessage(id=msg.id) for msg in last_history]
            # 要保留的msg ids
            keep_msg_ids= round_msg_ids[-window_size:]
            keep_msg_ids.append([]) # 添加一个空列表用于存储本轮的msg ids
            return {
                "history_messages_summary": resp_summary, 
                "messages": delete_obj,
                "round_msg_ids": keep_msg_ids
            }
        else:   # 不需要更新摘要
            round_msg_ids.append([]) # 添加一个空列表用于存储本轮的msg ids
            return {"round_msg_ids": round_msg_ids}

    def get_msg_ids(msg_list):
        msg_ids= []
        for msg in msg_list:
            msg_ids.append(msg.id)
        return msg_ids
        
    def thought_node(state: dict):
        context= concatenate_summary_messages(state=state, pre_prompt=system_message, sub_prompt="")
        response= model.invoke(context)
        response.id= generate_msg_id()
        msg_ids= get_msg_ids([response])
        # print(f"thought_node: msg.id:{msg_ids}")
        round_msg_ids= state['round_msg_ids']
        round_msg_ids[-1].extend(msg_ids)
        return {
            "messages": [response], 
            "llm_calls": state.get("llm_calls", 0)+1,
            "round_msg_ids": round_msg_ids
        }

    # tool node
    def tool_node(state: dict, config: RunnableConfig):
        session_id = config.get("configurable", {}).get("session_id")
        if not session_id:
            raise ValueError("session_id not found in config['configurable']")
        print(f"[DEBUG] configurable: {config.get('configurable', {})}")
         # ★ 手动将 session_id 注入到每个 tool_call 的 args 中
        messages = state["messages"]
        last_msg = messages[-1]
        
        patched_tool_calls = []
        for tc in last_msg.tool_calls:
            patched_args = {**tc["args"], "session_id": session_id}
            patched_tool_calls.append({**tc, "args": patched_args})
        
        # 用 patch 后的消息替换最后一条
        patched_msg = last_msg.model_copy(update={"tool_calls": patched_tool_calls})
        patched_state = {**state, "messages": messages[:-1] + [patched_msg]}

        bt_tool_node= ToolNode(ALL_TOOLS)
        result= bt_tool_node.invoke(patched_state, config=config)
        new_tool_messages = result["messages"]
        new_msg_ids = [msg.id for msg in new_tool_messages]

        old_round_msg_ids = state.get("round_msg_ids", [])
        updated_round_msg_ids = [
            list(ids) for ids in old_round_msg_ids  # 深拷贝
        ]
        if updated_round_msg_ids:
            updated_round_msg_ids[-1] = updated_round_msg_ids[-1] + new_msg_ids
        else:
            updated_round_msg_ids.append(new_msg_ids)

        return {
            "messages": new_tool_messages,
            "round_msg_ids": updated_round_msg_ids,
        }

        # results= []
        # for tool_call in state['messages'][-1].tool_calls:
        #     # 找到工具
        #     tool= TOOL_BY_NAME[tool_call['name']]
        #     try:
        #         # 执行工具调用
        #         observation= tool.invoke(tool_call['args'])
        #     except Exception as e:
        #         observation= f"工具执行异常：{str(e)}"
        #     results.append(ToolMessage(content=observation, tool_call_id=tool_call['id'], id=generate_msg_id()))
        # msg_ids= get_msg_ids(results)
        # # print(f"tool_node: msg.id:{msg_ids}")

        # round_msg_ids= state['round_msg_ids']
        # round_msg_ids[-1].extend(msg_ids)
        # return {
        #     "messages": results,
        #     "round_msg_ids": round_msg_ids
        # }

    # reflection node
    def reflection_node(state: dict):
        context= concatenate_summary_messages(state=state, pre_prompt="", sub_prompt=reflection_message)
        # 强制结构化输出
        structured_out: ReflectionResult= model.with_structured_output(ReflectionResult).invoke(context)
        ret_message= AIMessage(content=f"反思分数：{structured_out.score}，理由：{structured_out.reason}", id=generate_msg_id())
        msg_ids= get_msg_ids([ret_message])
        # print(f"reflection_node: msg.id:{msg_ids}")

        round_msg_ids= state['round_msg_ids']
        round_msg_ids[-1].extend(msg_ids)
        return {
            "messages": [ret_message], 
            "llm_calls": state.get("llm_calls", 0)+ 1,
            "reflection_result": structured_out,
            "round_msg_ids": round_msg_ids
        }

    return summarize_node, thought_node, tool_node, reflection_node

""" 定义边的选择规则 """
# 调用工具，或者反思
def reflection_or_tool(state: State) -> Literal["tool_node", "reflection_node"]:
    last_msg= state['messages'][-1]
    if last_msg.tool_calls:
        return "tool_node"
    return "reflection_node"

# 根据反思进行下一步思考，或者结束
def thought_or_end(state: State) -> Literal['thought_node', END]:
    score= state['reflection_result'].score
    call_times= state['llm_calls']
    if score>=80 or call_times>15:
        return END
    return "thought_node"

""" 构建图 """
def build_agent():
    model= init_llm()
    summarize_node, thought_node, tool_node, reflection_node= get_nodes(system_message, reflection_message, model)
    agent_builder= StateGraph(State)
    # 添加节点
    agent_builder.add_node("summarize_node", summarize_node)
    agent_builder.add_node("thought_node", thought_node)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_node("reflection_node", reflection_node)
    # 添加边
    agent_builder.add_edge(START, "summarize_node")
    agent_builder.add_edge("summarize_node", "thought_node")
    agent_builder.add_conditional_edges("thought_node", reflection_or_tool, ["tool_node", "reflection_node"])
    agent_builder.add_edge("tool_node", "thought_node")
    agent_builder.add_conditional_edges("reflection_node", thought_or_end, ['thought_node', END])
    # 编译
    checkpointer= MemorySaver()
    agent= agent_builder.compile(checkpointer= checkpointer)

    return agent

""" 运行agent """
agent= build_agent()

def run_agent(message: str, session_id: str):
    print(f"message: {message}, session_id: {session_id}")
    # 任务配置
    config= {
            'configurable': {
                "thread_id": "task_001",
                "session_id": session_id
            }
        }
    messages= [HumanMessage(message)]
    results= agent.invoke({"messages": messages}, config= config)
    print(results)
    return results['messages'][-1]

""" 流式运行agent """
def run_agent_stream(message: str):
    for event in agent.stream({"messages": message}, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "agent":
                for msg in node_output.get("messages", []):
                    if isinstance(msg, AIMessage):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield {
                                    "type": "tool_call",
                                    "tool": tc["name"],
                                    "args": tc["args"],
                                }
                        elif msg.content:
                            yield {
                                "type": "response",
                                "content": msg.content,
                            }
            elif node_name == "tools":
                for msg in node_output.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        yield {
                            "type": "tool_result",
                            "tool": msg.name,
                            "content": msg.content[:500],  # 截断过长结果
                        }