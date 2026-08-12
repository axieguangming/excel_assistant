from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os
import json

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
    return model