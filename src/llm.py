from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import config

# 初始化 OpenAI LLM
openai_llm = ChatOpenAI(
    model=config.OPENAI_MODEL,
    temperature=config.OPENAI_TEMPERATURE,
    api_key=config.OPENAI_API_KEY,
)

qwen_llm = ChatOpenAI(
    model=config.QWEN_MODEL,
    temperature=config.QWEN_TEMPERATURE,
    api_key=config.QWEN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

minmax_llm = ChatOpenAI(
    model=config.MINMAX_MODEL,
    temperature=config.MINMAX_TEMPERATURE,
    api_key=config.MINMAX_API_KEY,
    base_url="https://api.minimax.io/v1",
)

ollama_llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)
