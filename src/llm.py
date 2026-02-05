from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from src import config

# 初始化 OpenAI LLM
openai_llm = ChatOpenAI(
    model=config.OPENAI_MODEL,
    temperature=config.OPENAI_TEMPERATURE,
    api_key=config.OPENAI_API_KEY,
)

minmax_llm = ChatOpenAI(
    model=config.MINMAX_MODEL,
    temperature=config.MINMAX_TEMPERATURE,
    api_key=config.MINMAX_API_KEY,
    base_url="https://api.minmax.com/v1",
)

ollama_llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)
