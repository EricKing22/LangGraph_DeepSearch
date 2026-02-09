from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from src import config

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
    base_url="https://api.minimax.io",
)

ollama_llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)

question_model = config.QUESTION_MODEL.lower()
if "qwen" in question_model:
    question_llm = ChatOpenAI(
        model=question_model,
        temperature=config.QWEN_TEMPERATURE,
        api_key=config.QWEN_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
else:
    print(
        f"Unsupported question model: {question_model}. No LLM will be configured for question nodes."
    )
    question_llm = None

report_model = config.REPORT_MODEL.lower()
if "qwen" in report_model:
    report_llm = ChatOpenAI(
        model=report_model,
        temperature=config.QWEN_TEMPERATURE,
        api_key=config.QWEN_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
else:
    print(
        "Unsupported report model: {report_model}. No LLM will be configured for report nodes."
    )
    report_llm = None

__all__ = ["question_llm", "report_llm"]
