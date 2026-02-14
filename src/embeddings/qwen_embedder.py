from openai import AsyncOpenAI
from src.config import QWEN_API_KEY

# Dashscope API Base URL
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = AsyncOpenAI(api_key=QWEN_API_KEY, base_url=DASHSCOPE_BASE_URL)


async def aembed_texts(texts: list[str]) -> list[list[float]]:
    """
    qwen embedding
    """
    model_name = "text-embedding-v3"

    response = await client.embeddings.create(model=model_name, input=texts)

    return [e.embedding for e in response.data]
