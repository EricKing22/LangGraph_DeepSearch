"""
Configuration management module - Load environment variables from .env file
"""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def get_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def get_int(key: str, default: int = 0) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def get_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_TEMPERATURE = get_float("OPENAI_TEMPERATURE", 0.7)

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_TEMPERATURE = get_float("QWEN_TEMPERATURE", 0.7)

MINMAX_API_KEY = os.getenv("MINMAX_API_KEY", "")
MINMAX_MODEL = os.getenv("MINMAX_MODEL", "MiniMax-M2.1")
MINMAX_TEMPERATURE = get_float("MINMAX_TEMPERATURE", 0.7)

# Tools
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SERP_API_KEY = os.getenv("SERP_API_KEY", "")
SERP_ENABLED = get_bool("SERP_ENABLED", True)

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")
SEARXNG_ENABLED = get_bool("SEARXNG_ENABLED", False)

WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID", "")

# Search
MAX_SUB_QUESTIONS: int = get_int("MAX_SUB_QUESTIONS", 5)
MAX_SEARCH_RESULTS: int = get_int("MAX_SEARCH_RESULTS", 5)
SEARCH_TIMEOUT: int = get_int("SEARCH_TIMEOUT", 10)

# Scraping
SCRAPING_STRATEGY = os.getenv("SCRAPING_STRATEGY", "crawl4ai")
MAX_SCRAPE_PAGES = get_int("MAX_SCRAPE_PAGES", 5)
SCRAPE_TIMEOUT = get_int("SCRAPE_TIMEOUT", 30)


RERANKER_MODEL = os.getenv("RERANKER_MODEL", "jina")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
INFINITY_API_URL = os.getenv("INFINITY_API_URL", "http://localhost:7997")

DEBUG = get_bool("DEBUG", False)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
