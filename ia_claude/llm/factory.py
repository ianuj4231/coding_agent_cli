import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ia_claude.config import config
from ia_claude.observability.logger import get_logger

logger = get_logger(__name__)


def get_llm():
    """Return the configured LLM."""

    provider = config["llm"]["provider"]
    model = config["llm"]["model"]

    logger.info(
        f"Using LLM provider: {provider}, model: {model}"
    )

    if provider == "openrouter":
        return ChatOpenAI(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model)

    return ChatOpenAI(model=model)


def get_embedder():
    """Return the configured embedding model."""

    provider = config["embeddings"]["provider"]
    model = config["embeddings"]["model"]

    logger.info(
        f"Using embeddings provider: {provider}, model: {model}"
    )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model)

    return OpenAIEmbeddings(model=model)