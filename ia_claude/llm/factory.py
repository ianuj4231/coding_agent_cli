import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ia_claude.config import config
from ia_claude.observability.logger import get_logger

logger = get_logger(__name__)


class FastEmbedEmbeddings(Embeddings):
    """LangChain adapter for FastEmbed's lightweight ONNX runtime."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def get_llm():
    """Return the configured LLM."""

    provider = config["llm"]["provider"]
    model = config["llm"]["model"]

    logger.info(
        f"Using LLM provider: {provider}, model: {model}"
    )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model)

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


@lru_cache(maxsize=1)
def get_embedder():
    """Return the configured embedding model, loading it once per process."""

    provider = config["embeddings"]["provider"]
    model = config["embeddings"]["model"]

    logger.info(
        f"Using embeddings provider: {provider}, model: {model}"
    )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model)

    if provider == "fastembed":
        return FastEmbedEmbeddings(model_name=model)

    return OpenAIEmbeddings(model=model)
