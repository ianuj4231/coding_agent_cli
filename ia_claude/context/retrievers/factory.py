from ia_claude.config import config
from ia_claude.observability.logger import get_logger


logger = get_logger(__name__)


def get_retriever():
    """Return the appropriate retriever based on vector store configuration."""

    provider = config["vector_store"]["provider"].lower()
    retrieval_mode = config["vector_store"].get(
        "retrieval_mode",
        "",
    ).lower()

    logger.info(
        f"Using vector store for retrieval: {provider} ({retrieval_mode})"
    )

    if provider == "qdrant" and retrieval_mode == "hybrid":
        from .hybrid_qdrant import retrieve

        logger.info("Using Qdrant hybrid retriever")
        return retrieve

    raise ValueError(
        "Unsupported vector store configuration: "
        f"provider={provider!r}, retrieval_mode={retrieval_mode!r}. "
        "Use provider='qdrant' and retrieval_mode='hybrid'."
    )
