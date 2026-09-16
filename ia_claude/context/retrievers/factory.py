from ia_claude.config import config
from ia_claude.observability.logger import get_logger


logger = get_logger(__name__)


def get_retriever():
    """Return the appropriate retriever based on vector store configuration."""

    provider = config["vector_store"]["provider"].lower()

    logger.info(
        f"Using vector store for retrieval: {provider}"
    )

    if provider == "qdrant":

        retrieval_mode = config["vector_store"].get(
            "retrieval_mode",
            "dense",
        ).lower()

        if retrieval_mode in ("hybrid", "sparse"):

            from .hybrid_qdrant import retrieve

            logger.info(
                f"Using Qdrant {retrieval_mode} retriever"
            )

            return retrieve

        else:

            from .semantic_qdrant import retrieve

            logger.info(
                "Using Qdrant dense retriever"
            )

            return retrieve

    elif provider == "chroma":

        from .semantic_chroma import retrieve

        logger.info(
            "Using Chroma retriever"
        )

        return retrieve

    else:

        raise ValueError(
            f"Unsupported vector store provider: {provider}"
        )