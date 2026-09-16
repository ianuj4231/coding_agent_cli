from ia_claude.config import config
from ia_claude.observability.logger import get_logger


logger = get_logger(__name__)


def get_indexer():
    """Return the appropriate indexer based on vector store configuration."""

    provider = config["vector_store"]["provider"].lower()

    logger.info(
        f"Using vector store: {provider}"
    )

    if provider == "qdrant":

        retrieval_mode = config["vector_store"].get(
            "retrieval_mode",
            "dense",
        ).lower()

        if retrieval_mode in ("hybrid", "sparse"):

            from .hybrid_qdrant import index_codebase

            logger.info(
                f"Using Qdrant {retrieval_mode} indexer"
            )

            return index_codebase

        else:

            from .semantic_qdrant import index_codebase

            logger.info(
                "Using Qdrant dense indexer"
            )

            return index_codebase

    elif provider == "chroma":

        from .semantic_chroma import index_codebase

        logger.info(
            "Using Chroma indexer"
        )

        return index_codebase

    else:

        raise ValueError(
            f"Unsupported vector store provider: {provider}"
        )