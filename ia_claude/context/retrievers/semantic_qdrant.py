import os

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore

from ia_claude.config import config
from ia_claude.llm.factory import get_embedder
from ia_claude.observability.logger import get_logger


load_dotenv()

logger = get_logger(__name__)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Embed the query and find the k most similar chunks in Qdrant.

    Returns a list of dictionaries containing:
    - content
    - source
    - name
    - type
    - start_line
    - end_line
    - distance
    """

    embedder = get_embedder()

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise ValueError(
            "QDRANT_URL is not set in the environment."
        )

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedder,
        url=url,
        api_key=api_key,
        collection_name=config["qdrant"]["collection_name"],
    )

    logger.info(
        f"Retrieving top {k} chunks for query: {query}"
    )

    results = vector_store.similarity_search_with_score(
        query,
        k=k,
    )

    chunks = []

    for doc, score in results:

        meta = doc.metadata

        chunk = {
            "content": doc.page_content,
            "source": meta["source"],
            "name": meta["name"],
            "type": meta["type"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "distance": score,
        }

        chunks.append(chunk)

        logger.debug(
            f"Retrieved {meta['type']} "
            f"'{meta['name']}' from {meta['source']} "
            f"(score: {score:.4f})"
        )

    logger.info(
        f"Retrieved {len(chunks)} chunks"
    )

    return chunks