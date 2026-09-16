import chromadb

from ia_claude.config import config
from ia_claude.llm.factory import get_embedder
from ia_claude.observability.logger import get_logger


logger = get_logger(__name__)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Embed the query and find the k most similar chunks in ChromaDB.
    """

    # 1. Load the same embedding model used during indexing
    embedder = get_embedder()

    # 2. Connect to our existing persistent ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=config["chromadb"]["persist_dir"]
    )

    collection = chroma_client.get_or_create_collection(
        name=config["chromadb"]["collection_name"]
    )

    logger.info(f"Retrieving top {k} chunks for query: {query}")

    # 3. Convert user's question into an embedding
    query_embedding = embedder.embed_query(query)

    # 4. Search Chroma for the closest vectors
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # 5. Convert Chroma's response into a clean format
    chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        chunks.append(
            {
                "content": document,
                "source": metadata["source"],
                "name": metadata["name"],
                "type": metadata["type"],
                "start_line": metadata["start_line"],
                "end_line": metadata["end_line"],
                "distance": distance,
            }
        )

        logger.debug(
            f"Retrieved {metadata['type']} "
            f"'{metadata['name']}' "
            f"from {metadata['source']} "
            f"(distance: {distance:.4f})"
        )

    logger.info(f"Retrieved {len(chunks)} chunks")

    return chunks