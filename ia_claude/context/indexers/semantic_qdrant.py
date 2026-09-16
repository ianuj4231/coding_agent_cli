import os

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from ia_claude.config import config
from ia_claude.context.indexers.code_parser import (
    parse_file,
    get_source_files,
)
from ia_claude.llm.factory import get_embedder
from ia_claude.observability.logger import get_logger


load_dotenv()

logger = get_logger(__name__)


def index_codebase(repo_path: str) -> QdrantVectorStore:
    """
    Parse all source files in repo_path, embed each chunk,
    and store the chunks in Qdrant.

    If the Qdrant collection already contains data,
    load and return the existing collection.
    """

    embedder = get_embedder()

    collection_name = config["qdrant"]["collection_name"]

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise ValueError(
            "QDRANT_URL is not set in the environment."
        )

    # Connect to Qdrant
    client = QdrantClient(
        url=url,
        api_key=api_key,
    )

    # Check whether the collection already exists
    existing_collections = [
        collection.name
        for collection in client.get_collections().collections
    ]

    if collection_name in existing_collections:
        info = client.get_collection(collection_name)

        if info.points_count > 0:
            logger.info(
                f"Loaded existing Qdrant index "
                f"with {info.points_count} chunks"
            )

            return QdrantVectorStore.from_existing_collection(
                embedding=embedder,
                url=url,
                api_key=api_key,
                collection_name=collection_name,
            )

    # Collection does not exist or is empty
    logger.info(
        f"Starting semantic indexing of {repo_path}"
    )

    files = get_source_files(repo_path)

    docs = []

    logger.info(
        f"Found {len(files)} source files"
    )

    for filepath in files:

        try:
            chunks = parse_file(filepath)

        except (SyntaxError, ValueError) as e:
            logger.error(
                f"Skipping {filepath}: {e}"
            )
            continue

        for chunk in chunks:

            logger.debug(
                f"Embedding {chunk.type} '{chunk.name}'"
            )

            docs.append(
                Document(
                    page_content=chunk.content,
                    metadata={
                        "source": chunk.source,
                        "name": chunk.name,
                        "type": chunk.type,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    },
                )
            )

    if not docs:
        logger.warning(
            "No documents found to index."
        )

        return QdrantVectorStore.from_existing_collection(
            embedding=embedder,
            url=url,
            api_key=api_key,
            collection_name=collection_name,
        )

    # Create the Qdrant vector store and index documents
    vector_store = QdrantVectorStore.from_documents(
        docs,
        embedder,
        url=url,
        api_key=api_key,
        collection_name=collection_name,
        batch_size=50,
    )

    logger.info(
        f"Semantic indexing complete. "
        f"Total chunks: {len(docs)}"
    )

    return vector_store


def show_index(vector_store: QdrantVectorStore) -> None:
    """Display all documents and embeddings stored in Qdrant."""

    from rich.console import Console

    console = Console()

    client = vector_store.client

    collection_name = config["qdrant"]["collection_name"]

    results = client.scroll(
        collection_name=collection_name,
        with_payload=True,
        with_vectors=True,
        limit=1000,
    )

    points = results[0]

    console.print(
        f"\n[bold]Semantic Index — "
        f"{len(points)} chunks[/bold]\n"
    )

    for i, point in enumerate(points):

        payload = point.payload or {}

        emb = point.vector or []

        console.print(
            f"[bold cyan]── Chunk {i + 1} "
            f"──────────────────────────[/bold cyan]"
        )

        console.print(
            f"  File      : {payload.get('source', '')}"
        )

        console.print(
            f"  Name      : "
            f"{payload.get('name', '')} "
            f"({payload.get('type', '')})"
        )

        console.print(
            f"  Lines     : "
            f"{payload.get('start_line', '')} - "
            f"{payload.get('end_line', '')}"
        )

        console.print(
            f"  Embedding : "
            f"[{', '.join(f'{v:.4f}' for v in emb[:5])}...] "
            f"({len(emb)} dims)"
        )

        console.print(
            f"  Code      :\n"
            f"[dim]{payload.get('page_content', '')[:300]}"
            f"[/dim]\n"
        )



