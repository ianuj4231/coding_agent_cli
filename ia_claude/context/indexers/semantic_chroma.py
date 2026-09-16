import chromadb

from ia_claude.config import config
from ia_claude.context.indexers.code_parser import (
    parse_file,
    get_source_files,
)
from ia_claude.llm.factory import get_embedder
from ia_claude.observability.logger import get_logger

logger = get_logger(__name__)


def index_codebase(repo_path: str) -> chromadb.Collection:
    """Parse, embed, and store all code chunks in ChromaDB."""

    embedder = get_embedder()

    chroma_client = chromadb.PersistentClient(
        path=config["chromadb"]["persist_dir"]
    )

    collection = chroma_client.get_or_create_collection(
        name=config["chromadb"]["collection_name"]
    )

    files = get_source_files(repo_path)

    logger.info(f"Indexing {len(files)} source files")

    for filepath in files:

        try:
            chunks = parse_file(filepath)

        except (SyntaxError, ValueError) as e:
            logger.error(f"Skipping {filepath}: {e}")
            continue

        for chunk in chunks:

            # 1. CHUNK
            logger.debug(
                f"Embedding {chunk.type} '{chunk.name}'"
            )

            # 2. EMBEDDING
            embedding = embedder.embed_query(chunk.content)

            # 3. UNIQUE ID
            doc_id = (
                f"{chunk.source}::{chunk.name}::{chunk.start_line}"
            )

            # 4. STORE IN CHROMA
            collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk.content],
                metadatas=[
                    {
                        "source": chunk.source,
                        "name": chunk.name,
                        "type": chunk.type,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    }
                ],
            )

    logger.info(
        f"Indexing complete. Total chunks: {collection.count()}"
    )

    return collection