from __future__ import annotations

import math
import os,uuid
import time
from typing import Callable, TypeVar
from dotenv import load_dotenv
from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode,
    FastEmbedSparse,
)
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import ResponseHandlingException

from ia_claude.config import config
from ia_claude.context.indexers.code_parser import (
    parse_file,
    get_source_files,
)
from ia_claude.llm.factory import get_embedder
from ia_claude.observability.logger import get_logger
from ia_claude.user_context import validate_user_id

load_dotenv()

logger = get_logger(__name__)

RETRIEVAL_MODE_MAP = {
    "dense": RetrievalMode.DENSE,
    "sparse": RetrievalMode.SPARSE,
    "hybrid": RetrievalMode.HYBRID,
}

# Qdrant returns JSON floating-point payload values. At Unix-epoch scale, a
# Windows file timestamp can differ by one float ULP after a round trip (about
# 0.24 microseconds). File identity must therefore use integer nanoseconds,
# not an exact comparison of floating-point seconds.
_LEGACY_MTIME_ABS_TOLERANCE_SECONDS = 1e-6
_QDRANT_MAX_ATTEMPTS = 3
_QDRANT_RETRY_BASE_DELAY_SECONDS = 1.0
_T = TypeVar("_T")


def _retry_qdrant(operation: Callable[[], _T], description: str) -> _T:
    """Retry transient Qdrant transport failures with bounded backoff."""
    for attempt in range(1, _QDRANT_MAX_ATTEMPTS + 1):
        try:
            return operation()
        except ResponseHandlingException:
            if attempt == _QDRANT_MAX_ATTEMPTS:
                logger.exception(
                    "Qdrant operation failed after %d attempts: %s",
                    attempt,
                    description,
                )
                raise

            delay = _QDRANT_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Transient Qdrant failure (%s), retrying attempt %d/%d in %.1fs",
                description,
                attempt + 1,
                _QDRANT_MAX_ATTEMPTS,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError("Unreachable Qdrant retry state")


def _get_retrieval_mode() -> RetrievalMode:
    mode = config["vector_store"].get(
        "retrieval_mode",
        "hybrid",
    ).lower()

    if mode not in RETRIEVAL_MODE_MAP:
        raise ValueError(
            f"Unsupported retrieval mode: {mode}. "
            f"Use dense, sparse, or hybrid."
        )

    return RETRIEVAL_MODE_MAP[mode]


def _get_client_and_settings() -> tuple[QdrantClient, str, any, FastEmbedSparse, RetrievalMode, str, str | None]:
    """Initialize core connections, embedders, and hybrid configuration settings."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise ValueError("QDRANT_URL is not set in the environment.")

    client = QdrantClient(url=url, api_key=api_key)
    collection_name = config["qdrant"]["collection_name"]
    embedder = get_embedder()
    sparse_embedder = FastEmbedSparse(model_name="Qdrant/bm25")
    retrieval_mode = _get_retrieval_mode()

    return client, collection_name, embedder, sparse_embedder, retrieval_mode, url, api_key


def _ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Index metadata fields used by mandatory ownership/source filters."""
    try:
        for field_name in ("metadata.user_id", "metadata.source"):
            _retry_qdrant(
                lambda field_name=field_name: client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                    wait=True,
                ),
                f"create payload index {field_name}",
            )
        logger.debug("Ensured keyword payload indexes for user and source.")
    except Exception as exc:
        # A fresh collection is created later in index_codebase; in that one
        # specific case there is no collection to index yet. Every other
        # failure must propagate so a caller never claims a fresh index when
        # delete-by-source cannot be guaranteed.
        if "doesn't exist" in str(exc).lower() or "not found" in str(exc).lower():
            logger.debug("Collection %s does not exist yet; source index deferred.", collection_name)
            return
        logger.exception("Unable to ensure metadata.source payload index for %s", collection_name)
        raise


def _get_indexed_mtimes(
    client: QdrantClient,
    collection_name: str,
    user_id: str,
) -> dict[str, tuple[int | None, float]]:
    """Map each source to its stored ``(mtime_ns, mtime_seconds)`` version.

    ``mtime_ns`` is the authoritative value for new points. ``mtime_seconds``
    is retained to read collections written by older versions of this project.
    """
    mtimes: dict[str, tuple[int | None, float]] = {}
    try:
        offset = None
        while True:
            records, offset = _retry_qdrant(
                lambda: client.scroll(
                    collection_name=collection_name,
                    with_payload=[
                        "metadata.source",
                        "metadata.mtime",
                        "metadata.mtime_ns",
                    ],
                    with_vectors=False,
                    scroll_filter=qdrant_models.Filter(
                        must=[qdrant_models.FieldCondition(
                            key="metadata.user_id",
                            match=qdrant_models.MatchValue(value=user_id),
                        )]
                    ),
                    limit=100,
                    offset=offset,
                ),
                "scroll indexed file versions",
            )
            for record in records:
                payload = record.payload or {}
                metadata = payload.get("metadata") or {}
                source = metadata.get("source")
                mtime = float(metadata.get("mtime", 0.0))
                raw_mtime_ns = metadata.get("mtime_ns")
                mtime_ns = (
                    int(raw_mtime_ns)
                    if isinstance(raw_mtime_ns, int) and not isinstance(raw_mtime_ns, bool)
                    else None
                )
                if not source:
                    continue

                existing = mtimes.get(source)
                # A nanosecond version is preferred over a legacy float. For
                # duplicate chunks, retain the latest version for the source.
                if (
                    existing is None
                    or (mtime_ns is not None and (existing[0] is None or mtime_ns > existing[0]))
                    or (mtime_ns is None and existing[0] is None and mtime > existing[1])
                ):
                    mtimes[source] = (mtime_ns, mtime)
            if offset is None:
                break
    except ResponseHandlingException:
        raise
    except Exception as e:
        logger.debug(f"Collection likely empty or does not exist yet: {e}")
    logger.debug("Loaded indexed file versions for %d source files.", len(mtimes))
    return mtimes


def _mtime_matches(
    indexed_version: tuple[int | None, float] | None,
    disk_mtime_ns: int,
) -> bool:
    """Compare stored and on-disk file versions without float-rounding churn."""
    if indexed_version is None:
        return False

    indexed_mtime_ns, indexed_mtime = indexed_version
    if indexed_mtime_ns is not None:
        return indexed_mtime_ns == disk_mtime_ns

    # Existing points created before mtime_ns was added get a safe compatibility
    # comparison. New points always use the exact integer branch above.
    return math.isclose(
        indexed_mtime,
        disk_mtime_ns / 1_000_000_000,
        rel_tol=0.0,
        abs_tol=_LEGACY_MTIME_ABS_TOLERANCE_SECONDS,
    )


def _delete_file_chunks(
    client: QdrantClient, collection_name: str, filepath: str, user_id: str
) -> None:
    """Purge all vector points associated with a specific file source path."""
    try:
        _retry_qdrant(
            lambda: client.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="metadata.source",
                                match=qdrant_models.MatchValue(value=filepath),
                            ),
                            qdrant_models.FieldCondition(
                                key="metadata.user_id",
                                match=qdrant_models.MatchValue(value=user_id),
                            ),
                        ]
                    )
                ),
                wait=True,
            ),
            f"delete chunks for {filepath}",
        )
        logger.debug(f"Successfully purged old chunks for file: {filepath}")
    except Exception as e:
        logger.error(f"Failed to delete chunks for {filepath}: {e}")
        raise

# In compound queries involving multiple fields, Qdrant will attempt to use the most restrictive index first.
# https://qdrant.tech/documentation/manage-data/payload 
# exmaple- user_id matches 50,000 chunks
# source matches 200 chunks
# both together match 10 chunks
# Qdrant may begin with the more restrictive source index so search space becomes small and then intersect it with user_id.






def _chunk_id(user_id: str, filepath: str, start_line: int, end_line: int) -> str:
    return str(uuid.uuid5(uuid.UUID(user_id), f"{filepath}|{start_line}|{end_line}"))


def index_single_file(filepath: str, user_id: str) -> None:
    """Targeted incremental hook called instantly by watchdog events."""
    if not os.path.exists(filepath):
        return

    user_id = validate_user_id(user_id)
    filepath = os.path.normcase(os.path.abspath(filepath))
    client, collection_name, embedder, sparse_embedder, retrieval_mode, url, api_key = _get_client_and_settings()
    _ensure_payload_indexes(client, collection_name)
    
    vector_store = _retry_qdrant(
        lambda: QdrantVectorStore.from_existing_collection(
            embedding=embedder,
            sparse_embedding=sparse_embedder,
            retrieval_mode=retrieval_mode,
            url=url,
            api_key=api_key,
            collection_name=collection_name,
        ),
        f"connect to collection {collection_name}",
    )

    stat = os.stat(filepath)
    mtime = stat.st_mtime
    mtime_ns = stat.st_mtime_ns
    logger.info(f"Watchdog triggered processing for: {filepath}")
    _delete_file_chunks(client, collection_name, filepath, user_id)

    try:
        chunks = parse_file(filepath)
    except (SyntaxError, ValueError) as e:
        logger.error(f"Skipping file parse error on {filepath}: {e}")
        return

    if not chunks:
        return

    docs = []
    ids = []
    for chunk in chunks:
        doc_id = _chunk_id(user_id, filepath, chunk.start_line, chunk.end_line)
        ids.append(doc_id)
        docs.append(
            Document(
                page_content=chunk.content,
                metadata={
                    "source": filepath,
                    "user_id": user_id,
                    "name": chunk.name,
                    "type": chunk.type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "mtime": mtime,
                    "mtime_ns": mtime_ns,
                },
            )
        )
    _retry_qdrant(
        lambda: vector_store.add_documents(documents=docs, ids=ids, wait=True),
        f"upsert chunks for {filepath}",
    )
    logger.info(
        "Hybrid upserted %d chunks for file: %s (mtime_ns: %d)",
        len(docs),
        filepath,
        mtime_ns,
    )


def remove_file_from_index(filepath: str, user_id: str) -> None:
    """Targeted deletion hook called instantly when a file is deleted/moved."""
    user_id = validate_user_id(user_id)
    filepath = os.path.normcase(os.path.abspath(filepath))
    client, collection_name, _, _, _, _, _ = _get_client_and_settings()
    _ensure_payload_indexes(client, collection_name)
    logger.info(f"Watchdog removal triggered for: {filepath}")
    _delete_file_chunks(client, collection_name, filepath, user_id)


def index_codebase(repo_path: str, user_id: str) -> QdrantVectorStore:
    """
    Parse source files and store hybrid vectors in Qdrant using incremental 
    delta-sync based on modification timestamps (mtime) and stable deterministic IDs.
    """
    user_id = validate_user_id(user_id)
    client, collection_name, embedder, sparse_embedder, retrieval_mode, url, api_key = _get_client_and_settings()
    _ensure_payload_indexes(client, collection_name)

    existing_collections = [
        collection.name
        for collection in _retry_qdrant(
            client.get_collections, "list collections"
        ).collections
    ]

    vector_store = None
    if collection_name in existing_collections:
        info = _retry_qdrant(
            lambda: client.get_collection(collection_name),
            f"get collection {collection_name}",
        )
        if info.points_count > 0:
            logger.info(
                f"Loaded existing Qdrant hybrid index with "
                f"{info.points_count} chunks. Running delta sync scan..."
            )
            vector_store = _retry_qdrant(
                lambda: QdrantVectorStore.from_existing_collection(
                    embedding=embedder,
                    sparse_embedding=sparse_embedder,
                    retrieval_mode=retrieval_mode,
                    url=url,
                    api_key=api_key,
                    collection_name=collection_name,
                ),
                f"connect to collection {collection_name}",
            )

    current_files = {
        os.path.normcase(os.path.abspath(f)): os.stat(f)
        for f in get_source_files(repo_path)
    }
    indexed_mtimes = _get_indexed_mtimes(client, collection_name, user_id) if vector_store else {}

    # Step 1: Remove stale files no longer on disk
    deleted_files = set(indexed_mtimes) - set(current_files)
    for filepath in deleted_files:
        logger.info(f"Purging deleted file from vector index: {filepath}")
        _delete_file_chunks(client, collection_name, filepath, user_id)

    # Step 2: Evaluate deltas
    added = changed = skipped = 0
    docs_to_add = []
    ids_to_add = []

    logger.info(f"Starting {retrieval_mode.value} indexing of {repo_path}")
    logger.info(f"Found {len(current_files)} source files on disk")

    for filepath, stat in current_files.items():
        mtime = stat.st_mtime
        mtime_ns = stat.st_mtime_ns
        indexed_version = indexed_mtimes.get(filepath)
        if _mtime_matches(indexed_version, mtime_ns):
            skipped += 1
            continue

        if filepath in indexed_mtimes:
            stored_mtime_ns, stored_mtime = indexed_version
            logger.info(
                "Detected file modification. Refreshing: %s "
                "(stored_mtime_ns=%s stored_mtime=%.9f "
                "disk_mtime_ns=%d disk_mtime=%.9f)",
                filepath,
                stored_mtime_ns,
                stored_mtime,
                mtime_ns,
                mtime,
            )
            _delete_file_chunks(client, collection_name, filepath, user_id)
            changed += 1
        else:
            logger.info(f"Detected new file to index: {filepath}")
            added += 1

        try:
            chunks = parse_file(filepath)
        except (SyntaxError, ValueError) as e:
            logger.error(f"Skipping {filepath}: {e}")
            continue

        for chunk in chunks:

            doc_id = _chunk_id(user_id, filepath, chunk.start_line, chunk.end_line)

            ids_to_add.append(doc_id)
            docs_to_add.append(
                Document(
                    page_content=chunk.content,
                    metadata={
                        "source": filepath,
                        "user_id": user_id,
                        "name": chunk.name,
                        "type": chunk.type,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "mtime": mtime,
                        "mtime_ns": mtime_ns,
                    },
                )
            )
            logger.debug(
                f"Prepared {chunk.type} "
                f"'{chunk.name}' from {filepath}"
            )

    # Step 3: Instantiate or update VectorStore
    if vector_store is None:
        if not docs_to_add:
            docs_to_add.append(Document(page_content="Placeholder initialization document", metadata={"user_id": user_id, "source": "init", "name": "init", "type": "placeholder", "start_line": 0, "end_line": 0, "mtime": 0.0, "mtime_ns": 0}))
            ids_to_add.append(_chunk_id(user_id, "init", 0, 0))
        
        logger.info(f"Building fresh Qdrant hybrid collection '{collection_name}' with initial chunks.")
        vector_store = _retry_qdrant(
            lambda: QdrantVectorStore.from_documents(
                documents=docs_to_add,
                ids=ids_to_add,
                embedding=embedder,
                sparse_embedding=sparse_embedder,
                retrieval_mode=retrieval_mode,
                url=url,
                api_key=api_key,
                collection_name=collection_name,
                batch_size=50,
            ),
            f"create and populate collection {collection_name}",
        )
        # The initial collection did not exist when payload-index setup ran.
        # Create the nested payload index now that the collection exists.
        _ensure_payload_indexes(client, collection_name)
    elif docs_to_add:
        logger.info(f"Upserting batch of {len(docs_to_add)} updated/new chunks into Qdrant hybrid store.")
        _retry_qdrant(
            lambda: vector_store.add_documents(
                documents=docs_to_add, ids=ids_to_add, wait=True
            ),
            f"upsert changed chunks into {collection_name}",
        )

    logger.info(
        f"Qdrant {retrieval_mode.value} indexing complete. "
        f"Added: {added}, Changed: {changed}, Deleted: {len(deleted_files)}, Skipped: {skipped}"
    )

    return vector_store


def show_index(vector_store: QdrantVectorStore, user_id: str) -> None:
    """Display all documents, hybrid metadata (including mtime), and vector dimensions stored in Qdrant."""
    from rich.console import Console

    console = Console()
    client = vector_store.client
    collection_name = config["qdrant"]["collection_name"]

    user_id = validate_user_id(user_id)
    results = client.scroll(
        collection_name=collection_name,
        with_payload=True,
        with_vectors=True,
        limit=1000,
        scroll_filter=qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(
                key="metadata.user_id",
                match=qdrant_models.MatchValue(value=user_id),
            )]
        ),
    )

    points = results[0]

    console.print(
        f"\n[bold]Hybrid Qdrant Index — "
        f"{len(points)} chunks[/bold]\n"
    )

    for i, point in enumerate(points):
        payload = point.payload or {}
        emb = point.vector or []
        if isinstance(emb, dict):
            # Handle hybrid dictionaries if vector returns named vectors layout
            emb = list(emb.values())[0] if emb else []

        console.print(
            f"[bold cyan]── Chunk {i + 1} "
            f"──────────────────────────[/bold cyan]"
        )
        metadata = payload.get("metadata") or {}
        console.print(f"  File     : {metadata.get('source', '')}")
        console.print(f"  Name     : {metadata.get('name', '')} ({metadata.get('type', '')})")
        console.print(f"  Lines    : {metadata.get('start_line', '')} - {metadata.get('end_line', '')}")
        console.print(f"  Mod Time : {metadata.get('mtime', 'N/A')}")
        console.print(f"  Embedding: [{', '.join(f'{v:.4f}' for v in emb[:5])}...] ({len(emb)} dims)")
        console.print(
            f"  Code     :\n"
            f"[dim]{payload.get('page_content', '')[:300]}[/dim]\n"
        )





# Delta-Sync Evaluation and Parsing (hybrid_qdrant.py)

# _get_indexed_mtimes() scrolls through Qdrant payloads to gather the stored mtime for each file already in the vector database.

# Deletion Sync: Files present in Qdrant's index but missing from disk are purged via _delete_file_chunks().

# Delta Check: For files remaining on disk:

# If the local mtime matches the indexed mtime, the file is skipped.

# If the file is new or its mtime has changed, parse_file(filepath) is called.

# parse_file() routes code files through Tree-sitter AST parsing (extracting functions, classes, and methods into structured blocks) or falls back to a sliding window for config/text files, returning a list of ParsedChunk dataclass objects.        


##
# For file edits/creations, index_single_file(filepath) is called, which purges old chunks for that file and re-indexes it using the same UUIDv5 delta pattern.

# For deletions, remove_file_from_index(filepath) clears out the corresponding vector points from Qdrant instantly.
