import os

from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode,
    FastEmbedSparse,
)
from qdrant_client.http import models as qdrant_models

from ia_claude.config import config
from ia_claude.llm.factory import get_embedder
from ia_claude.observability.logger import get_logger
from ia_claude.user_context import validate_user_id


logger = get_logger(__name__)


RETRIEVAL_MODE_MAP = {
    "dense": RetrievalMode.DENSE,
    "sparse": RetrievalMode.SPARSE,
    "hybrid": RetrievalMode.HYBRID,
}


def retrieve(
    query: str,
    user_id: str,
    k: int = 40,
    fetch_k: int | None = None,
) -> list[dict]:
    """
    Retrieve top-k chunks using dense, sparse, or hybrid mode.

    In hybrid mode, fetch_k candidates are collected independently from the
    dense and sparse indexes before Qdrant fuses them with RRF. The final
    result is still limited to k chunks. Dense and sparse-only modes retain
    their existing behavior.

    The retrieval mode is controlled by config.yaml.
    """

    user_id = validate_user_id(user_id)
    embedder = get_embedder()

    collection_name = config["qdrant"]["collection_name"]

    mode = config["vector_store"].get(
        "retrieval_mode",
        "hybrid",
    ).lower()

    retrieval_mode = RETRIEVAL_MODE_MAP.get(
        mode,
        RetrievalMode.HYBRID,
    )

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise ValueError(
            "QDRANT_URL is not set in the environment."
        )

    sparse_embedder = FastEmbedSparse(
        model_name="Qdrant/bm25"
    )

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedder,
        sparse_embedding=sparse_embedder,
        retrieval_mode=retrieval_mode,
        url=url,
        api_key=api_key,
        collection_name=collection_name,
    )

    logger.info(
        f"Retrieving top {k} chunks "
        f"— mode: {mode} "
        f"— query: {query}"
    )

    user_filter = qdrant_models.Filter(
        must=[qdrant_models.FieldCondition(
            key="metadata.user_id",
            match=qdrant_models.MatchValue(value=user_id),
        )]
    )

    if retrieval_mode == RetrievalMode.HYBRID:
        configured_fetch_k = config["qdrant"].get(
            "hybrid_candidate_k",
            20,
        )
        candidate_k = configured_fetch_k if fetch_k is None else fetch_k

        if not isinstance(candidate_k, int) or isinstance(candidate_k, bool):
            raise ValueError("fetch_k must be an integer.")
        if candidate_k < k:
            raise ValueError("fetch_k must be greater than or equal to k.")

        dense_vector = embedder.embed_query(query)
        sparse_vector = sparse_embedder.embed_query(query)

        logger.info(
            f"Hybrid candidate pools: {candidate_k} dense + "
            f"{candidate_k} sparse; returning top {k} after RRF"
        )

        points = vector_store.client.query_points(
            collection_name=collection_name,
            prefetch=[
                qdrant_models.Prefetch(
                    using=vector_store.vector_name,
                    query=dense_vector,
                    limit=candidate_k,
                    filter=user_filter,
                ),
                qdrant_models.Prefetch(
                    using=vector_store.sparse_vector_name,
                    query=qdrant_models.SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values,
                    ),
                    limit=candidate_k,
                    filter=user_filter,
                ),
            ],
            query=qdrant_models.FusionQuery(
                fusion=qdrant_models.Fusion.RRF
            ),
            limit=k,
            with_payload=True,
            with_vectors=False,
        ).points

        results = []
        for point in points:
            document = vector_store._document_from_point(
                point,
                collection_name,
                vector_store.content_payload_key,
                vector_store.metadata_payload_key,
            )
            results.append((document, point.score))
    else:
        results = vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=user_filter,
        )

    logger.info(
        "Retriever returned %d result(s) before chunk conversion "
        "(requested_k=%d, mode=%s)",
        len(results),
        k,
        mode,
    )
    logger.debug(
        "Retrieved point IDs before chunk conversion: %s",
        [str(doc.metadata.get("_id")) for doc, _ in results],
    )

    chunks = []

    for doc, score in results:

        meta = doc.metadata

        chunks.append(
            {
                "id": meta.get("_id"),
                "content": doc.page_content,
                "source": meta["source"],
                "name": meta["name"],
                "type": meta["type"],
                "start_line": meta["start_line"],
                "end_line": meta["end_line"],
                "distance": score,
            }
        )

        logger.debug(
            f"Retrieved {meta['type']} "
            f"'{meta['name']}' "
            f"from {meta['source']} "
            f"(score: {score:.4f})"
        )

    logger.info(
        f"Retrieved {len(chunks)} chunks"
    )

    return chunks
