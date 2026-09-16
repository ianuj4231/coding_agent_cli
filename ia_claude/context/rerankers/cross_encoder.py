from __future__ import annotations

from functools import lru_cache
from typing import Any

from ia_claude.config import config
from ia_claude.observability.logger import get_logger


logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_model(model_name: str, max_length: int):
    """Load the cross-encoder once and reuse it for later queries."""
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder reranker: %s", model_name)
    return CrossEncoder(model_name, max_length=max_length)


def _positive_int(settings: dict[str, Any], key: str, default: int) -> int:
    value = settings.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"reranker.{key} must be a positive integer.")
    return value


def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Rerank retrieved chunks and return only the configured top results."""
    settings = config.get("reranker", {})
    top_n = _positive_int(settings, "top_n", 5)

    if not chunks:
        return []

    if not settings.get("enabled", False):
        logger.info(
            "Reranker disabled; selecting RRF top %d of %d chunks",
            min(top_n, len(chunks)),
            len(chunks),
        )
        return chunks[:top_n]

    model_name = settings.get(
        "model",
        "cross-encoder/ms-marco-MiniLM-L6-v2",
    )
    batch_size = _positive_int(settings, "batch_size", 16)
    max_length = _positive_int(settings, "max_length", 512)
    model = _get_model(model_name, max_length)

    logger.info(
        "Cross-encoder reranking %d chunks (model=%s, top_n=%d)",
        len(chunks),
        model_name,
        top_n,
    )
    pairs = [(query, chunk["content"]) for chunk in chunks]
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    if len(scores) != len(chunks):
        raise RuntimeError(
            "Cross-encoder returned a different number of scores than chunks."
        )

    reranked = []
    for chunk, score in zip(chunks, scores):
        scored_chunk = dict(chunk)
        scored_chunk["retrieval_score"] = chunk.get("distance")
        scored_chunk["reranker_score"] = float(score)
        reranked.append(scored_chunk)

    reranked.sort(key=lambda chunk: chunk["reranker_score"], reverse=True)
    selected = reranked[:top_n]

    logger.info(
        "Cross-encoder selected top %d of %d reranked chunks",
        len(selected),
        len(reranked),
    )
    logger.debug(
        "Selected reranked point IDs: %s",
        [str(chunk.get("id")) for chunk in selected],
    )
    return selected

