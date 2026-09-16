"""Freshness barriers between filesystem mutations and RAG retrieval.

The filesystem watcher is intentionally asynchronous and debounced.  It is
useful for edits made by an IDE, but it cannot be the correctness mechanism
for an agent that writes a file and then immediately searches for it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from ia_claude.config import config
from ia_claude.context.indexers.code_parser import ALL_EXTENSIONS
from ia_claude.context.indexers.factory import get_indexer
from ia_claude.observability.logger import get_logger


logger = get_logger(__name__)

# One process-wide lock prevents a watchdog job and an agent write-through job
# from deleting/upserting the same file concurrently.
_INDEX_OPERATION_LOCK = threading.RLock()
_dirty_paths: set[str] = set()


class IndexFreshnessError(RuntimeError):
    """Raised when retrieval cannot safely claim to use a fresh index."""


def is_indexable(filepath: str) -> bool:
    return Path(filepath).suffix.lower() in ALL_EXTENSIONS


def synchronize_file_change(filepath: str, action: str, user_id: str) -> None:
    """Synchronously apply one successful agent filesystem mutation to RAG.

    This is deliberately a *targeted* upsert/delete, not a full reindex.  The
    function returns only after Qdrant has accepted the mutation, so a later
    tool call in the same agent turn can retrieve the new content.
    """
    path = str(Path(filepath).resolve())
    if not is_indexable(path):
        logger.debug("Freshness skip: non-indexable path=%s action=%s", path, action)
        return
    if action not in {"upsert", "delete"}:
        raise ValueError(f"Unsupported index synchronization action: {action}")

    started = time.perf_counter()
    _dirty_paths.add(path)
    logger.info("Freshness write-through started: action=%s path=%s", action, path)
    try:
        provider = config["vector_store"]["provider"].lower()
        mode = config["vector_store"].get("retrieval_mode", "dense").lower()
        # This project has a true incremental, delete-aware indexer only for
        # Qdrant sparse/hybrid mode.  Failing closed is preferable to claiming
        # fresh retrieval with a provider that cannot apply this mutation.
        if provider != "qdrant" or mode not in {"sparse", "hybrid"}:
            raise IndexFreshnessError(
                "Synchronous single-file freshness is currently implemented "
                "for the Qdrant sparse/hybrid indexer only."
            )

        # The hybrid Qdrant indexer supports dense, sparse, and hybrid modes.
        from ia_claude.context.indexers.hybrid_qdrant import (
            index_single_file,
            remove_file_from_index,
        )

        with _INDEX_OPERATION_LOCK:
            if action == "delete":
                remove_file_from_index(path, user_id)
            else:
                index_single_file(path, user_id)
        _dirty_paths.discard(path)
        logger.info(
            "Freshness write-through complete: action=%s path=%s duration_ms=%.1f",
            action,
            path,
            (time.perf_counter() - started) * 1000,
        )
    except Exception as exc:
        logger.exception(
            "Freshness write-through failed: action=%s path=%s duration_ms=%.1f",
            action,
            path,
            (time.perf_counter() - started) * 1000,
        )
        raise IndexFreshnessError(f"Index update failed for {path}: {exc}") from exc


def reconcile_before_retrieval(repo_path: str, user_id: str) -> None:
    """Run the configured strict retrieval barrier, if enabled.

    A targeted write-through covers agent filesystem tools.  This incremental
    delta scan additionally covers files written through terminal commands or
    by an IDE, including writes whose watchdog event has not fired yet.
    """
    settings = config.get("index_freshness", {})
    strict = settings.get("reconcile_before_retrieval", True)
    if not strict and not _dirty_paths:
        return

    provider = config["vector_store"]["provider"].lower()
    mode = config["vector_store"].get("retrieval_mode", "dense").lower()
    if provider != "qdrant" or mode not in {"sparse", "hybrid"}:
        raise IndexFreshnessError(
            "Strict freshness requires the Qdrant sparse/hybrid incremental "
            "indexer configured for this project."
        )

    started = time.perf_counter()
    logger.info(
        "Freshness retrieval barrier started: repo=%s strict=%s dirty_paths=%d",
        repo_path,
        strict,
        len(_dirty_paths),
    )
    try:
        with _INDEX_OPERATION_LOCK:
            get_indexer()(str(Path(repo_path).resolve()), user_id)
        _dirty_paths.clear()
        logger.info(
            "Freshness retrieval barrier complete: duration_ms=%.1f",
            (time.perf_counter() - started) * 1000,
        )
    except Exception as exc:
        logger.exception(
            "Freshness retrieval barrier failed: repo=%s duration_ms=%.1f",
            repo_path,
            (time.perf_counter() - started) * 1000,
        )
        raise IndexFreshnessError(
            "Codebase index could not be reconciled; retrieval was blocked to "
            "avoid returning stale guidance."
        ) from exc
