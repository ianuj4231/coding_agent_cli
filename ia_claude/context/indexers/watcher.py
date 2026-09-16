from __future__ import annotations

import platform
import threading
from pathlib import Path

from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from ia_claude.context.indexers.code_parser import ALL_EXTENSIONS
from ia_claude.memory.history import HISTORY_EXPORT_DIR_NAME
from ia_claude.observability.logger import get_logger

logger = get_logger(__name__)

_DEBOUNCE_SECONDS = 5

#watcher to notify the cache on changes in files.

def _get_observer() -> Observer:
    """
    Return the best available watchdog observer for the current OS.
    Windows uses PollingObserver to prevent file-locking and network drive edge cases.
    """
    if platform.system() == "Windows":
        return PollingObserver(timeout=2)
    return Observer()


class _CodebaseEventHandler(FileSystemEventHandler):
    """Translates raw watchdog filesystem events into debounced indexer calls."""

    def __init__(self, user_id: str, on_change=None) -> None:
        self._timers: dict[str, threading.Timer] = {}
        self._user_id = user_id
        self._on_change = on_change

    def _is_indexable(self, path: str) -> bool:
        candidate = Path(path)
        return (
            candidate.suffix.lower() in ALL_EXTENSIONS
            and HISTORY_EXPORT_DIR_NAME not in candidate.parts
        )

    def _schedule(self, action: str, filepath: str) -> None:
        # Check if a timer already exists for this file and remove it
        existing = self._timers.pop(filepath, None)
        if existing:
            existing.cancel()
        # Start a brand new timer from scratch
        timer = threading.Timer(_DEBOUNCE_SECONDS, self._run, args=(action, filepath))
        timer.daemon = True
        timer.start()
        self._timers[filepath] = timer

    def _run(self, action: str, filepath: str) -> None:
        self._timers.pop(filepath, None)
        from ia_claude.context.indexers.freshness import _INDEX_OPERATION_LOCK
        from ia_claude.context.indexers.hybrid_qdrant import (
            index_single_file,
            remove_file_from_index,
        )
        try:
            with _INDEX_OPERATION_LOCK:
                # Debounced events describe what happened when they were
                # queued, not necessarily the filesystem's current state. A
                # short-lived file can be created and deleted before its
                # pending upsert runs. In that case, converge the index on the
                # current state instead of trying to index a missing file.
                if action == "delete" or not Path(filepath).is_file():
                    if action != "delete":
                        logger.debug(
                            "Watchdog converted stale upsert to delete: path=%s",
                            filepath,
                        )
                    remove_file_from_index(filepath, self._user_id)
                else:
                    try:
                        index_single_file(filepath, self._user_id)
                    except FileNotFoundError:
                        # The file may disappear after is_file() but before or
                        # during parsing. Removal is idempotent and guarantees
                        # that no stale chunks remain in the index.
                        logger.debug(
                            "File disappeared during watchdog upsert; removing "
                            "it from the index: path=%s",
                            filepath,
                        )
                        remove_file_from_index(filepath, self._user_id)
        except Exception:
            logger.exception("Watchdog index update failed: action=%s path=%s", action, filepath)

        if self._on_change is not None:
            self._on_change()

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_indexable(event.src_path):
            self._schedule("upsert", event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_indexable(event.src_path):
            self._schedule("upsert", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_indexable(event.src_path):
            self._schedule("delete", event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            if self._is_indexable(event.src_path):
                self._schedule("delete", event.src_path)
            if self._is_indexable(event.dest_path):
                self._schedule("upsert", event.dest_path)


def start_watcher(repo_path: str, user_id: str, on_change=None) -> Observer:
    handler = _CodebaseEventHandler(user_id=user_id, on_change=on_change)
    observer = _get_observer()
    observer.schedule(handler, repo_path, recursive=True)
    observer.daemon = True
    observer.start()
    logger.info(f"Watchdog started on {repo_path} (backend: {type(observer).__name__})")
    return observer


def stop_watcher(observer: Observer) -> None:
    observer.stop()
    observer.join()
    logger.info("Watchdog stopped")
