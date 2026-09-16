# Index freshness: why the agent can safely search after a write

## The problem the watcher alone cannot solve

The filesystem watcher is an **eventual-consistency** mechanism. On Windows it
uses `PollingObserver(timeout=2)`, then waits a further five seconds to debounce
each file event. A file may therefore be absent from the vector index for about
five to seven seconds after a write. If an agent writes a file and calls
`search_codebase` in the next tool call, the watcher may not even have detected
the change yet.

The result in the question is **C**: retrieval does not see the new file and can
give incomplete or contradictory advice. Vector retrieval does not read the
working tree; it only reads already-indexed Qdrant points.

## What changed

There are now two complementary mechanisms:

1. **Write-through indexing.** The agent's `write_file`, `append_file`, and
   `delete_file` tools synchronously index only the changed file before their
   tool result says `... and indexed ...`.
2. **Retrieval barrier.** Immediately before every `search_codebase` search, a
   strict incremental delta sync checks the whole repository. This catches
   changes made outside the filesystem tools, such as an IDE or a terminal
   command, without waiting for the watcher.

The watcher remains useful as a background repair path for human edits while
the agent is idle. It is no longer the path that guarantees same-workflow
freshness.

The incremental scanner reads LangChain's nested Qdrant fields
`metadata.source`, `metadata.mtime_ns`, and `metadata.mtime`. It also filters deletions through
`metadata.source`; using top-level `source`/`mtime` would make every startup
mistake all files for new files.

New points use the exact integer `mtime_ns` version. Older points that contain
only float `mtime` values use a microsecond-scale compatibility tolerance, so a
Qdrant floating-point round trip cannot cause a false modification.

`index_freshness.reconcile_before_retrieval: true` enables the strict barrier.
It is currently designed for the repository's Qdrant sparse/hybrid incremental
indexer. It fails closed: if Qdrant reconciliation fails, the search tool
returns an error rather than silently consulting stale vectors.

## Scenario 1: a brand-new file is added by the agent

Example: the agent calls `write_file("src/payment.py", content)` and then asks
`search_codebase("where is payment validation?")`.

| Microstep | Before this change: watcher only | With the new code |
| --- | --- | --- |
| 1 | `write_file` writes `src/payment.py` to disk. (It was not exposed in the agent's tool list before this change.) | `write_file` writes `src/payment.py` to disk and logs the path and byte count. |
| 2 | Windows polling notices the file on its next scan, which can take up to about 2 seconds. | `synchronize_file_change(..., "upsert")` runs in the same tool call. No polling delay is involved. |
| 3 | The watcher schedules an upsert five seconds later to avoid indexing each keystroke. | The file is parsed immediately into code/text chunks. |
| 4 | Until the timer runs, Qdrant has no points whose `source` is `src/payment.py`. | Existing points for that source are removed (normally none for a new file), then the new chunk vectors and metadata are added to Qdrant. |
| 5 | The next `search_codebase` call can search Qdrant before the file is present. This is the stale-read bug. | The write tool returns `Written and indexed ...` only after the targeted index operation completes. A subsequent agent tool call can retrieve the file. |
| 6 | At roughly 5–7 seconds, the watcher eventually makes the index correct. | `search_codebase` also takes the strict delta-sync barrier before querying. It independently confirms files changed through another path; then it performs vector retrieval. |

The watcher may still receive the event and later execute. That is harmless:
the shared index-operation lock prevents it from deleting/upserting the same
file concurrently with a foreground freshness operation.

## Scenario 2: two existing files are modified by the agent

Example: the agent appends to `src/auth.py`, writes new content to
`src/routes.py`, and then searches for authentication routes.

| Microstep | Before this change: watcher only | With the new code |
| --- | --- | --- |
| 1 | `auth.py` and `routes.py` are changed on disk. | `append_file(auth.py, ...)` finishes its targeted upsert before returning; then `write_file(routes.py, ...)` does the same. |
| 2 | The watcher creates independent five-second timers, one per file. | Each target is normalized to an absolute path and serialized through the shared indexing lock. |
| 3 | At timer expiry, the watcher deletes old chunks for `auth.py`, parses it, then adds replacement vectors. `routes.py` has an independent, similarly delayed job. | For each modified file, old chunks with the matching `source` are deleted, the current disk content is parsed, and new chunks with stable IDs and current `mtime` metadata are upserted. |
| 4 | If search runs before one or both timers, Qdrant can contain two old versions, or one new version and one old version. The answer can contradict the working tree. | By the time each filesystem tool returns success, Qdrant contains the new version for that file. The agent can safely issue its next search. |
| 5 | Eventually both timer jobs complete. | Before the search itself, the delta-sync barrier scans source-file mtimes and skips unchanged files. It only parses/upserts files whose current mtime differs, so it is not a full re-embedding of every file. |

## If files are changed outside the agent filesystem tools

For a terminal command, an IDE save, or a human creating a file, no synchronous
`write_file` hook runs. The next `search_codebase` call handles it as follows:

1. It enters `reconcile_before_retrieval` before constructing the retriever.
2. It scans indexable files and their current modification times.
3. It compares them with the mtimes stored in Qdrant payloads.
4. It removes Qdrant chunks for deleted files, skips unchanged files, and
   re-parses/re-upserts only new or changed files.
5. Only after that completes does vector retrieval run.
6. If reconciliation fails, search returns an explicit freshness error instead
   of potentially stale code.

This is more expensive than an ordinary search because it scans metadata, so it
is a correctness-first setting. If latency later becomes a concern, keep
write-through for agent tools and change the barrier configuration only after
adding an authoritative dirty-path tracker for every other mutation source.

## Important distinction: targeted sync vs. full reindex

Do **not** run a full reindex after every file write. That would re-embed the
whole repository 20 times when an agent creates 20 files.

The implementation instead does this:

```text
agent file mutation -> synchronous single-file delete/upsert -> tool returns
external mutation   -> next retrieval performs incremental mtime delta sync
watcher event        -> background debounced repair/update
```

For 20 agent-created files, there are 20 targeted upserts. That preserves
correctness but can still be slower than a future bulk-write API. A later
optimization can collect a declared batch of paths and index them in one
embedding batch; it must still flush that batch before any retrieval is allowed.
//imp later
## Observable log sequence

Successful agent write:

```text
INFO  Filesystem write succeeded: path=src/payment.py bytes=...
INFO  Freshness write-through started: action=upsert path=.../src/payment.py
INFO  Hybrid upserted ... chunks for file: .../src/payment.py
INFO  Freshness write-through complete: action=upsert path=... duration_ms=...
```

Subsequent search:

```text
INFO  Tool called: search_codebase with query: ...
INFO  Freshness retrieval barrier started: repo=... strict=True dirty_paths=0
INFO  Qdrant hybrid indexing complete. Added: 0, Changed: 0, Deleted: 0, Skipped: ...
INFO  Freshness retrieval barrier complete: duration_ms=...
INFO  Retrieving top 5 chunks ...
```

If any foreground index operation fails, the logs include the action, absolute
path, exception stack trace, and duration. The caller sees an error rather than
a false `indexed` success. If the retrieval barrier fails, retrieval is blocked.
