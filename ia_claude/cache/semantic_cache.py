from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from redis.asyncio import Redis
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from redisvl.redis.utils import array_to_buffer

from ia_claude.config import config
from ia_claude.llm.factory import get_embedder
from ia_claude.observability.logger import get_logger

logger = get_logger(__name__)

VECTOR_DTYPE = "float32"
CACHE_INDEX_NAME = "semantic_cache_v3"
CACHE_KEY_PREFIX = "cache:v3:"


@dataclass(frozen=True)
class CacheTiming:
    """Timing baseline recorded when an answer is first cached."""

    agent_ms: float
    end_to_end_ms: float


@dataclass(frozen=True)
class CacheHit:
    """A cached answer together with timing from the miss it avoids."""

    response: str
    timing: CacheTiming | None = None


def _cache_doc_id(
    query: str, domain: str, agent_namespace: str, model: str
) -> str:
    """Return an ID unique to a repository, agent, model, and question."""
    identity = "\x1f".join((domain, agent_namespace, model, query))
    return hashlib.sha256(identity.encode()).hexdigest()[:32]


def _build_index_schema(dims: int) -> dict:
    return {
        # A versioned index/prefix leaves legacy cache entries intact. The
        # previous schemas lacked explicit timing for hit/miss comparisons.
        "index": {"name": CACHE_INDEX_NAME, "prefix": CACHE_KEY_PREFIX},
        "fields": [
            {
                "name": "query_vector",
                "type": "vector",
                "attrs": {
                    "dims": dims,
                    "algorithm": "HNSW",
                    "distance_metric": "cosine",
                    "datatype": VECTOR_DTYPE,
                },
            },
            {"name": "response", "type": "text"},
            {"name": "query_text", "type": "text"},
            {"name": "domain", "type": "tag"},
            {"name": "agent_namespace", "type": "tag"},
            {"name": "model", "type": "tag"},
            {"name": "created_at", "type": "numeric"},
            {"name": "origin_agent_ms", "type": "numeric"},
            {"name": "origin_end_to_end_ms", "type": "numeric"},
        ],
    }


class SemanticCache:
    """Async Redis-backed semantic cache for /ask responses.

    embeddings come from the app's own get_embedder() (so the cache follows
    whichever provider config.yaml selects) and lookups filter on both
    repository domain, agent namespace, and LLM model.  This prevents one
    repository, agent, or provider/model from serving another's cached answer.
    """

    def __init__(
        self, redis_url: str, threshold: float, dims: int, agent_namespace: str
    ):
        self.client = Redis.from_url(redis_url)
        self.embedder = get_embedder()
        self.threshold = threshold
        self.agent_namespace = agent_namespace
        self.index = AsyncSearchIndex.from_dict(
            _build_index_schema(dims), redis_client=self.client
        )

    async def setup(self) -> None:
        # overwrite=False: reuse the index if it already exists from a
        # previous run instead of wiping cached entries on every startup.
        await self.index.create(overwrite=False)

    async def _embed(self, text: str) -> list[float]:
        return await self.embedder.aembed_query(text)

    async def _top_match(self, query: str, domain: str, model: str) -> dict | None:
        vector = await self._embed(query)
        q = VectorQuery(
            vector=vector,
            vector_field_name="query_vector",
            return_fields=[
                "response",
                "query_text",
                "vector_distance",
                "origin_agent_ms",
                "origin_end_to_end_ms",
            ],
            num_results=1,
            dtype=VECTOR_DTYPE,
        )
        q.set_filter(
            (Tag("domain") == domain)
            & (Tag("agent_namespace") == self.agent_namespace)
            & (Tag("model") == model)
        )
        results = await self.index.query(q)
        return results[0] if results else None

    @staticmethod
    def _optional_float(value) -> float | None:
        return float(value) if value is not None else None

    async def get_with_metadata(
        self, query: str, domain: str, model: str
    ) -> CacheHit | None:
        """Look up a cached answer and its original miss timing."""
        hit = await self._top_match(query, domain=domain, model=model)
        if hit is None:
            return None
        similarity = 1 - float(hit["vector_distance"])
        if similarity >= self.threshold:
            agent_ms = self._optional_float(hit.get("origin_agent_ms"))
            timing = None
            if agent_ms is not None:
                timing = CacheTiming(
                    agent_ms=agent_ms,
                    end_to_end_ms=self._optional_float(
                        hit.get("origin_end_to_end_ms")
                    ) or agent_ms,
                )
            return CacheHit(response=hit["response"], timing=timing)
        # A candidate exists but isn't close enough - treat as a miss rather
        # than returning a possibly-wrong cached answer.
        return None

    async def get(self, query: str, domain: str, model: str) -> str | None:
        """Look up a cached response. Returns None on miss."""
        hit = await self.get_with_metadata(query, domain, model)
        return hit.response if hit is not None else None

    async def put(
        self,
        query: str,
        response: str,
        domain: str,
        model: str,
        ttl: int,
        timing: CacheTiming | None = None,
    ) -> None:
        """Store a (query, response) pair."""
        vector = await self._embed(query)
        # Include all fields used by retrieval filtering. Otherwise an equal
        # question from another repository, agent, or model replaces it.
        doc_id = _cache_doc_id(query, domain, self.agent_namespace, model)
        redis_key = self.index.key(doc_id)
        entry = {
            "query_vector": array_to_buffer(vector, VECTOR_DTYPE),
            "response": response,
            "query_text": query,
            "domain": domain,
            "agent_namespace": self.agent_namespace,
            "model": model,
            "created_at": time.time(),
        }
        if timing is not None:
            entry.update(
                {
                    "origin_agent_ms": timing.agent_ms,
                    "origin_end_to_end_ms": timing.end_to_end_ms,
                }
            )
        await self.index.load([entry], keys=[redis_key], ttl=ttl)

    async def invalidate_domain(self, domain: str) -> None:
        """Delete all cache entries for a domain (e.g. after a codebase change)."""
        async for key in self.client.scan_iter(f"{CACHE_KEY_PREFIX}*"):
            entry = await self.client.hget(key, "domain")
            if entry and entry.decode() == domain:
                await self.client.delete(key)


def get_repo_domain(repo_path: str) -> str:
    """Stable cache domain derived from the repo path, so cached entries
    persist across restarts for the same repo without colliding with
    entries from a different repo indexed by the same tool."""
    return hashlib.sha256(repo_path.encode()).hexdigest()[:16]


async def build_semantic_cache() -> SemanticCache | None:
    """Build and initialize the semantic cache per config.yaml.

    Returns None (and logs why) if caching is disabled or Redis is
    unreachable, so the caller can run without caching instead of crashing.
    """
    cache_config = config.get("semantic_cache", {})
    if not cache_config.get("enabled", False):
        logger.info("Semantic cache: disabled (semantic_cache.enabled is false in config.yaml)")
        return None

    try:
        dims = config["embeddings"]["dims"]
        cache = SemanticCache(
            redis_url=cache_config["redis_url"],
            threshold=cache_config.get("threshold", 0.85),
            dims=dims,
            agent_namespace=cache_config.get("agent_namespace", "coding_agent"),
        )
        await cache.setup()
        logger.info(
            "Semantic cache: enabled (threshold=%s, dims=%s, agent_namespace=%s)",
            cache.threshold,
            dims,
            cache.agent_namespace,
        )
        return cache
    except Exception as error:
        logger.warning(f"Semantic cache: disabled (init failed: {error})")
        return None
