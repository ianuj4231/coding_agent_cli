"""Read-only inspector for this project's Redis semantic cache.

Run from the repository root:
    poetry run python inspect_semantic_cache.py

Useful options:
    --limit 50
    --full-response
    --all-versions
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from redis.asyncio import Redis


CACHE_PREFIX = "cache:v3:"
DEFAULT_LIMIT = 20
RESPONSE_PREVIEW_LENGTH = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display Redis semantic-cache records without modifying them."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum entries to display (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--full-response",
        action="store_true",
        help="Print complete cached responses instead of a short preview.",
    )
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="Inspect legacy cache:* entries too, not only the active cache:v3:* entries.",
    )
    return parser.parse_args()


def redis_url() -> str:
    """Load the Redis URL without printing the credential-bearing value."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    url = os.environ.get("REDIS_URL")
    if not url:
        raise RuntimeError(
            "REDIS_URL is not set. Add it to .env before running this inspector."
        )
    return url


def decode(value: bytes | str | None) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def preview(text: str | None, full_response: bool) -> str:
    if not text:
        return ""
    if full_response or len(text) <= RESPONSE_PREVIEW_LENGTH:
        return text
    return f"{text[:RESPONSE_PREVIEW_LENGTH]}... [truncated; use --full-response]"


def format_created_at(value: str | None) -> str:
    if not value:
        return "unknown"
    try:
        return datetime.fromtimestamp(float(value)).astimezone().isoformat(timespec="seconds")
    except ValueError:
        return value


async def inspect_cache(args: argparse.Namespace) -> int:
    patterns = [f"{CACHE_PREFIX}*"]
    if args.all_versions:
        patterns.append("cache:*")

    client = Redis.from_url(redis_url(), decode_responses=False)
    shown_keys: set[bytes] = set()
    displayed = 0
    try:
        await client.ping()
        print(f"Semantic-cache inspector (active prefix: {CACHE_PREFIX})")

        for pattern in patterns:
            async for key in client.scan_iter(match=pattern):
                if key in shown_keys:
                    continue
                shown_keys.add(key)
                if displayed >= args.limit:
                    break

                fields = await client.hgetall(key)
                if not fields:
                    continue
                displayed += 1

                decoded = {decode(name): decode(value) for name, value in fields.items()}
                vector_bytes = len(fields.get(b"query_vector", b""))
                ttl = await client.ttl(key)

                print("\n" + "=" * 80)
                print(f"Redis key:          {decode(key)}")
                print(f"TTL:                {'no expiry' if ttl == -1 else f'{ttl} seconds'}")
                print(f"Domain:             {decoded.get('domain', 'missing')}")
                print(f"Agent namespace:    {decoded.get('agent_namespace', 'legacy/missing')}")
                print(f"LLM model:          {decoded.get('model', 'missing')}")
                print(f"Created at:         {format_created_at(decoded.get('created_at'))}")
                print(f"Question:           {decoded.get('query_text', 'missing')}")
                print(f"Response:           {preview(decoded.get('response'), args.full_response)}")
                print(f"Embedding:          {vector_bytes} bytes (binary float32 vector)")

                if decoded.get("origin_agent_ms") is not None:
                    print(f"Original agent time:{float(decoded['origin_agent_ms']):.1f} ms")
                    print(
                        "Original total time:"
                        f"{float(decoded.get('origin_end_to_end_ms') or 0):.1f} ms"
                    )

            if displayed >= args.limit:
                break

        if displayed == 0:
            print("No matching semantic-cache entries found.")
        elif displayed >= args.limit:
            print(f"\nDisplayed the first {args.limit} entries. Use --limit for more.")
        return 0
    finally:
        await client.aclose()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    try:
        raise SystemExit(asyncio.run(inspect_cache(args)))
    except RuntimeError as error:
        raise SystemExit(f"Inspector error: {error}") from error


if __name__ == "__main__":
    main()
