import os

from dotenv import load_dotenv
from redis import Redis


# All semantic-cache record versions use the cache: prefix. Keep the pattern
# scoped so unrelated data in the same Redis database is never removed.
CACHE_KEY_PATTERN = "cache:*"


def clear_semantic_cache() -> None:
    load_dotenv()

    redis_url = os.getenv("REDIS_URL")

    if not redis_url:
        raise RuntimeError(
            "REDIS_URL was not found in the .env file."
        )

    redis = Redis.from_url(redis_url)

    try:
        print("Redis connection:", redis.ping())
        deleted = 0
        for key in redis.scan_iter(CACHE_KEY_PATTERN):
            deleted += redis.delete(key)

        remaining = sum(1 for _ in redis.scan_iter(CACHE_KEY_PATTERN))

        print(f"Semantic cache entries deleted (all versions): {deleted}")
        print(f"Semantic cache entries remaining (all versions): {remaining}")

    finally:
        redis.close()


if __name__ == "__main__":
    clear_semantic_cache()
