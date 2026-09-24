import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from ia_claude.config import config
from ia_claude.user_context import DEFAULT_USER_ID


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete all Qdrant chunks owned by the evaluation user."
    )
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if not args.yes:
        raise SystemExit("Refusing to delete without --yes")

    url = os.getenv("QDRANT_URL")
    if not url:
        raise SystemExit("QDRANT_URL is not set")

    client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
    collection = config["qdrant"]["collection_name"]

    if not client.collection_exists(collection):
        print(f"Collection does not exist: {collection}")
        return

    user_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.user_id",
                match=models.MatchValue(value=DEFAULT_USER_ID),
            )
        ]
    )
    count = client.count(
        collection_name=collection,
        count_filter=user_filter,
        exact=True,
    ).count
    client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(filter=user_filter),
        wait=True,
    )
    print(f"Deleted {count} chunk(s) for user {DEFAULT_USER_ID}")


if __name__ == "__main__":
    main()
