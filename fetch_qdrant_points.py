from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from qdrant_client import QdrantClient

from ia_claude.config import config


DEFAULT_POINT_IDS =  ['45e953af-49eb-5dc5-a2c2-f25b82b60197', 'd7c28104-a04b-5a54-b725-3630bb43aada', '7b516363-a31f-5ef8-ac06-2b19e68f3328', '2d199999-ad73-54ea-9e62-ccfce727335a', 'bdd360b6-30d3-552a-a21a-090b9665a448']

def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fetch Qdrant points by ID and save their payloads as JSON."
    )
    parser.add_argument(
        "ids",
        nargs="*",
        default=DEFAULT_POINT_IDS,
        help="Qdrant point IDs. Uses the latest five reranked IDs by default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "evaluation_results" / "reranked_points.json",
    )
    args = parser.parse_args()

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        raise RuntimeError("QDRANT_URL is not set in .env or the environment.")

    collection_name = config["qdrant"]["collection_name"]
    client = QdrantClient(url=url, api_key=api_key)
    points = client.retrieve(
        collection_name=collection_name,
        ids=args.ids,
        with_payload=True,
        with_vectors=False,
    )

    points_by_id = {str(point.id): point for point in points}
    ordered_points = []
    missing_ids = []
    for requested_rank, point_id in enumerate(args.ids, start=1):
        point = points_by_id.get(str(point_id))
        if point is None:
            missing_ids.append(str(point_id))
            continue
        ordered_points.append(
            {
                "requested_rank": requested_rank,
                "id": str(point.id),
                "payload": point.payload,
            }
        )

    report = {
        "collection_name": collection_name,
        "requested_count": len(args.ids),
        "found_count": len(ordered_points),
        "missing_ids": missing_ids,
        "points": ordered_points,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"Fetched {len(ordered_points)} of {len(args.ids)} points.")
    if missing_ids:
        print(f"Missing IDs: {', '.join(missing_ids)}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
