import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


HERE = Path(__file__).parent
DEFAULT_USER_ID = "3b2eb43f-3a0e-41d7-82b7-cde6dc9357d8"


def post(base_url: str, path: str, body: dict) -> dict:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return json.load(response)


def expected_evidence(case: dict) -> list[tuple[str, str]]:
    return [
        (document["path"], symbol.rsplit(".", 1)[-1])
        for document in case["expected_documents"]
        for symbol in document["symbols"]
    ]


def chunk_matches(chunk: dict, evidence: tuple[str, str]) -> bool:
    path, symbol = evidence
    source = chunk["source"].replace("\\", "/").lower()
    return source.endswith(f"/{path.lower()}") and symbol in chunk["content"]


def recall(chunks: list[dict], evidence: list[tuple[str, str]]) -> float | None:
    if not evidence:
        return None
    found = sum(any(chunk_matches(chunk, item) for chunk in chunks) for item in evidence)
    return found / len(evidence)


def precision_at_5(chunks: list[dict], evidence: list[tuple[str, str]]) -> float | None:
    if not evidence:
        return None
    relevant = sum(
        any(chunk_matches(chunk, item) for item in evidence)
        for chunk in chunks[:5]
    )
    return relevant / 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, default=HERE / "golden_dataset.json")
    args = parser.parse_args()

    cases = json.loads(args.dataset.read_text(encoding="utf-8"))["examples"]
    rows = []

    for case in cases:
        session = post(args.base_url, "/sessions", {"user_id": DEFAULT_USER_ID})
        result = post(
            args.base_url,
            "/ask",
            {
                "question": case["question"],
                "user_id": DEFAULT_USER_ID,
                "session_id": session["session_id"],
                "include_evaluation_details": True,
            },
        )
        details = result["evaluation_details"]
        evidence = expected_evidence(case)
        row = {
            "id": case["id"],
            "recall_at_80": recall(details["candidates"], evidence),
            "recall_at_5": recall(details["selected"], evidence),
            "precision_at_5": precision_at_5(details["selected"], evidence),
        }
        rows.append(row)
        print(json.dumps(row))

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "retrieval.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    summary = {
        f"mean_{metric}": sum(values) / len(values)
        for metric in ("recall_at_80", "recall_at_5", "precision_at_5")
        if (values := [row[metric] for row in rows if row[metric] is not None])
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
