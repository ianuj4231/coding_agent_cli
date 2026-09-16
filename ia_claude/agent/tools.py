from pathlib import Path

from langchain.tools import tool

from ia_claude.context.indexers.freshness import (
    IndexFreshnessError,
    reconcile_before_retrieval,
)
from ia_claude.context.rerankers.cross_encoder import rerank_chunks
from ia_claude.context.retrievers.factory import get_retriever
from ia_claude.observability.logger import get_logger
from ia_claude.user_context import validate_user_id


logger = get_logger(__name__)


def create_search_codebase_tool(user_id: str):
    """Create a code-search tool bound to one validated startup user ID."""
    user_id = validate_user_id(user_id)

    @tool
    def search_codebase(query: str) -> str:
        """
        Search the codebase for relevant classes, functions, or logic.

        Args:
            query: Required natural-language or code search query. The tool-call
                argument must be named exactly ``query``, not ``question``.
        """
        logger.info("Tool called: search_codebase with query: %s", query)

        try:
            reconcile_before_retrieval(str(Path.cwd()), user_id)
        except IndexFreshnessError as exc:
            logger.error("Blocked stale retrieval: query=%r error=%s", query, exc)
            return f"Error: retrieval blocked because the code index is not fresh: {exc}"

        retrieve = get_retriever()
        chunks = retrieve(query, user_id=user_id, k=40)
        retrieved_count = len(chunks)
        try:
            chunks = rerank_chunks(query, chunks)
        except Exception:
            logger.exception(
                "Cross-encoder reranking failed; falling back to RRF top 5"
            )
            chunks = chunks[:5]

        logger.info(
            "Selected %d of %d retrieved chunks for LLM context",
            len(chunks),
            retrieved_count,
        )

        if not chunks:
            return "No relevant code found."

        results = []
        for chunk in chunks:
            results.append(
                f"ID: {chunk['id']}\n"
                f"File: {chunk['source']} "
                f"(lines {chunk['start_line']}-{chunk['end_line']})\n"
                f"Reranker score: {chunk.get('reranker_score', 'n/a')}\n"
                f"Type: {chunk['type']} — {chunk['name']}\n"
                f"Code:\n{chunk['content']}\n"
            )

        return "\n---\n".join(results)

    return search_codebase
