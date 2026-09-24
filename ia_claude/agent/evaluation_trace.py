from contextvars import ContextVar, Token


_trace: ContextVar[dict | None] = ContextVar("evaluation_trace", default=None)


def start_trace() -> Token:
    return _trace.set({"candidates": [], "selected": []})


def record_retrieval(candidates: list[dict], selected: list[dict]) -> None:
    trace = _trace.get()
    if trace is None or trace["candidates"]:
        return

    def keep(chunk: dict) -> dict:
        return {"source": chunk["source"], "content": chunk["content"]}

    trace["candidates"] = [keep(chunk) for chunk in candidates]
    trace["selected"] = [keep(chunk) for chunk in selected]


def finish_trace(token: Token) -> dict:
    trace = _trace.get()
    _trace.reset(token)
    return trace or {"candidates": [], "selected": []}
