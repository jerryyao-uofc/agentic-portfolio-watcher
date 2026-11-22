"""LangGraph run: context, four specialists in parallel, then the writer.

Short-term memory is a LangGraph InMemorySaver thread named {symbol}-{run_id}.
The previous written memo, if any, is loaded from the gitignored memory file.
"""

from __future__ import annotations

import json
import operator
import threading
import uuid
from typing import Annotated

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from research import agents, memory
from research.flags import select_flags
from research.tools import bind_run, end_run, start_run

TRACE_ORDER = ["context", "financial", "news", "competitive", "risk", "writer"]
STEP_LABELS = {
    "context": "Reading the position",
    "financial": "Financial statements",
    "news": "News and market",
    "competitive": "Competitive intelligence",
    "risk": "Risk",
    "writer": "Writing the suggestion",
}
_checkpointer = InMemorySaver()
_compiled = None
_progress_lock = threading.Lock()
_progress = {"active": False, "symbol": "", "steps": []}


def progress() -> dict:
    with _progress_lock:
        return json.loads(json.dumps(_progress))


def _set_progress(symbol: str) -> None:
    with _progress_lock:
        _progress["active"] = True
        _progress["symbol"] = symbol
        _progress["steps"] = [
            {"id": name, "label": STEP_LABELS[name], "state": "waiting"} for name in TRACE_ORDER
        ]


def _mark(node: str, state: str) -> None:
    with _progress_lock:
        for step in _progress["steps"]:
            if step["id"] == node:
                step["state"] = state
                break


def _clear_progress() -> None:
    with _progress_lock:
        _progress["active"] = False
        _progress["symbol"] = ""
        _progress["steps"] = []


class ResearchState(TypedDict):
    symbol: str
    payload: dict
    prior_memo: str
    run_id: str
    context: dict
    financial_memo: str
    news_memo: str
    competitive_memo: str
    risk_memo: str
    writer_memo: str
    chart_specs: list
    trace: Annotated[list, operator.add]


def build_context(payload: dict, symbol: str, prior_memo: str) -> dict:
    holdings = {item.get("symbol"): item for item in payload.get("holdings") or []}
    item = holdings.get(symbol) or {}
    reasons = []
    for flag in payload.get("flags") or []:
        if flag.get("symbol") == symbol:
            reasons = list(flag.get("reasons") or [])
    look_weight = None
    for row in (payload.get("lookthrough") or {}).get("rows") or []:
        if row.get("symbol") == symbol:
            look_weight = row.get("weight")
    return {
        "symbol": symbol,
        "name": item.get("name") or symbol,
        "shares": item.get("shares"),
        "price": item.get("price"),
        "weight": item.get("weight"),
        "lookthroughWeight": look_weight if look_weight is not None else item.get("weight"),
        "change3m": item.get("change3m"),
        "industry": item.get("industry"),
        "reasons": reasons,
        "priorMemo": prior_memo or None,
    }


def _context_node(state: ResearchState) -> dict:
    bind_run(state["run_id"])
    _mark("context", "working")
    context = build_context(state["payload"], state["symbol"], state.get("prior_memo") or "")
    _mark("context", "done")
    return {"context": context, "trace": [{"node": "context", "tools": []}]}


def _financial_node(state: ResearchState) -> dict:
    bind_run(state["run_id"])
    _mark("financial", "working")
    memo, tools = agents.run_financial(agents.build_chat("financial"), state["context"])
    _mark("financial", "done")
    return {"financial_memo": memo, "trace": [{"node": "financial", "tools": tools}]}


def _news_node(state: ResearchState) -> dict:
    bind_run(state["run_id"])
    _mark("news", "working")
    memo, tools = agents.run_news(agents.build_chat("news"), state["context"])
    _mark("news", "done")
    return {"news_memo": memo, "trace": [{"node": "news", "tools": tools}]}


def _competitive_node(state: ResearchState) -> dict:
    bind_run(state["run_id"])
    _mark("competitive", "working")
    memo, tools = agents.run_competitive(agents.build_chat("competitive"), state["context"])
    _mark("competitive", "done")
    return {"competitive_memo": memo, "trace": [{"node": "competitive", "tools": tools}]}


def _risk_node(state: ResearchState) -> dict:
    bind_run(state["run_id"])
    _mark("risk", "working")
    memo, tools = agents.run_risk(agents.build_chat("risk"), state["context"])
    _mark("risk", "done")
    return {"risk_memo": memo, "trace": [{"node": "risk", "tools": tools}]}


def _writer_node(state: ResearchState) -> dict:
    bind_run(state["run_id"])
    _mark("writer", "working")
    memos = {
        "financial": state.get("financial_memo") or "",
        "news": state.get("news_memo") or "",
        "competitive": state.get("competitive_memo") or "",
        "risk": state.get("risk_memo") or "",
    }
    memo, tools = agents.run_writer(
        agents.build_chat("writer"),
        state["context"],
        memos,
        state.get("prior_memo") or "",
    )
    from research.tools import current_run

    run = current_run()
    charts = list(run.charts) if run else []
    _mark("writer", "done")
    return {
        "writer_memo": memo,
        "chart_specs": charts,
        "trace": [{"node": "writer", "tools": tools}],
    }


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("context", _context_node)
    graph.add_node("financial", _financial_node)
    graph.add_node("news", _news_node)
    graph.add_node("competitive", _competitive_node)
    graph.add_node("risk", _risk_node)
    graph.add_node("writer", _writer_node)
    graph.add_edge(START, "context")
    for name in ("financial", "news", "competitive", "risk"):
        graph.add_edge("context", name)
    graph.add_edge(["financial", "news", "competitive", "risk"], "writer")
    graph.add_edge("writer", END)
    return graph


def compiled():
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile(checkpointer=_checkpointer)
    return _compiled


def _ordered(trace: list[dict]) -> list[dict]:
    rank = {name: index for index, name in enumerate(TRACE_ORDER)}
    return sorted(trace or [], key=lambda row: rank.get(row.get("node"), 99))


def run_symbol(payload: dict, symbol: str) -> dict:
    """One symbol. The checkpointer thread is discarded with the process."""
    symbol = symbol.upper()
    prior = memory.load(symbol)
    run_id = uuid.uuid4().hex[:12]
    thread_id = f"{symbol}-{run_id}"
    reasons = []
    for flag in payload.get("flags") or []:
        if flag.get("symbol") == symbol:
            reasons = list(flag.get("reasons") or [])
    start_run(symbol, payload, run_id)
    _set_progress(symbol)
    try:
        final = compiled().invoke(
            {
                "symbol": symbol,
                "payload": payload,
                "prior_memo": prior,
                "run_id": run_id,
                "context": {},
                "financial_memo": "",
                "news_memo": "",
                "competitive_memo": "",
                "risk_memo": "",
                "writer_memo": "",
                "chart_specs": [],
                "trace": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )
    finally:
        end_run(run_id)
        _clear_progress()
    writer = final.get("writer_memo") or ""
    memory.save(symbol, writer, run_id)
    return {
        "symbol": symbol,
        "reasons": reasons,
        "memos": {
            "financial": final.get("financial_memo") or "",
            "news": final.get("news_memo") or "",
            "competitive": final.get("competitive_memo") or "",
            "risk": final.get("risk_memo") or "",
        },
        "writer": writer,
        "trace": _ordered(final.get("trace") or []),
        "charts": final.get("chart_specs") or [],
        "threadId": thread_id,
    }


def _direct(payload: dict, symbol: str) -> bool:
    for item in payload.get("holdings") or []:
        if item.get("symbol") == symbol and item.get("type") != "crypto" and not item.get("isEtf"):
            return True
    return False


def run_request(payload: dict, body: dict) -> dict:
    flags = payload.get("flags")
    if not flags:
        flags = select_flags(payload)
        payload = {**payload, "flags": flags}
    if body.get("shortlist"):
        symbols = [row["symbol"] for row in flags]
    else:
        symbols = [str(symbol).upper() for symbol in (body.get("symbols") or [])]
    if not symbols:
        return {"runs": [], "note": "No direct stocks were selected."}
    if len(symbols) > 3:
        raise ValueError("At most 3 names per research run.")
    missing = [symbol for symbol in symbols if not _direct(payload, symbol)]
    if missing:
        raise ValueError(
            "Research covers direct stocks only. Not in that list: " + ", ".join(missing) + "."
        )
    return {"runs": [run_symbol(payload, symbol) for symbol in symbols]}


def handle_request(raw: bytes) -> tuple[int, dict]:
    if agents.MODEL_OVERRIDE is None and not (os_key()):
        return 400, {"error": agents.MISSING_KEY}
    try:
        body = json.loads(raw.decode() or "{}")
    except json.JSONDecodeError:
        return 400, {"error": "Research expects JSON, for example {\"symbols\": [\"NVDA\"]}."}
    if not isinstance(body, dict):
        return 400, {"error": "Research expects a JSON object."}
    try:
        import app

        payload = app.build_payload(force=False)
        result = run_request(payload, body)
    except agents.MissingDeepSeekKey as error:
        return 400, {"error": str(error)}
    except ValueError as error:
        return 400, {"error": str(error)}
    except Exception as error:
        return 500, {"error": str(error)}
    return 200, result


def os_key() -> str:
    import os

    return (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
