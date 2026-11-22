"""Tools wrapped around the portfolio snapshot, plus a capped web search.

Search and charts are limited in this module. Prompts are not the limit.
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from langchain_core.tools import tool

SEARCH_CAP = 2
CHART_CAP = 2
RANGES = {"3m": 92, "1y": 365, "5y": 365 * 5}

# Tests replace these. None means use the live source.
PEER_LOADER = None
HEADLINE_LOADER = None
SEARCH_TRANSPORT = None

_run = contextvars.ContextVar("research_run", default=None)
_lock = threading.Lock()
_runs: dict[str, "ResearchRun"] = {}


class ResearchRun:
    def __init__(self, symbol: str, payload: dict, run_id: str):
        self.symbol = symbol
        self.payload = payload
        self.run_id = run_id
        self.search_counts: dict[str, int] = {}
        self.chart_counts: dict[str, int] = {}
        self.charts: list[dict] = []


def start_run(symbol: str, payload: dict, run_id: str = "local") -> ResearchRun:
    run = ResearchRun(symbol, payload, run_id)
    _runs[run_id] = run
    _run.set(run)
    return run


def bind_run(run_id: str) -> ResearchRun | None:
    run = _runs.get(run_id)
    _run.set(run)
    return run


def end_run(run_id: str | None = None) -> None:
    current = _run.get()
    key = run_id or (current.run_id if current else None)
    if key:
        _runs.pop(key, None)
    _run.set(None)


def current_run() -> ResearchRun | None:
    return _run.get()


def _books() -> dict:
    import app

    return app.SYMBOL_BOOKS


def _portfolio_symbols(payload: dict) -> set[str]:
    return {item.get("symbol") for item in (payload or {}).get("holdings") or [] if item.get("symbol")}


def _direct_stocks(payload: dict) -> list[dict]:
    rows = [
        item
        for item in (payload or {}).get("holdings") or []
        if item.get("type") != "crypto" and not item.get("isEtf")
    ]
    rows.sort(key=lambda item: (-float(item.get("value") or 0), str(item.get("symbol") or "")))
    return rows


@tool
def get_fundamentals(symbol: str) -> dict:
    """Trailing PE, profit margin, revenue growth, and debt-to-equity when Yahoo had them."""
    book = _books().get(symbol) or {}
    figures = {}
    for key, value in (book.get("fundamentals") or {}).items():
        if value is not None:
            figures[key] = value
    return {"symbol": symbol, **figures}


@tool
def get_headlines(symbol: str) -> dict:
    """Recent Yahoo Finance headlines for one symbol. An empty list is not a claim that nothing happened."""
    if HEADLINE_LOADER is not None:
        rows = HEADLINE_LOADER(symbol)
    else:
        import app

        book = _books().get(symbol) or {}
        rows = app._headlines([book.get("yahoo") or symbol])
    return {"symbol": symbol, "headlines": list(rows or []), "source": "yahoo-rss"}


@tool
def get_position_risk(symbol: str) -> dict:
    """Direct weight, look-through weight, rank among direct stocks, value, and the dollar loss if the price falls 20%."""
    run = current_run()
    payload = run.payload if run else {}
    holdings = {item.get("symbol"): item for item in payload.get("holdings") or []}
    item = holdings.get(symbol)
    if not item:
        return {"symbol": symbol, "error": f"{symbol} is not in the loaded portfolio."}
    direct = _direct_stocks(payload)
    rank = next((index for index, row in enumerate(direct, start=1) if row.get("symbol") == symbol), None)
    look_weight = None
    for row in (payload.get("lookthrough") or {}).get("rows") or []:
        if row.get("symbol") == symbol:
            look_weight = row.get("weight")
            break
    value = float(item.get("value") or 0)
    return {
        "symbol": symbol,
        "directWeight": item.get("weight"),
        "lookthroughWeight": look_weight if look_weight is not None else item.get("weight"),
        "rankAmongDirectStocks": rank,
        "value": value,
        "dollarLossAtMinus20": value * 0.20,
    }


def _yahoo_peers(symbol: str) -> list[str]:
    import app

    url = "https://query1.finance.yahoo.com/v6/finance/recommendationsbysymbol/" + urllib.parse.quote(symbol)
    try:
        payload = app.YAHOO.get_json(url)
    except Exception:
        return []
    found = []
    finance = (payload or {}).get("finance") or {}
    results = finance.get("result") or payload.get("result") or []
    if isinstance(results, dict):
        results = [results]
    for block in results:
        if not isinstance(block, dict):
            continue
        symbols = block.get("recommendedSymbols") or block.get("symbols") or []
        for row in symbols:
            ticker = row.get("symbol") if isinstance(row, dict) else row
            if ticker:
                found.append(str(ticker).upper())
    unique = []
    for ticker in found:
        if ticker not in unique and ticker != symbol.upper():
            unique.append(ticker)
    return unique[:8]


def _mark_peers(symbol: str, peers: list[str], source: str, extra: dict | None = None) -> dict:
    run = current_run()
    held = _portfolio_symbols(run.payload if run else {})
    body = {
        "symbol": symbol,
        "source": source,
        "peers": [{"symbol": ticker, "inPortfolio": ticker in held} for ticker in peers],
    }
    if extra:
        body.update(extra)
    return body


@tool
def get_peers(symbol: str) -> dict:
    """Peer companies. Uses Yahoo related symbols, or one web search when that list is empty."""
    if PEER_LOADER is not None:
        peers = list(PEER_LOADER(symbol) or [])
        source = "yahoo"
    else:
        peers = _yahoo_peers(symbol)
        source = "yahoo"
    if peers:
        return _mark_peers(symbol, peers, source)
    found = search_web.invoke({"query": f"{symbol} publicly traded competitors"})
    return _mark_peers(
        symbol,
        [],
        "search",
        {"search": found, "note": "Yahoo returned no related symbols, so this list came from search."},
    )


def _reserve(kind: str, symbol: str, cap: int) -> str | None:
    run = current_run()
    if run is None:
        return None
    with _lock:
        counts = run.search_counts if kind == "search" else run.chart_counts
        used = counts.get(symbol, 0)
        if used >= cap:
            return (
                f"{'search_web' if kind == 'search' else 'chart_series'} is limited to "
                f"{cap} calls for {symbol} in this run."
            )
        counts[symbol] = used + 1
    return None


def _tavily(query: str) -> dict:
    key = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("missing key")
    body = json.dumps({"api_key": key, "query": query, "search_depth": "basic"}).encode()
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "portfolio-tracker"},
        method="POST",
    )
    import app

    with urllib.request.urlopen(request, timeout=20, context=app.ssl_context()) as response:
        payload = json.loads(response.read().decode())
    results = []
    for row in payload.get("results") or []:
        results.append(
            {
                "title": row.get("title") or "",
                "url": row.get("url") or "",
                "snippet": row.get("content") or "",
            }
        )
    return {"query": query, "source": "tavily", "results": results}


def _duckduckgo(query: str) -> dict:
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
    )
    request = urllib.request.Request(url, headers={"User-Agent": "portfolio-tracker"})
    import app

    with urllib.request.urlopen(request, timeout=20, context=app.ssl_context()) as response:
        payload = json.loads(response.read().decode())
    results = []
    abstract = (payload.get("AbstractText") or "").strip()
    if abstract:
        results.append(
            {
                "title": payload.get("Heading") or query,
                "url": payload.get("AbstractURL") or "",
                "snippet": abstract,
            }
        )
    for topic in payload.get("RelatedTopics") or []:
        if not isinstance(topic, dict):
            continue
        text = (topic.get("Text") or "").strip()
        if not text:
            continue
        results.append({"title": text[:120], "url": topic.get("FirstURL") or "", "snippet": text})
    return {"query": query, "source": "instant answer", "results": results}


def _quota_error(error: Exception) -> bool:
    code = getattr(error, "code", None)
    if code in {402, 429, 432, 433}:
        return True
    text = str(error).lower()
    return any(word in text for word in ("quota", "credit", "rate limit", "429", "432"))


def _live_search(query: str) -> dict:
    key = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if not key:
        return _duckduckgo(query)
    try:
        return _tavily(query)
    except Exception as error:
        if _quota_error(error) or not key:
            fallback = _duckduckgo(query)
            fallback["tavily"] = "unavailable"
            return fallback
        fallback = _duckduckgo(query)
        fallback["tavily"] = "unavailable"
        return fallback


@tool
def search_web(query: str) -> dict:
    """Web search. At most two calls per symbol per run. Empty results are not the same as no news."""
    run = current_run()
    symbol = run.symbol if run else ""
    refusal = _reserve("search", symbol or "_", SEARCH_CAP)
    if refusal:
        return {"query": query, "refused": True, "error": refusal, "results": []}
    transport = SEARCH_TRANSPORT or _live_search
    try:
        found = transport(query)
    except Exception as error:
        return {
            "query": query,
            "source": "instant answer",
            "results": [],
            "error": str(error),
        }
    if not isinstance(found, dict):
        found = {"results": []}
    found.setdefault("query", query)
    found.setdefault("source", "search")
    found.setdefault("results", [])
    return found


def _slice_history(history: list[dict], days: int) -> list[dict]:
    if not history:
        return []
    if days >= RANGES["5y"]:
        return list(history)
    end = datetime.fromisoformat(history[-1]["date"])
    cutoff = (end - timedelta(days=days)).date().isoformat()
    window = [point for point in history if point["date"] >= cutoff]
    return window or list(history)


@tool
def chart_series(symbol: str, range: str, compare: str = "none") -> dict:
    """Price chart spec for 3m, 1y, or 5y. compare is portfolio or none. This does not call Yahoo or the web."""
    run = current_run()
    label = (run.symbol if run else symbol) or symbol
    refusal = _reserve("chart", label, CHART_CAP)
    if refusal:
        return {"symbol": symbol, "refused": True, "error": refusal}
    window = RANGES.get(range)
    if window is None:
        return {"symbol": symbol, "error": "range must be 3m, 1y, or 5y"}
    book = _books().get(symbol) or {}
    history = _slice_history(book.get("history") or [], window)
    if len(history) < 2:
        return {"symbol": symbol, "error": f"No stored history for {symbol}."}
    dates = [point["date"] for point in history]
    closes = [point["close"] for point in history]
    high_index = max(enumerate(closes), key=lambda pair: pair[1])[0]
    spec = {
        "symbol": symbol,
        "range": range,
        "compare": compare if compare in {"portfolio", "none"} else "none",
        "title": f"{symbol} close, {range}",
        "dates": dates,
        "closes": closes,
        "markers": [
            {"kind": "latest", "date": dates[-1], "price": closes[-1]},
            {"kind": "window high", "date": dates[high_index], "price": closes[high_index]},
        ],
    }
    if compare == "portfolio" and run is not None:
        import app

        holdings = []
        for ticker, stored in _books().items():
            if stored.get("history") and stored.get("price") is not None and stored.get("shares") is not None:
                holdings.append(stored)
        series = app._series(holdings, window)
        spec["portfolio"] = {
            "dates": [point["date"] for point in series],
            "values": [point["value"] for point in series],
        }
        spec["title"] = f"{symbol} close, {range}, versus the portfolio"
    if run is not None:
        run.charts.append(spec)
    return spec
