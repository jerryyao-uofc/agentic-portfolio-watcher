"""Research flags, graph, tool caps, and the HTTP route. No live DeepSeek key."""

from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from langchain_core.messages import AIMessage

import app
from research import agents, graph, memory, tools
from research.flags import select_flags

ROOT = Path(__file__).resolve().parents[1]


def holding(**overrides):
    row = {
        "symbol": "AAA",
        "name": "Aaa",
        "type": "stock",
        "isEtf": False,
        "shares": 10,
        "price": 100.0,
        "value": 1000.0,
        "weight": 0.1,
        "industry": "Software",
        "change3m": 0.0,
        "high3m": 120.0,
    }
    row.update(overrides)
    return row


class Sink:
    def __init__(self):
        self.bound = {}
        self.messages = {}

    def __call__(self, role):
        return FakeChat(role, self)


class FakeChat:
    def __init__(self, role, sink):
        self.role = role
        self.sink = sink

    def bind_tools(self, tool_list):
        names = [tool.name for tool in tool_list]
        self.sink.bound[self.role] = names
        return self

    def invoke(self, messages):
        self.sink.messages.setdefault(self.role, []).append(messages)
        tool_names = [getattr(message, "name", None) for message in messages if getattr(message, "type", "") == "tool"]
        symbol = _symbol_from(messages)
        if self.role == "writer":
            if "chart_series" not in tool_names:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "chart_series",
                            "args": {"symbol": symbol, "range": "3m", "compare": "none"},
                            "id": "chart-1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(
                content=(
                    "What changed is covered against the previous memo. "
                    "Stretched areas, what would have to change to justify selling, "
                    "what argues against selling, and what is unknown."
                )
            )
        tool_name = {
            "financial": "get_fundamentals",
            "news": "get_headlines",
            "competitive": "get_peers",
            "risk": "get_position_risk",
        }[self.role]
        if tool_name not in tool_names:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": {"symbol": symbol},
                        "id": f"{self.role}-1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content=f"{self.role} memo for the position.")


def _symbol_from(messages) -> str:
    for message in messages:
        text = message.content if isinstance(getattr(message, "content", ""), str) else ""
        marker = '"symbol": "'
        if marker in text:
            return text.split(marker, 1)[1].split('"', 1)[0]
    return "NVDA"


def sample_history():
    start = datetime(2026, 1, 1)
    return [
        {"date": (start + timedelta(days=offset)).date().isoformat(), "close": 100 + offset}
        for offset in range(120)
    ]


def nvda_payload():
    row = holding(
        symbol="NVDA",
        name="NVIDIA",
        value=5000,
        weight=0.2,
        industry="Semiconductors",
        change3m=0.1,
        high3m=250,
        price=200,
    )
    return {
        "holdings": [row],
        "lookthrough": {"rows": [{"symbol": "NVDA", "weight": 0.2, "directWeight": 0.2}]},
        "flags": [
            {
                "symbol": "NVDA",
                "name": "NVIDIA",
                "reasons": ["one of the three largest direct stock positions"],
            }
        ],
        "files": ["holdings.csv"],
        "warnings": [],
        "asOf": "2026-11-22T00:00:00+00:00",
    }


class FlagTests(unittest.TestCase):
    def test_rules_ignore_etfs_and_crypto_and_cap_at_three(self):
        payload = {
            "holdings": [
                holding(symbol="AAA", value=500, weight=0.2, price=100, high3m=140, change3m=0.0),
                holding(symbol="BBB", value=200, weight=0.05, price=100, high3m=140, change3m=0.0),
                holding(symbol="CCC", value=100, weight=0.04, price=100, high3m=140, change3m=0.0),
                holding(symbol="DDD", value=50, weight=0.02, price=96, high3m=100, change3m=0.40),
                holding(symbol="EEE", value=40, weight=0.01, price=50, high3m=100, change3m=0.25),
                holding(symbol="VOO", value=900, isEtf=True, type="stock", change3m=0.80, price=100, high3m=100),
                holding(symbol="BTC", value=800, type="crypto", isEtf=False, change3m=0.90, price=100, high3m=100),
            ],
            "lookthrough": {
                "rows": [
                    {"symbol": "BBB", "weight": 0.09},
                    {"symbol": "AAA", "weight": 0.05},
                    {"symbol": "VOO", "weight": 0.40},
                    {"symbol": "BTC", "weight": 0.30},
                ]
            },
        }
        flags = select_flags(payload)
        symbols = [row["symbol"] for row in flags]
        self.assertEqual(symbols, ["BBB", "DDD", "AAA"])
        self.assertNotIn("VOO", symbols)
        self.assertNotIn("BTC", symbols)
        self.assertNotIn("CCC", symbols)
        self.assertNotIn("EEE", symbols)
        by_symbol = {row["symbol"]: row["reasons"] for row in flags}
        self.assertIn("look-through weight above 8%", by_symbol["BBB"])
        self.assertIn("one of the three largest direct stock positions", by_symbol["BBB"])
        self.assertIn("price within 5% of the 3-month high", by_symbol["DDD"])
        self.assertIn("up more than 25% over 3 months", by_symbol["DDD"])


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = ROOT / "data" / f"research_memory_test_{threading.get_ident()}.json"
        self.prior_path = memory.PATH
        memory.PATH = self.tmp
        self.prior_override = agents.MODEL_OVERRIDE
        self.prior_peers = tools.PEER_LOADER
        self.prior_headlines = tools.HEADLINE_LOADER
        self.prior_transport = tools.SEARCH_TRANSPORT
        self.prior_books = dict(app.SYMBOL_BOOKS)
        self.prior_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        agents.MODEL_OVERRIDE = None
        tools.PEER_LOADER = lambda symbol: ["AMD", "AVGO"]
        tools.HEADLINE_LOADER = lambda symbol: [
            {"symbol": symbol, "title": "Chip demand holds", "published": "2026-11-01"},
            {"symbol": symbol, "title": "Supply comment", "published": "2026-11-02"},
        ]
        tools.SEARCH_TRANSPORT = lambda query: {"query": query, "source": "tavily", "results": []}
        app.SYMBOL_BOOKS.clear()
        app.SYMBOL_BOOKS["NVDA"] = {
            "symbol": "NVDA",
            "yahoo": "NVDA",
            "name": "NVIDIA",
            "shares": 10,
            "price": 200,
            "type": "stock",
            "isEtf": False,
            "history": sample_history(),
            "fundamentals": {"trailingPe": 40.0},
        }

    def tearDown(self):
        memory.PATH = self.prior_path
        agents.MODEL_OVERRIDE = self.prior_override
        tools.PEER_LOADER = self.prior_peers
        tools.HEADLINE_LOADER = self.prior_headlines
        tools.SEARCH_TRANSPORT = self.prior_transport
        app.SYMBOL_BOOKS.clear()
        app.SYMBOL_BOOKS.update(self.prior_books)
        if self.prior_key is not None:
            os.environ["DEEPSEEK_API_KEY"] = self.prior_key
        else:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        if self.tmp.exists():
            self.tmp.unlink()

    def test_graph_order_and_writer_cannot_search(self):
        sink = Sink()
        agents.MODEL_OVERRIDE = sink
        result = graph.run_symbol(nvda_payload(), "NVDA")
        nodes = [step["node"] for step in result["trace"]]
        self.assertEqual(nodes, ["context", "financial", "news", "competitive", "risk", "writer"])
        self.assertEqual(
            [step["tools"] for step in result["trace"] if step["node"] != "context"],
            [
                ["get_fundamentals"],
                ["get_headlines"],
                ["get_peers"],
                ["get_position_risk"],
                ["chart_series"],
            ],
        )
        self.assertEqual(agents.writer_tool_names(), ["chart_series"])
        self.assertEqual(sink.bound["writer"], ["chart_series"])
        self.assertNotIn("search_web", sink.bound["writer"])
        for role in ("financial", "news", "competitive", "risk"):
            self.assertIn(role, sink.messages)
        writer_text = "\n".join(
            message.content
            for batch in sink.messages["writer"]
            for message in batch
            if isinstance(getattr(message, "content", ""), str)
        )
        self.assertIn("financial memo for the position.", writer_text)
        self.assertIn("news memo for the position.", writer_text)
        self.assertIn("competitive memo for the position.", writer_text)
        self.assertIn("risk memo for the position.", writer_text)
        self.assertTrue(result["charts"])
        self.assertIn("dates", result["charts"][0])

        class Searchy:
            def bind_tools(self, tool_list):
                return self

            def invoke(self, messages):
                if not any(getattr(message, "type", "") == "tool" for message in messages):
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_web",
                                "args": {"query": "should fail"},
                                "id": "denied",
                                "type": "tool_call",
                            }
                        ],
                    )
                joined = "\n".join(message.content for message in messages if getattr(message, "type", "") == "tool")
                self.denied = joined
                return AIMessage(content="writer stopped")

        searchy = Searchy()
        text, called = agents.run_agent(searchy, "system", "human", agents.WRITER_TOOLS)
        self.assertEqual(called, ["search_web"])
        self.assertIn("not available", searchy.denied)
        self.assertEqual(text, "writer stopped")

    def test_search_refuses_a_third_call(self):
        calls = []

        def transport(query):
            calls.append(query)
            return {"query": query, "source": "tavily", "results": [{"title": "A", "url": "https://example.com", "snippet": "note"}]}

        tools.SEARCH_TRANSPORT = transport
        tools.start_run("NVDA", nvda_payload(), "search-cap")
        try:
            first = tools.search_web.invoke({"query": "one"})
            second = tools.search_web.invoke({"query": "two"})
            third = tools.search_web.invoke({"query": "three"})
        finally:
            tools.end_run("search-cap")
        self.assertNotIn("refused", first)
        self.assertNotIn("refused", second)
        self.assertTrue(third["refused"])
        self.assertEqual(calls, ["one", "two"])

    def test_chart_spec_and_third_call_refused(self):
        tools.start_run("NVDA", nvda_payload(), "chart-cap")
        try:
            first = tools.chart_series.invoke({"symbol": "NVDA", "range": "3m", "compare": "none"})
            second = tools.chart_series.invoke({"symbol": "NVDA", "range": "1y", "compare": "none"})
            third = tools.chart_series.invoke({"symbol": "NVDA", "range": "5y", "compare": "none"})
        finally:
            tools.end_run("chart-cap")
        self.assertEqual(first["title"], "NVDA close, 3m")
        self.assertGreaterEqual(len(first["dates"]), 2)
        self.assertEqual(len(first["dates"]), len(first["closes"]))
        kinds = [marker["kind"] for marker in first["markers"]]
        self.assertEqual(kinds, ["latest", "window high"])
        self.assertNotIn("refused", second)
        self.assertTrue(third["refused"])

    def test_prior_memo_is_writer_input(self):
        memory.save("NVDA", "PRIOR-MEMO-ALPHA", "old")
        sink = Sink()
        agents.MODEL_OVERRIDE = sink
        graph.run_symbol(nvda_payload(), "NVDA")
        writer_text = "\n".join(
            message.content
            for batch in sink.messages["writer"]
            for message in batch
            if isinstance(getattr(message, "content", ""), str)
        )
        self.assertIn("PRIOR-MEMO-ALPHA", writer_text)
        self.assertIn("what changed", writer_text.lower())

    def test_http_research_and_missing_key(self):
        sink = Sink()
        agents.MODEL_OVERRIDE = sink
        app._cache["payload"] = nvda_payload()
        app._cache["at"] = app.time.time()
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/research",
                data=json.dumps({"symbols": ["NVDA"]}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode())
            self.assertEqual(response.status, 200)
            run = body["runs"][0]
            self.assertIn("financial memo", run["memos"]["financial"])
            self.assertTrue(run["writer"])
            self.assertEqual([step["node"] for step in run["trace"]][0], "context")
            self.assertEqual([step["node"] for step in run["trace"]][-1], "writer")
            self.assertTrue(run["charts"])
            self.assertIn("reasons", run)

            agents.MODEL_OVERRIDE = None
            os.environ.pop("DEEPSEEK_API_KEY", None)
            missing = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/research",
                data=json.dumps({"shortlist": True}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(missing, timeout=10)
            self.assertEqual(caught.exception.code, 400)
            error = json.loads(caught.exception.read().decode())
            self.assertIn("DEEPSEEK_API_KEY", error["error"])

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/portfolio", timeout=10) as response:
                portfolio = json.loads(response.read().decode())
            self.assertEqual(response.status, 200)
            self.assertEqual(portfolio["holdings"][0]["symbol"], "NVDA")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
