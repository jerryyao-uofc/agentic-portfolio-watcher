"""LangChain agents: one chat model, its tools, and its messages.

The graph decides when each agent runs. These functions only run that agent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from research.tools import (
    chart_series,
    get_fundamentals,
    get_headlines,
    get_peers,
    get_position_risk,
    search_web,
)

MISSING_KEY = (
    "DEEPSEEK_API_KEY is not set. Add it to the environment, a local .env file, or secrets/deepseek.md. "
    "Portfolio prices still work. Research does not run without this key."
)

MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KEY_FILE = ROOT / "secrets" / "deepseek.md"
# Tests can point this at a temp file or a path that does not exist.
KEY_FILE: Path | None = None

# Tests inject a callable role -> chat model. The server leaves this as None.
MODEL_OVERRIDE = None

WRITER_TOOLS = [chart_series]


class MissingDeepSeekKey(RuntimeError):
    pass


def writer_tool_names() -> list[str]:
    return [tool.name for tool in WRITER_TOOLS]


def load_deepseek_key(path: Path | None = None) -> None:
    """Set DEEPSEEK_API_KEY from secrets/deepseek.md when the env var is empty.

    Uses the first non-empty line that is not a markdown heading and not the
    note about gitignore. Does nothing when the variable is already set, the
    file is missing, or that line is blank.
    """
    if (os.environ.get("DEEPSEEK_API_KEY") or "").strip():
        return
    key_path = path if path is not None else (KEY_FILE if KEY_FILE is not None else DEFAULT_KEY_FILE)
    if not key_path.is_file():
        return
    try:
        lines = key_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "gitignore" in stripped.lower():
            continue
        os.environ["DEEPSEEK_API_KEY"] = stripped
        return


def build_chat(role: str):
    if MODEL_OVERRIDE is not None:
        return MODEL_OVERRIDE(role)
    load_deepseek_key()
    key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if not key:
        raise MissingDeepSeekKey(MISSING_KEY)
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model=MODEL,
        api_key=key,
        base_url=BASE_URL,
        temperature=0,
        max_retries=1,
        extra_body={"thinking": {"type": "disabled"}},
    )


def _text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return "".join(parts).strip()
    return str(content or "").strip()


def run_agent(model, system: str, human: str, tools: list) -> tuple[str, list[str]]:
    """Tool loop owned by LangChain messages. Unknown tools are refused."""
    allowed = {tool.name: tool for tool in tools}
    bound = model.bind_tools(tools) if tools else model
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    called = []
    for _ in range(6):
        ai = bound.invoke(messages)
        if not isinstance(ai, AIMessage):
            ai = AIMessage(content=_text(ai))
        messages.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            return _text(ai), called
        for call in tool_calls:
            name = call.get("name") or ""
            called.append(name)
            if name not in allowed:
                payload = {"error": f"{name} is not available to this agent.", "refused": True}
            else:
                payload = allowed[name].invoke(call.get("args") or {})
            messages.append(
                ToolMessage(
                    content=json.dumps(payload, default=str),
                    name=name,
                    tool_call_id=call.get("id") or name,
                )
            )
    return _text(messages[-1]), called


def _context_blob(context: dict) -> str:
    return json.dumps(context, default=str, indent=2)


def run_financial(model, context: dict) -> tuple[str, list[str]]:
    system = (
        "You are a financial statement analyst on a research desk. "
        "Use get_fundamentals. Describe profitability, valuation, growth, and leverage only from the figures returned. "
        "If a field is missing, say it is missing. Do not recommend a trade."
    )
    human = f"Research this direct stock.\n{_context_blob(context)}"
    return run_agent(model, system, human, [get_fundamentals])


def run_news(model, context: dict) -> tuple[str, list[str]]:
    preview = get_headlines.invoke({"symbol": context.get("symbol")})
    headlines = preview.get("headlines") or []
    tools = [get_headlines]
    # RSS is thin when there are fewer than two headlines. Search stays inside the two-call cap.
    if len(headlines) < 2:
        tools.append(search_web)
    system = (
        "You are a news and market researcher. Start from the headlines. "
        "Call search_web only when those headlines are thin, and at most twice. "
        "An empty search result is not proof that there is no news. Do not recommend a trade."
    )
    human = (
        f"Research this direct stock.\n{_context_blob(context)}\n"
        f"Headlines already loaded:\n{json.dumps(preview, default=str)}"
    )
    return run_agent(model, system, human, tools)


def run_competitive(model, context: dict) -> tuple[str, list[str]]:
    system = (
        "You are a competitive intelligence analyst. Call get_peers. "
        "Note which peers are already in the portfolio. Do not recommend a trade."
    )
    human = f"Research this direct stock.\n{_context_blob(context)}"
    return run_agent(model, system, human, [get_peers])


def run_risk(model, context: dict) -> tuple[str, list[str]]:
    system = (
        "You are a risk analyst. Call get_position_risk. "
        "Explain concentration and what a 20 percent equity drop does to this position in dollars. "
        "Do not recommend a trade."
    )
    human = f"Research this direct stock.\n{_context_blob(context)}"
    return run_agent(model, system, human, [get_position_risk])


def run_writer(model, context: dict, memos: dict, prior_memo: str) -> tuple[str, list[str]]:
    prior = (prior_memo or "").strip()
    if prior:
        prior_block = (
            "Previous memo for this symbol:\n"
            f"{prior}\n"
            "Say what changed relative to that previous memo.\n"
        )
    else:
        prior_block = "There is no previous memo for this symbol.\n"
    system = (
        "You are the writer. You see only the four specialist memos and the position context. "
        "You may call chart_series at most twice. You must not call Yahoo, search, or any other tool. "
        "Synthesize the four memos into one conclusion. Cover what looks stretched, "
        "what would have to change to justify selling, what argues against selling, and what is unknown. "
        "End with a heading named Suggestion. Under it, recommend either selling a specific portion of this holding "
        "or doing nothing. If you recommend a sale, say what share of the current position to sell, "
        "as a percent of these shares and as a share count, and why. "
        "If the evidence is too thin or the position is not stretched, say to do nothing and why. "
        "Do not invent figures the memos did not provide."
    )
    human = (
        f"{prior_block}\nPosition context:\n{_context_blob(context)}\n\n"
        f"Financial statement analyst:\n{memos.get('financial') or ''}\n\n"
        f"News and market researcher:\n{memos.get('news') or ''}\n\n"
        f"Competitive intelligence:\n{memos.get('competitive') or ''}\n\n"
        f"Risk analyst:\n{memos.get('risk') or ''}\n"
    )
    return run_agent(model, system, human, WRITER_TOOLS)
