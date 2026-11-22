"""Direct-stock flags. No model calls."""

from __future__ import annotations

MAX_FLAGS = 3
LOOKTHROUGH_LIMIT = 0.08
RALLY_LIMIT = 0.25
NEAR_HIGH = 0.95


def _direct_stocks(holdings: list[dict]) -> list[dict]:
    rows = []
    for item in holdings:
        if item.get("type") == "crypto" or item.get("isEtf"):
            continue
        if item.get("type") not in (None, "stock"):
            continue
        rows.append(item)
    rows.sort(key=lambda item: (-float(item.get("value") or 0), str(item.get("symbol") or "")))
    return rows


def _look_weight(symbol: str, lookthrough: dict) -> float | None:
    for row in (lookthrough or {}).get("rows") or []:
        if row.get("symbol") == symbol and row.get("weight") is not None:
            weight = float(row["weight"])
            if weight > 1:
                weight /= 100
            return weight
    return None


def select_flags(payload: dict) -> list[dict]:
    """At most three direct stocks. ETFs and crypto are never flagged.

    When more than three names match, keep the ones with more reasons,
    then the larger direct position.
    """
    holdings = payload.get("holdings") or []
    direct = _direct_stocks(holdings)
    largest = {item.get("symbol") for item in direct[:3]}
    lookthrough = payload.get("lookthrough") or {}
    matched = []
    for item in direct:
        symbol = item.get("symbol")
        reasons = []
        look_weight = _look_weight(symbol, lookthrough)
        if look_weight is not None and look_weight > LOOKTHROUGH_LIMIT:
            reasons.append("look-through weight above 8%")
        if symbol in largest:
            reasons.append("one of the three largest direct stock positions")
        price = item.get("price")
        high = item.get("high3m")
        if price is not None and high:
            if float(price) >= float(high) * NEAR_HIGH:
                reasons.append("price within 5% of the 3-month high")
        change = item.get("change3m")
        if change is not None and float(change) > RALLY_LIMIT:
            reasons.append("up more than 25% over 3 months")
        if not reasons:
            continue
        matched.append(
            {
                "symbol": symbol,
                "name": item.get("name") or symbol,
                "reasons": reasons,
                "value": float(item.get("value") or 0),
            }
        )
    matched.sort(key=lambda row: (-len(row["reasons"]), -row["value"], row["symbol"]))
    chosen = []
    for row in matched[:MAX_FLAGS]:
        chosen.append({"symbol": row["symbol"], "name": row["name"], "reasons": row["reasons"]})
    return chosen
