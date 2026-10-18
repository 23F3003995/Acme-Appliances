"""Acme ledger Q&A service.

POST /  {"question": "..."}  ->  {"answer": ...}

Flow: an LLM turns the English question into a small JSON "spec";
plain Python then computes the answer exactly from the ledger.
"""
import asyncio
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

IST = ZoneInfo("Asia/Kolkata")
EXPORT_URL = os.environ.get("LEDGER_EXPORT_URL", "")
RATES_URL = os.environ.get("LEDGER_RATES_URL", "")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4.1-mini")
LLM_URL = "https://aipipe.org/openrouter/v1/chat/completions"

app = FastAPI(title="Acme ledger agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------
# Data loading and cleaning
# --------------------------------------------------------------------------
def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clean_orders(rows: List[dict], usd_per_unit: Dict[str, float]) -> List[dict]:
    """Keep the row with the latest updated_at (compared as instants) per order id,
    convert dates to Asia/Kolkata and amounts to USD."""
    latest: Dict[str, dict] = {}
    for r in rows:
        upd = parse_ts(r["updated_at"])
        cur = latest.get(r["id"])
        if cur is None or upd >= cur["_upd"]:
            latest[r["id"]] = {**r, "_upd": upd}
    out = []
    for r in latest.values():
        created = parse_ts(r["created_at"]).astimezone(IST)
        rate = Decimal(str(usd_per_unit[r["currency"].strip().upper()]))
        out.append({
            "id": r["id"],
            "customer": r["customer"],
            "region": r["region"],
            "product": r["product"],
            "qty": int(r["qty"]),
            "status": r["status"].strip().lower(),
            "currency": r["currency"].strip().upper(),
            "usd": Decimal(str(r["amount"])) * rate,
            "date": created.date(),
            "month": created.strftime("%Y-%m"),
        })
    return out


DEFAULT_RATES = {"USD": 1, "EUR": 1.14, "INR": 0.01182}


def parse_export(text: str) -> tuple:
    """Return (rows, next_url). Accepts NDJSON, a JSON array, or a paged {"orders": [...]} object."""
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(l) for l in text.splitlines() if l.strip()], None
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict) and "orders" in data:
        return data["orders"], data.get("next")
    if isinstance(data, dict) and "id" in data:  # a single-line NDJSON file with one row
        return [data], None
    raise ValueError("unrecognised export format")


class Ledger:
    def __init__(self):
        self.orders: List[dict] = []
        self.ready = asyncio.Event()
        self.lock = asyncio.Lock()
        self.last_error = ""

    async def load(self):
        async with self.lock:
            if self.ready.is_set():
                return
            try:
                async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
                    rates = DEFAULT_RATES
                    if RATES_URL:
                        rr = await c.get(RATES_URL)
                        rr.raise_for_status()
                        rates = rr.json()["usd_per_unit"]
                    rows: List[dict] = []
                    url: Optional[str] = EXPORT_URL
                    for _ in range(200):  # follows "next" links if the export is paged
                        er = await c.get(url)
                        er.raise_for_status()
                        part, url = parse_export(er.text)
                        rows.extend(part)
                        if not url:
                            break
                self.orders = clean_orders(rows, rates)
                self.last_error = ""
                self.ready.set()
            except Exception as e:
                # never put the exception text in output: it can contain the signed URL
                code = getattr(getattr(e, "response", None), "status_code", "")
                self.last_error = f"{type(e).__name__} {code}".strip()
                print("ledger load failed:", self.last_error)
                raise

    async def get(self) -> List[dict]:
        if not self.ready.is_set():
            try:
                await self.load()
            except Exception:
                raise HTTPException(503, f"Ledger not loaded: {self.last_error}")
        return self.orders


ledger = Ledger()


@app.on_event("startup")
async def _startup():
    async def warm():
        try:
            await ledger.load()
        except Exception as e:  # retried lazily on first request
            print("startup load failed:", e)
    asyncio.create_task(warm())


# --------------------------------------------------------------------------
# Query engine (deterministic)
# --------------------------------------------------------------------------
REFUND_METRICS = {"refund_amount_usd", "refund_count"}


def q2(x: Decimal) -> float:
    return float(x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def norm(s: Any) -> str:
    return str(s).strip().casefold()


def metric_value(metric: str, rows: List[dict]):
    if metric in ("revenue_usd", "refund_amount_usd"):
        return q2(sum((r["usd"] for r in rows), Decimal(0)))
    if metric in ("order_count", "refund_count"):
        return len(rows)
    if metric == "units_sold":
        return sum(r["qty"] for r in rows)
    if metric == "unique_customers":
        return len({r["customer"] for r in rows})
    if metric == "avg_order_value_usd":
        return q2(sum((r["usd"] for r in rows), Decimal(0)) / len(rows)) if rows else 0.0
    raise ValueError(f"unknown metric {metric}")


def run_spec(spec: dict, orders: List[dict]):
    metric = spec.get("metric", "revenue_usd")
    statuses = spec.get("statuses") or (["refunded"] if metric in REFUND_METRICS else ["paid"])
    statuses = {norm(s) for s in statuses}
    f = spec.get("filters") or {}

    rows = [o for o in orders if o["status"] in statuses]
    for key in ("region", "product", "customer", "currency"):
        if f.get(key):
            want = f[key] if isinstance(f[key], list) else [f[key]]
            want = {norm(w) for w in want}
            rows = [o for o in rows if norm(o[key]) in want]
    if f.get("date_from"):
        d0 = datetime.fromisoformat(f["date_from"]).date()
        rows = [o for o in rows if o["date"] >= d0]
    if f.get("date_to"):
        d1 = datetime.fromisoformat(f["date_to"]).date()
        rows = [o for o in rows if o["date"] <= d1]

    group_by = spec.get("group_by")
    if not group_by:
        return metric_value(metric, rows)

    groups: Dict[str, List[dict]] = defaultdict(list)
    for o in rows:
        groups[o[group_by]].append(o)
    values = {k: metric_value(metric, v) for k, v in groups.items()}

    rank = spec.get("rank")
    if not rank:
        return dict(sorted(values.items()))
    reverse = rank.get("order", "desc") == "desc"
    items = sorted(values.items(), key=lambda kv: (-kv[1] if reverse else kv[1], kv[0]))
    limit = int(rank.get("limit") or 1)
    items = items[:limit]
    ret = rank.get("return", "key")
    if ret == "value":
        return items[0][1] if limit == 1 else [v for _, v in items]
    if ret == "both":
        return {k: v for k, v in items}
    keys = [k for k, _ in items]
    return keys[0] if limit == 1 and keys else keys


# --------------------------------------------------------------------------
# LLM: English question -> spec
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You convert questions about the Acme Appliances order ledger into a JSON query spec.
Reply with ONLY a JSON object, no prose.

Spec fields:
- metric: one of "revenue_usd" (sum of paid order amounts in USD), "refund_amount_usd" (USD value of refunded orders),
  "refund_count" (number of refunded orders), "order_count", "units_sold" (sum of qty), "unique_customers", "avg_order_value_usd"
- statuses: optional list overriding which order statuses count (values: "paid","refunded","void").
  Defaults: refund_* metrics use ["refunded"]; every other metric uses ["paid"]. Only override when the question
  clearly asks for different statuses (e.g. "orders placed" regardless of outcome -> ["paid","refunded","void"]).
- filters: optional object with any of: region, product, customer, currency (strings, exact spelling from the lists below),
  date_from, date_to (inclusive, YYYY-MM-DD, business dates). A month like "March 2026" = 2026-03-01..2026-03-31.
  A quarter like Q1 2026 = 2026-01-01..2026-03-31. "H1" = Jan 1..Jun 30.
- group_by: optional, one of "region","product","customer","month","currency". Use it for "which/top/most/least/by" questions.
- rank: optional, only with group_by: {"order":"desc"|"asc","limit":N,"return":"key"|"value"|"both"}.
  "which region had the highest revenue" -> group_by region, rank {order desc, limit 1, return key}.
  "top 3 products by units" -> rank {order desc, limit 3, return key}. If asked for the amount of the top item, return "value".
  Without rank, group_by returns a dict of group -> value.

Known regions: {regions}
Known products: {products}
Customer ids look like C1234.

Examples:
Q: What was the total revenue in USD from the North region in March 2026?
{"metric":"revenue_usd","filters":{"region":"North","date_from":"2026-03-01","date_to":"2026-03-31"}}
Q: How many orders were refunded for Kettle in Q2 2026?
{"metric":"refund_count","filters":{"product":"Kettle","date_from":"2026-04-01","date_to":"2026-06-30"}}
Q: Which product sold the most units in May 2026?
{"metric":"units_sold","filters":{"date_from":"2026-05-01","date_to":"2026-05-31"},"group_by":"product","rank":{"order":"desc","limit":1,"return":"key"}}
Q: How many distinct customers bought a Mixer?
{"metric":"unique_customers","filters":{"product":"Mixer"}}
"""


async def question_to_spec(question: str, orders: List[dict]) -> dict:
    regions = sorted({o["region"] for o in orders})
    products = sorted({o["product"] for o in orders})
    system = SYSTEM_PROMPT.replace("{regions}", ", ".join(regions)).replace("{products}", ", ".join(products))
    body = {
        "model": LLM_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
    }
    last_err: Optional[Exception] = None
    async with httpx.AsyncClient(timeout=8) as c:
        for _ in range(2):
            try:
                r = await c.post(LLM_URL, json=body, headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"})
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
                return json.loads(text)
            except Exception as e:  # retry once
                last_err = e
    raise HTTPException(502, f"LLM failed: {last_err}")


class Ask(BaseModel):
    question: str


@app.post("/")
@app.post("/{full_path:path}")  # accept any path, so /ask, /answer, /query etc. all work
async def ask(body: Ask, full_path: str = ""):
    orders = await ledger.get()
    spec = await question_to_spec(body.question, orders)
    try:
        return {"answer": run_spec(spec, orders)}
    except Exception as e:
        raise HTTPException(400, f"Could not answer (spec={spec}): {e}")


@app.get("/")
async def health():
    if not ledger.ready.is_set():
        try:
            await ledger.load()
        except Exception:
            pass
    return {"status": "ok" if ledger.ready.is_set() else "error", "orders_loaded": len(ledger.orders),
            "error": ledger.last_error,
            "export_url_set": bool(EXPORT_URL), "rates_url_set": bool(RATES_URL), "token_set": bool(AIPIPE_TOKEN)}

