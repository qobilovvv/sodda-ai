from __future__ import annotations

import json
import os
from typing import Any, Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from tools import tool_list_brands, tool_list_categories, tool_search_products


RouteType = Literal["products", "brands", "categories", "greeting", "advice", "chat", "intent_search"]


@tool("search_products")
def search_products_tool(query: str, limit: int = 10) -> str:
    """Search products in DB and return a JSON array of product summaries."""
    products = tool_search_products(query, limit=limit)
    data = [
        {
            "id": p.id,
            "title": p.title,
            "model": p.model,
            "brand": p.brand,
            "category": p.category,
            "price_uzs": p.price_uzs,
            "image_urls": p.image_urls[:3],
        }
        for p in products
    ]
    return json.dumps(data, ensure_ascii=False)


@tool("list_categories")
def list_categories_tool(limit: int = 50) -> str:
    """List categories from DB and return JSON array."""
    return json.dumps(tool_list_categories(limit=limit), ensure_ascii=False)


@tool("list_brands")
def list_brands_tool(limit: int = 50) -> str:
    """List brands from DB and return JSON array."""
    return json.dumps(tool_list_brands(limit=limit), ensure_ascii=False)


_router_llm = ChatOpenAI(
    model=os.getenv("SODDA_ROUTER_MODEL", "gpt-4o-mini"),
    temperature=0,
    max_tokens=400,
).bind_tools([search_products_tool, list_categories_tool, list_brands_tool])


_intent_llm = ChatOpenAI(
    model=os.getenv("SODDA_INTENT_MODEL", os.getenv("SODDA_ROUTER_MODEL", "gpt-4o-mini")),
    temperature=0,
    max_tokens=300,
)


def classify_intent(user_text: str) -> dict[str, Any]:
    """
    Returns JSON dict:
      intent: categories|brands|advice|product_search|chat|greeting
      query: string (for product_search)
      brand: optional string
      min_usd/max_usd: optional int
      size_token: optional string
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return {"intent": "chat"}

    system = (
        "You are an intent router for a Telegram shop bot.\n"
        "The shop sells only home appliances/electronics, but users may chat freely.\n"
        "Decide what the message means and extract filters.\n\n"
        "Return ONLY valid JSON with keys:\n"
        '  "intent": one of ["greeting","categories","brands","advice","product_search","chat"]\n'
        '  "query": string (only for product_search)\n'
        '  "brand": string|null\n'
        '  "min_usd": number|null\n'
        '  "max_usd": number|null\n'
        '  "size_token": string|null\n\n'
        "Rules:\n"
        "- If user asks what you sell/assortment => categories.\n"
        "- If user asks 'brands/brendlar' => brands.\n"
        "- If user asks for advice/recommendation => advice.\n"
        "- If user is searching/asking availability/price of an item => product_search.\n"
        "- Otherwise => chat.\n"
        "- If user included budget like '500$' treat it as max_usd unless explicitly 'from/above'.\n"
        "- If user included a brand name, put it in brand.\n"
    )

    res = _intent_llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]
    )
    try:
        data = json.loads((res.content or "").strip())
        if isinstance(data, dict) and "intent" in data:
            return data
    except Exception:
        pass
    return {"intent": "product_search", "query": user_text, "brand": None, "min_usd": None, "max_usd": None, "size_token": None}


def route_user_message(user_text: str) -> dict[str, Any]:
    """
    Returns a dict:
      - type: 'products'|'brands'|'categories'
      - data: tool output (parsed JSON) or []
      - query: used query string (for products)
      - error: optional
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return {"type": "products", "data": [], "query": ""}

    # Simple local intent handling: greetings/thanks should not trigger DB search.
    lowered = user_text.lower().strip()
    # Advice intent: user asks for help choosing, not searching a specific item.
    if any(
        p in lowered
        for p in [
            "maslahat",
            "tavsiya",
            "yordam bering",
            "help me choose",
            "help choose",
            "i need advice",
            "need advice",
            "recommend",
            "recommendation",
            "qaysi biri yaxshi",
            "qaysi yaxshiroq",
            "nima olsam bo'ladi",
        ]
    ):
        return {
            "type": "advice",
            "text": (
                "Albatta! Eng zo'r variantni tavsiya qilishim uchun 3 ta savol:\n"
                "1) Qaysi tur kerak: <b>Katta maishiy</b>, <b>Kichik maishiy</b>, <b>Iqlim texnikasi</b> yoki <b>Uy uchun elektronika</b>?\n"
                "2) Byudjetingiz qancha (<b>USD</b>)?\n"
                "3) Brend bo‘yicha xohish bormi? (ixtiyoriy)\n\n"
                "Masalan: “Konditsioner, 500$, Midea”"
            ),
        }
    if any(
        p in lowered
        for p in [
            "nima sotasiz",
            "nima sotasan",
            "nima sotiladi",
            "nima bor",
            "assortiment",
            "assortment",
            "what do you sell",
            "what do you have",
            "что продаете",
            "что вы продаете",
            "ассортимент",
            "что есть",
        ]
    ):
        # For "what do you sell?" always show categories.
        raw = list_categories_tool.invoke({"limit": 5000})
        return {"type": "categories", "data": json.loads(raw), "query": ""}
    if lowered in {
        "salom",
        "assalomu alaykum",
        "assalom alaykum",
        "salam",
        "hello",
        "hi",
        "hey",
        "rahmat",
        "thanks",
        "thank you",
        }:
            return {
                "type": "greeting",
                "text": (
                    "Assalomu alaykum! Qanday yordam bera olaman?\n"
                    "Mahsulot qidirish uchun nomini yozing (masalan: <b>iPhone 15</b> yoki <b>muzlatgich</b>)."
                ),
            }

    # LLM intent classification for everything else.
    intent = classify_intent(user_text)
    it = (intent.get("intent") or "").strip()
    if it == "greeting":
        return {
            "type": "greeting",
            "text": (
                "Assalomu alaykum! Qanday yordam bera olaman?\n"
                "Mahsulot qidirish uchun nomini yozing (masalan: <b>televizor</b> yoki <b>muzlatgich</b>)."
            ),
        }
    if it == "categories":
        raw = list_categories_tool.invoke({"limit": 5000})
        return {"type": "categories", "data": json.loads(raw), "query": ""}
    if it == "brands":
        raw = list_brands_tool.invoke({"limit": 5000})
        return {"type": "brands", "data": json.loads(raw), "query": ""}
    if it == "advice":
        return route_user_message("i need advice")
    if it == "chat":
        return {"type": "chat", "text": user_text}
    if it == "product_search":
        return {
            "type": "intent_search",
            "query": (intent.get("query") or user_text).strip(),
            "filters": {
                "brand": intent.get("brand"),
                "min_usd": intent.get("min_usd"),
                "max_usd": intent.get("max_usd"),
                "size_token": intent.get("size_token"),
            },
        }

    # NOTE: We intentionally do not hard-block "out of scope" queries at the router level.
    # The catalog itself (DB results) will naturally constrain what is shown.

    system = (
        "You are a router for a shop bot. Decide which DB tool to call.\n"
        "If user asks to see categories (kategoriyalar/bo'limlar), call list_categories.\n"
        "If user asks about brands (brendlar/brands), call list_brands.\n"
        "Otherwise, call search_products with query=user text.\n"
        "Return tool output only; do not answer in natural language."
    )

    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    res = _router_llm.invoke(msg)

    # If the model produced tool calls, execute them.
    tool_calls = getattr(res, "tool_calls", None) or []
    if not tool_calls:
        # Fallback to product search.
        raw = search_products_tool.invoke({"query": user_text, "limit": 10})
        data = json.loads(raw)
        return {"type": "products", "data": data, "query": user_text}

    call = tool_calls[0]
    name = call.get("name")
    args = call.get("args") or {}

    try:
        if name == "list_categories":
            if "limit" not in args:
                args["limit"] = 5000
            raw = list_categories_tool.invoke(args)
            return {"type": "categories", "data": json.loads(raw), "query": ""}
        if name == "list_brands":
            if "limit" not in args:
                args["limit"] = 5000
            raw = list_brands_tool.invoke(args)
            return {"type": "brands", "data": json.loads(raw), "query": ""}
        if name == "search_products":
            raw = search_products_tool.invoke(args)
            return {"type": "products", "data": json.loads(raw), "query": str(args.get("query", user_text))}
    except Exception as e:
        # Last-resort fallback to product search
        try:
            raw = search_products_tool.invoke({"query": user_text, "limit": 10})
            data = json.loads(raw)
            return {"type": "products", "data": data, "query": user_text, "error": str(e)}
        except Exception:
            return {"type": "products", "data": [], "query": user_text, "error": str(e)}

    raw = search_products_tool.invoke({"query": user_text, "limit": 10})
    data = json.loads(raw)
    return {"type": "products", "data": data, "query": user_text}
