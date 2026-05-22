from __future__ import annotations

import json
import os
from typing import Any, Literal

from langchain_openai import ChatOpenAI

from tools import tool_list_brands, tool_list_categories, tool_search_products
from history import get_last_messages


RouteType = Literal["products", "brands", "categories", "greeting", "advice", "chat", "intent_search"]


def list_categories_tool_call(limit: int = 50) -> str:
    """List categories from DB and return JSON array."""
    return json.dumps(tool_list_categories(limit=limit), ensure_ascii=False)


def list_brands_tool_call(limit: int = 50) -> str:
    """List brands from DB and return JSON array."""
    return json.dumps(tool_list_brands(limit=limit), ensure_ascii=False)


_intent_llm = ChatOpenAI(
    model=os.getenv("SODDA_INTENT_MODEL", os.getenv("SODDA_ROUTER_MODEL", "gpt-4o-mini")),
    temperature=0,
    max_tokens=300,
)


def classify_intent(user_text: str, chat_id: str, context: dict = None) -> dict[str, Any]:
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

    # Fetch last few messages for context
    history = get_last_messages(chat_id, limit=6)
    history_str = ""
    for m in history:
        role = "Assistant" if getattr(m, "type", "") == "ai" else "User"
        content = getattr(m, "content", "")
        # Truncate very long messages to save tokens
        if len(content) > 300:
            content = content[:300] + "..."
        history_str += f"{role}: {content}\n"

    prev_query = (context.get("query", "") if context else "") or ""
    prev_query = str(prev_query).strip()
    
    system = (
        "You are an intent router for a Telegram shop bot (Sodda.uz).\n"
        "The shop sells: Home appliances, Kitchen electronics, Climate tech.\n\n"
        "Based on the conversation history and the latest message, decide the user's intent.\n"
        "Return ONLY valid JSON with keys:\n"
        '  "intent": one of ["greeting","categories","brands","advice","product_search","chat"]\n'
        '  "query": string (product name/type for product_search)\n'
        '  "brand": string|null\n'
        '  "min_usd": number|null\n'
        '  "max_usd": number|null\n'
        '  "size_token": string|null\n\n'
        "Rules:\n"
        "- If user asks for a specific brand or budget for a previously discussed item, set intent to 'product_search'.\n"
        "- If user says 'Samsung' after looking for 'Televizor', query should be 'Televizor' and brand 'Samsung'.\n"
        "- If user asks what you sell => categories.\n"
        "- If user asks 'brands/brendlar' => brands.\n"
        "- If user asks for advice => advice.\n"
        "- If user is searching/asking availability/price => product_search.\n"
        "- Greeting/Thanks => greeting.\n"
        "- Otherwise => chat.\n"
    )

    prompt = f"HISTORY:\n{history_str}\nLATEST MESSAGE: {user_text}\n\nPREVIOUS CONTEXT QUERY: {prev_query}"

    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    
    try:
        res = _intent_llm.invoke(msgs, response_format={"type": "json_object"})
    except Exception:
        # Fallback for models not supporting response_format
        res = _intent_llm.invoke(msgs)

    try:
        content = (res.content or "").strip()
        data = json.loads(content)
        if isinstance(data, dict) and "intent" in data:
            # Soft fallback: if model returned product_search with empty query, reuse previous context.
            if (data.get("intent") == "product_search") and prev_query and not (data.get("query") or "").strip():
                data["query"] = prev_query
            return data
    except Exception:
        pass
    
    return {"intent": "product_search", "query": user_text, "brand": None, "min_usd": None, "max_usd": None, "size_token": None}


def route_user_message(user_text: str, chat_id: str, context: dict = None) -> dict[str, Any]:
    """
    Returns a dict:
      - type: 'products'|'brands'|'categories'|'greeting'|'advice'|'chat'|'intent_search'
      - data: tool output (parsed JSON) or []
      - query: used query string (for products)
      - text: text response (for greeting/advice/chat)
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return {"type": "chat", "text": ""}

    lowered = user_text.lower().strip()
    
    # Fast-path for common keywords to save LLM calls
    if any(k in lowered for k in ["brend", "brand", "brands", "бренд"]):
        raw = list_brands_tool_call(limit=5000)
        return {"type": "brands", "data": json.loads(raw), "query": ""}
    
    if any(k in lowered for k in ["kategoriya", "category", "categories", "категор", "bo'lim", "bo'limlar"]):
        raw = list_categories_tool_call(limit=5000)
        return {"type": "categories", "data": json.loads(raw), "query": ""}

    if any(p in lowered for p in ["salom", "assalomu alaykum", "salam", "hello", "hi"]):
        return {
            "type": "greeting",
            "text": "Assalomu alaykum! Qanday yordam bera olaman?\nMahsulot qidirish uchun nomini yozing (masalan: <b>televizor</b>)."
        }

    # LLM intent classification
    intent = classify_intent(user_text, chat_id, context=context)
    it = (intent.get("intent") or "").strip()
    
    if it == "greeting":
        return {
            "type": "greeting",
            "text": "Assalomu alaykum! Qanday yordam bera olaman?\nMahsulot qidirish uchun nomini yozing."
        }
    if it == "categories":
        raw = list_categories_tool_call(limit=5000)
        return {"type": "categories", "data": json.loads(raw), "query": ""}
    if it == "brands":
        raw = list_brands_tool_call(limit=5000)
        return {"type": "brands", "data": json.loads(raw), "query": ""}
    if it == "advice":
        return {
            "type": "advice",
            "text": (
                "Albatta! Tavsiya berishim uchun:\n"
                "1) Nima qidiryapsiz? (masalan: konditsioner)\n"
                "2) Byudjetingiz qancha?\n"
                "3) Brend xohishi bormi?"
            )
        }
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
    
    return {"type": "chat", "text": user_text}
