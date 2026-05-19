from __future__ import annotations

import re

from .catalog import Product

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    t = (text or "").replace("&nbsp;", " ")
    t = re.sub(r"(?i)<br\\s*/?>", "\n", t)
    t = re.sub(r"(?i)</p\\s*>", "\n", t)
    t = _TAG_RE.sub("", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _format_price_usd(price_usd: float) -> str:
    if price_usd <= 0:
        return "Narxi so'rov bo'yicha"
    if float(int(price_usd)) == float(price_usd):
        return f"{int(price_usd)}$"
    return f"{price_usd:.2f}$"


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _characteristics_to_bullets(raw: str, max_lines: int = 10) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []

    # Handle python-repr-ish leftovers like "[{'title_uz': ...}]" by trying JSON-ish normalization.
    if raw.startswith("[{") and "title_uz" in raw and "value_uz" in raw:
        # At this stage, catalog.clean_json_field should already normalize, but keep a fallback.
        pass

    # If already formatted as bullet points, keep lines.
    if "•" in raw:
        lines = [ln.strip().lstrip("•").strip() for ln in raw.splitlines() if ln.strip()]
        return [ln for ln in lines if ln][:max_lines]

    # Common format from sync: "Key: Val, Key2: Val2"
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    bullets: list[str] = []
    for p in parts:
        if ":" not in p:
            continue
        key, val = p.split(":", 1)
        key = re.sub(r"\s+", " ", key).strip()
        val = re.sub(r"\s+", " ", val).strip()
        if not key or not val:
            continue
        bullets.append(f"{key}: {val}")
        if len(bullets) >= max_lines:
            break
    return bullets


def format_product_caption(product: Product, manager_phone: str, max_chars: int = 900) -> str:
    title = (product.title or "").strip() or f"Mahsulot #{product.id}"
    price_line = _format_price_usd(product.price_usd)

    desc = _truncate(_strip_html(product.description), max_chars=420)
    bullets = _characteristics_to_bullets(product.characteristics, max_lines=10)

    lines: list[str] = []
    lines.append(title)
    lines.append(f"💰 Narxi: {price_line}")
    lines.append("")

    if desc:
        lines.append(f"Tavsif: {desc}")
        lines.append("")

    if bullets:
        lines.append("Asosiy xarakteristikalari:")
        lines.extend([f"• {b}" for b in bullets])
        lines.append("")

    # Lightweight, deterministic "seller" nudge without LLM tokens.
    lines.append("Savolingiz bo'lsa yozing — byudjet va ehtiyojingizga qarab mos variantlarni tavsiya qilaman.")
    lines.append("")
    lines.append(f"📞 {manager_phone}")

    caption = "\n".join(lines).strip()
    if len(caption) <= max_chars:
        return caption
    # Ensure Telegram caption limits are respected.
    return _truncate(caption, max_chars=max_chars)
