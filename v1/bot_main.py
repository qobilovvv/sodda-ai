import asyncio
import logging
import re

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from .ai_router import AiContext, generate_ai_reply
from .catalog import (
    find_brand_names_like,
    get_products_by_ids,
    list_all_brand_names,
    list_brands,
    list_categories,
    search_product_ids_by_brand,
    search_product_ids_sql,
)
from .config import get_settings
from .formatting import format_product_caption
from .images import resolve_first_image
from .retrieval_chroma import search_product_ids_with_scores
from .state_store import StateStore


logger = logging.getLogger("v1.bot")

router = Router()
store = StateStore()

def _strip_phone(text: str, manager_phone: str) -> str:
    if not text:
        return text
    t = text.replace(manager_phone, "")
    # Remove common Uzbek phone patterns the model may add.
    t = re.sub(r"\+998\d{9}\b", "", t)
    t = re.sub(r"\+\d{7,15}\b", "", t)
    # Cleanup leftover separators/spaces.
    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"\s+([,!.?])", r"\1", t)
    return t.strip()


def _is_brands_request(text: str) -> bool:
    t = text.lower()
    if _is_brand_products_request(t):
        return False
    has_brand_word = any(k in t for k in ["brend", "brand", "marka", "бренд", "марка"])
    has_list_intent = any(
        k in t
        for k in [
            "ro'yxat",
            "royxat",
            "list",
            "qaysi",
            "qanaqa",
            "qanday",
            "hammasi",
            "bar",
            "bor",
            "what brands",
            "which brands",
            "спис",
            "какие",
            "есть",
        ]
    )
    return has_brand_word and has_list_intent


def _is_categories_request(text: str) -> bool:
    t = text.lower()
    has_cat_word = any(k in t for k in ["kategori", "kategoriya", "bo'lim", "bolim", "category", "катег", "раздел"])
    has_list_intent = any(
        k in t
        for k in [
            "ro'yxat",
            "royxat",
            "list",
            "qaysi",
            "qanaqa",
            "qanday",
            "hammasi",
            "bar",
            "bor",
            "what categories",
            "which categories",
            "спис",
            "какие",
            "есть",
        ]
    )
    return has_cat_word and has_list_intent


def _normalize_common_typos(text: str) -> str:
    t = text
    # common Uzbek transliteration typos
    t = re.sub(r"\bkanditsioner\b", "konditsioner", t, flags=re.IGNORECASE)
    t = re.sub(r"\bkondisioner\b", "konditsioner", t, flags=re.IGNORECASE)
    # vacuum cleaners
    t = re.sub(r"\bpilesos\b", "pylesos", t, flags=re.IGNORECASE)
    return t


def _normalize_query(text: str) -> str:
    text = (text or "").strip()
    # Users may start with "\" accidentally (e.g. "\samsung").
    text = text.lstrip("\\/").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_greeting(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    greetings = {
        "salom",
        "assalomu alaykum",
        "asalomu alaykum",
        "assalom alaykum",
        "alaykum salom",
        "hello",
        "hi",
        "hey",
        "privet",
        "привет",
        "здравствуйте",
        "salam",
    }
    if t in greetings:
        return True
    # short greetings like "salom!" / "hi!"
    t2 = re.sub(r"[!?.]", "", t).strip()
    return t2 in greetings


def _is_too_vague(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 3:
        return True
    has_digit = any(ch.isdigit() for ch in t)
    # Single generic word without digits usually needs clarification (budget/size).
    if not has_digit and len(t.split()) == 1 and len(t) <= 14:
        return True
    return False


def _is_more_request(text: str) -> bool:
    t = (text or "").strip().lower()
    t = re.sub(r"[!?.]", "", t).strip()
    return t in {"yana", "koproq", "ko'proq", "yana ko'rsat", "yana korsat"}

def _parse_selection_index(text: str) -> int | None:
    t = (text or "").strip().lower()
    t = re.sub(r"[!?.]", "", t).strip()
    if t.isdigit():
        n = int(t)
        return n if 1 <= n <= 10 else None
    mapping = {"birinchi": 1, "1-chi": 1, "1chi": 1, "ikkinchi": 2, "2-chi": 2, "2chi": 2, "uchinchi": 3, "3-chi": 3, "3chi": 3}
    return mapping.get(t)

def _extract_budget_usd(text: str) -> int | None:
    t = (text or "").lower()
    m = re.search(r"(\d{2,5})\s*\$|\$\s*(\d{2,5})|(\d{2,5})\s*usd", t)
    if not m:
        return None
    for g in m.groups():
        if g:
            try:
                return int(g)
            except Exception:
                return None
    return None


def _is_no_requirements(text: str) -> bool:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    keywords = [
        "bilmayman",
        "bilmiman",
        "farqi yoq",
        "farqi yo'q",
        "bilmay",
        "o'zim ham bilmayman",
        "i dont know",
        "don't know",
        "no idea",
        "whatever",
        "any",
        "hammasi bo'ladi",
        "qanday bo'lsa ham",
        "tavsiya qiling",
        "o'zingiz tanlang",
    ]
    return any(k in t for k in keywords)

def _is_out_of_scope(text: str) -> bool:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    # Add more as needed; keep conservative.
    return any(k in t for k in ["kitob", "books", "book", "roman", "adabiyot"])


def _looks_like_exact_model_query(text: str) -> bool:
    t = (text or "").strip()
    # If it contains a long-ish alphanumeric model code, we can search immediately.
    return bool(re.search(r"\b[A-Za-z]{2,}\d{2,}[A-Za-z0-9-]*\b", t))


def _needs_clarification(text: str) -> bool:
    # Ask before sending any product list unless query is very specific.
    if _looks_like_exact_model_query(text):
        return False
    # If user didn't provide budget, ask for it.
    if _extract_budget_usd(text) is None:
        return True
    return False


def _is_brand_products_request(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["brend", "brand", "marka", "бренд"]) and any(
        k in t for k in ["mahsulot", "product", "nima bor", "nimalar bor", "qaysi bor", "bor?"]
    )


async def _send_lines_chunked(message: Message, header: str, lines: list[str], chunk_size: int = 60) -> None:
    if not lines:
        await message.answer(header)
        return
    for i in range(0, len(lines), chunk_size):
        part = lines[i : i + chunk_size]
        await message.answer("\n".join(([header] if i == 0 else []) + part))


async def _send_text_split_by_chars(message: Message, text: str, max_chars: int = 3800) -> None:
    """
    Telegram hard limit is 4096 chars; keep some headroom.
    Splits by lines.
    """
    if len(text) <= max_chars:
        await message.answer(text)
        return
    lines = text.splitlines()
    buf: list[str] = []
    size = 0
    for ln in lines:
        add = (len(ln) + 1) if buf else len(ln)
        if buf and size + add > max_chars:
            await message.answer("\n".join(buf))
            buf = [ln]
            size = len(ln)
        else:
            buf.append(ln)
            size += add
    if buf:
        await message.answer("\n".join(buf))


def _get_history(session: dict) -> list[tuple[str, str]]:
    hist = session.get("history")
    if isinstance(hist, list):
        out: list[tuple[str, str]] = []
        for item in hist[-10:]:
            if isinstance(item, dict) and "role" in item and "text" in item:
                role = str(item["role"])
                text = str(item["text"])
                if role in {"user", "assistant", "note"}:
                    out.append((role, text))
        return out
    return []


def _append_history(session: dict, role: str, text: str) -> None:
    if not text:
        return
    hist = session.get("history")
    if not isinstance(hist, list):
        hist = []
    hist.append({"role": role, "text": text})
    session["history"] = hist[-10:]


async def _send_product(message: Message, store: StateStore, caption: str, image_urls: list[str]) -> None:
    resolved_url = await resolve_first_image(image_urls)
    if not resolved_url:
        await message.answer(caption)
        return

    cached_file_id = store.get_file_id(resolved_url)
    if cached_file_id:
        try:
            await message.answer_photo(photo=cached_file_id, caption=caption)
            return
        except Exception:
            # Fall back to URL
            pass

    sent = await message.answer_photo(photo=resolved_url, caption=caption)
    if sent.photo:
        store.set_file_id(resolved_url, sent.photo[-1].file_id)


async def _send_product_simple(
    message: Message,
    store: StateStore,
    caption: str,
    image_urls: list[str],
) -> None:
    resolved_url = await resolve_first_image(image_urls)
    if not resolved_url:
        await message.answer(caption)
        return

    cached_file_id = store.get_file_id(resolved_url)
    if cached_file_id:
        try:
            await message.answer_photo(photo=cached_file_id, caption=caption)
            return
        except Exception:
            pass

    sent = await message.answer_photo(photo=resolved_url, caption=caption)
    if sent.photo:
        store.set_file_id(resolved_url, sent.photo[-1].file_id)


@router.message(CommandStart())
async def start(message: Message) -> None:
    settings = get_settings()
    session = store.get_session(message.from_user.id)
    reply = await asyncio.to_thread(
        generate_ai_reply,
        settings,
        AiContext(
            user_text="start",
            products=[],
            brands=None,
            categories=None,
            has_more=False,
            history=_get_history(session),
            last_shown_product_ids=session.get("last_shown") or [],
        ),
    )
    reply = _strip_phone(reply, settings.manager_phone)
    _append_history(session, "user", "/start")
    _append_history(session, "assistant", reply)
    store.set_session(message.from_user.id, session)
    await message.answer(reply)


@router.message(Command("clear"))
async def clear(message: Message) -> None:
    settings = get_settings()
    store.clear_session(message.from_user.id)
    await message.answer(_strip_phone("Suhbat konteksti tozalandi. Yana nimaga yordam beray?", settings.manager_phone))


@router.message(F.text)
async def handle_text(message: Message) -> None:
    settings = get_settings()
    session = store.get_session(message.from_user.id)

    query = _normalize_query(message.text or "")
    query = _normalize_common_typos(query)
    if not query:
        reply = await asyncio.to_thread(
            generate_ai_reply,
            settings,
            AiContext(
                user_text="(bo'sh xabar)",
                products=[],
                brands=None,
                categories=None,
                has_more=False,
                history=_get_history(session),
                last_shown_product_ids=session.get("last_shown") or [],
            ),
        )
        reply = _strip_phone(reply, settings.manager_phone)
        _append_history(session, "user", "(bo'sh)")
        _append_history(session, "assistant", reply)
        store.set_session(message.from_user.id, session)
        await message.answer(reply)
        return

    if _is_more_request(query):
        _append_history(session, "user", query)
        store.set_session(message.from_user.id, session)
        await _send_next_page(message, settings, store, user_text=query)
        return

    # If we previously asked clarification, merge user's answer into the pending query and continue.
    pending = session.get("pending")
    if isinstance(pending, dict) and pending.get("type") == "clarify" and pending.get("query"):
        base = str(pending.get("query") or "").strip()
        if _is_no_requirements(query):
            # User doesn't know requirements: proceed with the original query and show top products.
            session["pending"] = None
            session["force_top"] = True
            # Prefer prefetch pool captured during clarify.
            pre = session.get("prefetch_ids") or []
            if pre:
                session["ids"] = pre
                session["offset"] = 0
            store.set_session(message.from_user.id, session)
            query = base
        else:
            merged = f"{base} {query}".strip()
            session["pending"] = None
            session["force_top"] = False
            store.set_session(message.from_user.id, session)
            query = merged

    sel = _parse_selection_index(query)
    if sel is not None:
        last_shown: list[int] = session.get("last_shown") or []
        if 1 <= sel <= len(last_shown):
            pid = last_shown[sel - 1]
            products = await get_products_by_ids(settings, [pid])
            if products:
                p = products[0]
                caption = format_product_caption(p, settings.manager_phone, max_chars=settings.max_caption_chars)
                await _send_product_simple(message=message, store=store, caption=caption, image_urls=p.images)
                _append_history(session, "user", query)
                _append_history(session, "note", f"resent_product_id={pid}")
                store.set_session(message.from_user.id, session)
        return

    # For greetings/smalltalk, do NOT run retrieval (avoids random products like kir yuvish mashinasi).
    if _is_greeting(query) or len(query) <= 5:
        reply = await asyncio.to_thread(
            generate_ai_reply,
            settings,
            AiContext(
                user_text=query,
                products=[],
                brands=None,
                categories=None,
                has_more=False,
                history=_get_history(session),
                last_shown_product_ids=session.get("last_shown") or [],
            ),
        )
        reply = _strip_phone(reply, settings.manager_phone)
        _append_history(session, "user", query)
        _append_history(session, "assistant", reply)
        store.set_session(message.from_user.id, session)
        await message.answer(reply)
        return

    if _is_categories_request(query):
        cats = await list_categories(settings, limit=30)
        reply = await asyncio.to_thread(
            generate_ai_reply,
            settings,
            AiContext(
                user_text=query,
                products=[],
                categories=[n for _i, n in cats if n],
                brands=None,
                has_more=False,
                history=_get_history(session),
                last_shown_product_ids=session.get("last_shown") or [],
                mode="list",
            ),
        )
        reply = _strip_phone(reply, settings.manager_phone)
        _append_history(session, "user", query)
        _append_history(session, "assistant", reply)
        store.set_session(message.from_user.id, session)
        await message.answer(reply)
        return

    # "Samsung brendidan qaysi mahsulotlar bor?" style.
    if _is_brand_products_request(query):
        candidates = await find_brand_names_like(settings, query, limit=3)
        brand = candidates[0] if candidates else ""
        if brand:
            ids = await search_product_ids_by_brand(settings, brand, limit=80)
            if not ids:
                # Don't claim "mavjud emas". Ask a quick follow-up.
                reply = await asyncio.to_thread(
                    generate_ai_reply,
                    settings,
                    AiContext(
                        user_text=query,
                        products=[],
                        brands=None,
                        categories=None,
                        has_more=False,
                        history=_get_history(session),
                        last_shown_product_ids=session.get("last_shown") or [],
                        mode="clarify",
                    ),
                )
                reply = _strip_phone(reply, settings.manager_phone)
                _append_history(session, "user", query)
                _append_history(session, "assistant", reply)
                store.set_session(message.from_user.id, session)
                await message.answer(reply)
                return
            session["last_query"] = f"{brand} mahsulotlari"
            session["ids"] = ids
            session["offset"] = 0
            _append_history(session, "user", query)
            store.set_session(message.from_user.id, session)
            await _send_next_page(message, settings, store, user_text=query)
            return

    if _is_brands_request(query):
        # Send a single brand-list message (split only if Telegram limit is exceeded).
        brand_names = await list_all_brand_names(settings)
        _append_history(session, "user", query)
        store.set_session(message.from_user.id, session)
        text = "🏢 Brendlar ro'yxati:\n" + "\n".join([f"• {b}" for b in brand_names])
        await _send_text_split_by_chars(message, text)
        return

    # Clarify BEFORE retrieving/sending products (but not for brand-specific product listing).
    if _is_out_of_scope(query):
        reply = await asyncio.to_thread(
            generate_ai_reply,
            settings,
            AiContext(
                user_text=query,
                products=[],
                brands=None,
                categories=None,
                has_more=False,
                history=_get_history(session),
                last_shown_product_ids=session.get("last_shown") or [],
                mode="out_of_scope",
            ),
        )
        reply = _strip_phone(reply, settings.manager_phone)
        _append_history(session, "user", query)
        _append_history(session, "assistant", reply)
        store.set_session(message.from_user.id, session)
        await message.answer(reply)
        return

    if _needs_clarification(query) and not _is_brand_products_request(query):
        # Pre-search a pool so we can show "top products" if user says "bilmayman".
        pre_ids = await search_product_ids_sql(settings, query, limit=80)
        session["prefetch_ids"] = pre_ids
        reply = await asyncio.to_thread(
            generate_ai_reply,
            settings,
            AiContext(
                user_text=query,
                products=[],
                brands=None,
                categories=None,
                has_more=False,
                history=_get_history(session),
                last_shown_product_ids=session.get("last_shown") or [],
                mode="clarify",
            ),
        )
        reply = _strip_phone(reply, settings.manager_phone)
        _append_history(session, "user", query)
        _append_history(session, "assistant", reply)
        session["pending"] = {"type": "clarify", "query": query}
        session["force_top"] = False
        store.set_session(message.from_user.id, session)
        await message.answer(reply)
        return

    scored = await asyncio.to_thread(search_product_ids_with_scores, settings, query, 30)
    # Prefer vector results; if empty, fallback to SQL LIKE search.
    ids = [pid for pid, _s in scored]
    if not ids:
        ids = await search_product_ids_sql(settings, query, limit=30)
    # If user said "don't know requirements", just show more upfront.
    if session.get("force_top"):
        ids = ids[:80]

    session["last_query"] = query
    session["ids"] = ids
    session["offset"] = 0
    _append_history(session, "user", query)
    store.set_session(message.from_user.id, session)

    await _send_next_page(message, settings, store, user_text=query)


async def _send_next_page(message: Message, settings, store: StateStore, user_text: str) -> None:
    session = store.get_session(message.from_user.id)
    ids: list[int] = session.get("ids") or []
    offset: int = int(session.get("offset") or 0)
    if not ids and offset == 0:
        # No retrieval results: ask clarifying question via AI.
        reply = await asyncio.to_thread(
            generate_ai_reply,
            settings,
            AiContext(
                user_text=user_text,
                products=[],
                brands=None,
                categories=None,
                has_more=False,
                history=_get_history(session),
                last_shown_product_ids=session.get("last_shown") or [],
                mode="chat",
            ),
        )
        reply = _strip_phone(reply, settings.manager_phone)
        _append_history(session, "assistant", reply)
        store.set_session(message.from_user.id, session)
        await message.answer(reply)
        return

    batch_ids = ids[offset : offset + settings.max_products_per_reply]
    products = await get_products_by_ids(settings, batch_ids)
    if not products:
        reply = await asyncio.to_thread(
            generate_ai_reply,
            settings,
            AiContext(user_text=user_text, products=[], brands=None, categories=None, has_more=False),
        )
        reply = _strip_phone(reply, settings.manager_phone)
        await message.answer(reply)
        return

    if offset == 0:
        has_more = (offset + len(batch_ids)) < len(ids)
        reply = await asyncio.to_thread(
            generate_ai_reply,
            settings,
            AiContext(
                user_text=session.get("last_query") or user_text,
                products=products,
                brands=None,
                categories=None,
                has_more=has_more,
                history=_get_history(session),
                last_shown_product_ids=session.get("last_shown") or [],
                mode="results",
            ),
        )
        reply = _strip_phone(reply, settings.manager_phone)
        _append_history(session, "assistant", reply)
        store.set_session(message.from_user.id, session)
        await message.answer(reply)

    # Save what we show so the user can say "1" / "birinchi".
    session["last_shown"] = [p.id for p in products]
    store.set_session(message.from_user.id, session)

    for p in products:
        caption = format_product_caption(p, settings.manager_phone, max_chars=settings.max_caption_chars)
        await _send_product_simple(
            message=message,
            store=store,
            caption=caption,
            image_urls=p.images,
        )

    new_offset = offset + len(batch_ids)
    session["offset"] = new_offset
    store.set_session(message.from_user.id, session)

    if new_offset < len(ids):
        reply = await asyncio.to_thread(
            generate_ai_reply,
            settings,
            AiContext(
                user_text="Yana variantlar bormi?",
                products=[],
                brands=None,
                categories=None,
                has_more=True,
                history=_get_history(session),
                last_shown_product_ids=session.get("last_shown") or [],
                mode="pagination",
            ),
        )
        await message.answer(reply)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    bot = Bot(token=settings.telegram_token)
    dp = Dispatcher()
    dp.include_router(router)
    logger.info("Starting bot (v1)...")

    async def _cleanup_loop() -> None:
        while True:
            try:
                deleted = store.cleanup_expired_sessions(max_age_seconds=1200)
                if deleted:
                    logger.info("Session cleanup: deleted=%s", deleted)
            except Exception as e:
                logger.warning("Session cleanup failed: %s", e)
            await asyncio.sleep(300)  # every 5 minutes

    asyncio.create_task(_cleanup_loop())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
