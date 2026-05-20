import asyncio
import logging
import os
import re
import time

import aiohttp
import dataclasses
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.chat_action import ChatActionSender
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv

from ai import ask_ai_about_product, general_chat, get_session_history, summarize_product, summarize_product_50w
from products import Product, format_price_ui, truncate
from search import apply_corrections, get_product_by_id, normalize_query, search_products_filtered
from router import route_user_message
from filters import Filters, parse_budget_usd, parse_size_range, parse_size_token
from tools import tool_brand_options_for_query


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
dp = Dispatcher()

active_chats: dict[str, float] = {}
selected_product_by_chat: dict[str, int] = {}
refine_state_by_chat: dict[str, dict] = {}
product_list_summary_cache: dict[int, str] = {}
last_search_context_by_chat: dict[str, dict] = {}


def _with_description(p: Product, desc: str) -> Product:
    return dataclasses.replace(p, description=desc)


def _looks_like_specific_product_query(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    q_l = q.lower()

    # If it looks like a model/SKU code (alphanumeric with both letters and digits), treat as specific.
    # Examples: v10213242, a55, prd-1234, 55au7100, tsf01pkeu, rf295cd-mbg/hf etc.
    if re.search(r"\b(?=[a-z0-9-]*\d)(?=[a-z0-9-]*[a-z])[a-z0-9-]{5,}\b", q_l):
        return True
    if re.search(r"\b[a-z0-9]{2,}-[a-z0-9-]{2,}\b", q_l):
        return True

    tokens = [t for t in q_l.split() if t]

    # Without a model/SKU-like pattern, short queries should be treated as broad.
    # Examples: "televizor samsung", "toster", "kir mashina" => not specific.
    if len(tokens) <= 3:
        return False

    # Longer, detailed names (but without model code) can still be specific.
    return len(q) >= 28 or len(tokens) >= 5


def _looks_like_shop_query(text: str) -> bool:
    t = (text or "").lower()
    keywords = [
        "bormi",
        "bor mi",
        "narx",
        "price",
        "сколько",
        "цена",
        "есть",
        "mavjud",
        "sotib",
        "buy",
        "olmoq",
        "kerak",
        "qidir",
        "ищу",
        "поиск",
        "$",
    ]
    return any(k in t for k in keywords)


def _detect_brand_from_text(text: str, brand_options: list[str]) -> str | None:
    txt = (text or "").lower()
    for b in brand_options or []:
        bl = (b or "").lower().strip()
        if not bl:
            continue
        # prefer whole-word-ish match, but allow substring for short brand names
        if bl in txt:
            return b
    return None


def _looks_like_brand_only_message(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    # A single short token (e.g., "LG", "Samsung", "Bosch") is likely a brand follow-up.
    tokens = t.split()
    if len(tokens) != 1:
        return False
    tok = tokens[0]
    return 2 <= len(tok) <= 20


def _extract_brand_candidate(text: str) -> str | None:
    """
    Extract brand-like token from short follow-ups like:
      - "premier"
      - "premier?"
      - "premier chi?"
      - "premierchi"
    """
    t = (text or "").lower().strip()
    if not t:
        return None
    t = re.sub(r"[^\w\s]+", " ", t, flags=re.UNICODE).strip()
    if not t:
        return None
    m = re.match(r"^(\w+)\s*chi$", t)
    if m:
        return m.group(1)
    tokens = [x for x in t.split() if x]
    if not tokens:
        return None
    # If user wrote two tokens like "premier chi", take the first.
    if len(tokens) == 2 and tokens[1] == "chi":
        return tokens[0]
    if len(tokens) == 1:
        return tokens[0]
    return None


async def _try_brand_followup(message: types.Message) -> bool:
    """
    If user sends a brand-like follow-up and we have last context, start budget step.
    Returns True if handled.
    """
    chat_id = str(message.chat.id)
    if chat_id not in last_search_context_by_chat:
        return False
    cand = _extract_brand_candidate(message.text or "")
    if not cand:
        return False
    prev = last_search_context_by_chat[chat_id]
    base_query = prev.get("query") or ""
    if not base_query:
        return False
    brands = await asyncio.to_thread(tool_brand_options_for_query, base_query, 80)
    detected = _normalize_brand(cand, brands)
    # Only treat as a brand follow-up if it matches a known brand reasonably.
    if not detected:
        return False
    refine_state_by_chat[chat_id] = {
        "query": base_query,
        "brand": detected,
        "size_token": prev.get("filters", {}).get("size_token"),
        "stage": "budget",
        "brand_options": brands,
    }
    await safe_send_message(
        message,
        f"OK, brend: <b>{detected}</b>.\nByudjet (USD): masalan <b>300$ gacha</b>.",
        parse_mode="HTML",
    )
    return True


def _normalize_brand(brand: str | None, brand_options: list[str]) -> str | None:
    """
    Normalizes user-provided brand using search corrections and maps it to the closest offered brand.
    """
    if not brand:
        return None
    b = apply_corrections(normalize_query(brand)).strip()
    if not b:
        return None

    # Try direct/substring match against offered options
    for opt in brand_options or []:
        ol = (opt or "").lower().strip()
        if not ol:
            continue
        if b == ol or b in ol or ol in b:
            return opt

    # Fuzzy fallback (cheap): choose the best ratio
    try:
        import difflib

        best_opt = None
        best_score = 0.0
        for opt in brand_options or []:
            ol = (opt or "").lower().strip()
            if not ol:
                continue
            score = difflib.SequenceMatcher(None, b, ol).ratio()
            if score > best_score:
                best_score = score
                best_opt = opt
        if best_opt and best_score >= 0.8:
            return best_opt
    except Exception:
        pass

    return b


def cleanup_old_chats():
    current_time = time.time()
    expired: list[str] = []
    for chat_id, last_active_time in active_chats.items():
        if current_time - last_active_time > 1200:
            expired.append(chat_id)

    for chat_id in expired:
        try:
            get_session_history(chat_id).clear()
            active_chats.pop(chat_id, None)
            selected_product_by_chat.pop(chat_id, None)
            logging.info(f"🧹 Inactivity timeout: History for {chat_id} has been cleared.")
        except Exception as e:
            logging.error(f"Error clearing expired history for {chat_id}: {e}")


async def get_valid_image_url(url: str) -> str | None:
    filename = url.split("/")[-1]
    thumb_url = f"https://sodda.uz/storage/products/thumbs/{filename}"
    main_url = f"https://sodda.uz/storage/products/{filename}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.head(thumb_url, timeout=2) as resp:
                if resp.status == 200:
                    return thumb_url
        except Exception:
            pass
        try:
            async with session.head(main_url, timeout=2) as resp:
                if resp.status == 200:
                    return main_url
        except Exception:
            pass
    return None


def convert_markdown_to_html(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.*?)__", r"<i>\1</i>", text)
    return text


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)

def sanitize_telegram_html(text: str) -> str:
    """
    Telegram HTML supports only a limited set of tags.
    We allow only <b> and <i> and strip everything else (e.g., <ul>, <li>, <br>).
    """
    if not text:
        return ""
    # Remove all tags except b/i (both opening and closing).
    text = re.sub(r"</?(?!b\b|i\b)[a-zA-Z0-9_:-]+[^>]*>", "", text)
    return text


async def safe_send_message(
    message: types.Message,
    text: str,
    parse_mode: str = "HTML",
    reply_markup: InlineKeyboardMarkup | None = None,
):
    try:
        await message.answer(text, parse_mode=parse_mode, reply_markup=reply_markup)
    except Exception as e:
        if "can't parse entities" in str(e):
            await message.answer(strip_html(text), parse_mode=None, reply_markup=reply_markup)
        else:
            raise


def _chunk_lines(lines: list[str], header: str, max_chars: int = 3800) -> list[str]:
    chunks: list[str] = []
    cur = header.strip()
    for line in lines:
        add = ("\n" + line) if cur else line
        if len(cur) + len(add) > max_chars:
            chunks.append(cur)
            cur = header.strip() + "\n" + line
        else:
            cur += add
    if cur:
        chunks.append(cur)
    return chunks


async def safe_send_photo(
    message: types.Message,
    photo: str,
    caption: str,
    reply_markup: InlineKeyboardMarkup | None = None,
    parse_mode: str = "HTML",
):
    try:
        await message.answer_photo(photo=photo, caption=caption, parse_mode=parse_mode, reply_markup=reply_markup)
    except Exception as e:
        if "can't parse entities" in str(e):
            await message.answer_photo(photo=photo, caption=strip_html(caption), parse_mode=None, reply_markup=reply_markup)
        else:
            raise


def product_keyboard(product_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Batafsil / Savol berish", callback_data=f"p:{product_id}")],
        ]
    )


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⬅️ Qidiruvga qaytish", callback_data="back")],
        ]
    )

def product_chat_keyboard(product_id: int) -> InlineKeyboardMarkup:
    # User requested to remove extra buttons; keep only back-to-search.
    return back_keyboard()


def brand_keyboard(brands: list[str]) -> InlineKeyboardMarkup | None:
    if not brands:
        return None
    rows = []
    for b in brands[:8]:
        rows.append([InlineKeyboardButton(text=b, callback_data=f"brand:{b[:40]}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def format_product_caption(p: Product) -> str:
    price_txt = format_price_ui(p.price_uzs)
    title = p.title + (f" ({p.model})" if p.model and p.model not in p.title else "")
    desc = truncate(p.description, 520) if p.description else ""
    specs_lines = _spec_lines(p.specs, 6) if p.specs else []
    specs_block = "\n".join([f"• {line}" for line in specs_lines]) if specs_lines else "—"
    return (
        f"<b>{title}</b>\n"
        f"💰 Narxi: <b>{price_txt}</b>\n\n"
        f"<b>Tavsif:</b> {desc if desc else '—'}\n\n"
        f"<b>Asosiy xarakteristikalari:</b>\n{specs_block}\n\n"
        f"<i>Batafsil ma'lumot yoki savol uchun pastdagi tugmani bosing.</i>\n"
        f"📞 +998950001234"
    )


def _spec_lines(specs: str, max_lines: int = 6) -> list[str]:
    if not specs:
        return []
    s = re.sub(r"\s+", " ", specs).strip()
    # Split on common separators while keeping key:value pairs.
    parts = re.split(r"\s*[;,\n]\s*", s)
    parts = [p.strip("•- \t") for p in parts if p and p.strip()]
    kv = [p for p in parts if ":" in p]
    chosen = kv[:max_lines] if kv else parts[:max_lines]
    # Keep lines short for Telegram captions
    return [truncate(line, 90) for line in chosen]


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Assalomu alaykum! Sodda.uz botiga xush kelibsiz.\n"
        "Mahsulot qidirish uchun nomini yozing (masalan: 'televizor' yoki 'muzlatgich').\n"
        "Topilgan mahsulotlardan birini tanlab, keyin savol berishingiz mumkin."
    )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(
        "Yordam:\n"
        "1) Qidirish: mahsulot nomi/turi yozing.\n"
        "2) Batafsil: mahsulot ostidagi tugmani bosing.\n"
        "3) /clear - suhbat tarixini tozalash.\n"
        "4) Qidiruvga qaytish uchun: '⬅️ Qidiruvga qaytish' tugmasi."
    )


@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    chat_id = str(message.chat.id)
    get_session_history(chat_id).clear()
    selected_product_by_chat.pop(chat_id, None)
    refine_state_by_chat.pop(chat_id, None)
    active_chats.pop(chat_id, None)
    await message.answer("Suhbat tarixi tozalandi!")


@dp.callback_query(F.data == "back")
async def cb_back(callback: types.CallbackQuery):
    chat_id = str(callback.message.chat.id)
    selected_product_by_chat.pop(chat_id, None)
    refine_state_by_chat.pop(chat_id, None)
    await callback.answer()
    await callback.message.answer("Qidirish uchun mahsulot nomini yozing.")

@dp.callback_query(F.data.startswith("brand:"))
async def cb_choose_brand(callback: types.CallbackQuery):
    chat_id = str(callback.message.chat.id)
    state = refine_state_by_chat.get(chat_id)
    if not state:
        await callback.answer()
        return
    brand = callback.data.split(":", 1)[1]
    state["brand"] = brand
    state["stage"] = "budget"
    refine_state_by_chat[chat_id] = state
    await callback.answer("OK")
    await callback.message.answer(
        "Byudjetingizni <b>USD</b> da yozing (masalan: <b>300$ gacha</b> yoki <b>300$ dan yuqori</b>).",
        parse_mode="HTML",
    )


@dp.callback_query(F.data.startswith("p:"))
async def cb_select_product(callback: types.CallbackQuery):
    chat_id = str(callback.message.chat.id)
    try:
        product_id = int(callback.data.split(":", 1)[1])
    except Exception:
        await callback.answer("Xatolik", show_alert=False)
        return

    product = await asyncio.to_thread(get_product_by_id, product_id)
    if not product:
        await callback.answer("Mahsulot topilmadi", show_alert=False)
        return

    selected_product_by_chat[chat_id] = product_id
    await callback.answer("Tanlandi")
    # Send AI-generated product card based on description + specs (only for the selected product).
    card = await asyncio.to_thread(summarize_product, chat_id, product)
    card = convert_markdown_to_html(card)
    card = sanitize_telegram_html(card)
    await safe_send_message(callback.message, card, parse_mode="HTML", reply_markup=product_chat_keyboard(product_id))


@dp.callback_query(F.data.startswith("pi:"))
async def cb_product_info(callback: types.CallbackQuery):
    # Quick-action buttons removed; ignore.
    await callback.answer()
    return


@dp.message()
async def handle_message(message: types.Message):
    if not message.text:
        return

    chat_id = str(message.chat.id)
    active_chats[chat_id] = time.time()

    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        try:
            # Brand follow-up like "premier chi?" should reuse last search context.
            if await _try_brand_followup(message):
                return

            # If user is in "product chat" mode, send only selected product to AI.
            if chat_id in selected_product_by_chat:
                product_id = selected_product_by_chat[chat_id]
                product = await asyncio.to_thread(get_product_by_id, product_id)
                if not product:
                    selected_product_by_chat.pop(chat_id, None)
                    await safe_send_message(message, "Tanlangan mahsulot topilmadi. Qaytadan qidiring.")
                    return

                answer = await asyncio.to_thread(ask_ai_about_product, message.text, chat_id, product)
                answer = convert_markdown_to_html(answer)
                answer = sanitize_telegram_html(answer)
                # Always include an exit button so user can return to normal search mode.
                await safe_send_message(message, answer, parse_mode="HTML", reply_markup=back_keyboard())
                return

            # If user is in refinement flow, parse filters and run filtered search.
            if chat_id in refine_state_by_chat:
                state = refine_state_by_chat[chat_id]
                stage = state.get("stage", "brand")
                # If user typed a budget directly, allow skipping brand stage.
                min_usd_try, max_usd_try = parse_budget_usd(message.text)
                looks_like_budget = min_usd_try is not None or max_usd_try is not None
                offered: list[str] = state.get("brand_options") or []

                # If user sent budget + brand together (e.g. "500$ samsung"), capture brand automatically.
                if looks_like_budget and not state.get("brand"):
                    detected = _detect_brand_from_text(message.text, offered)
                    if detected:
                        state["brand"] = _normalize_brand(detected, offered)
                        refine_state_by_chat[chat_id] = state

                if stage == "brand" and not looks_like_budget:
                    txt = (message.text or "").strip()
                    # Accept typed brand if it matches offered brands.
                    chosen = None
                    for b in offered:
                        if txt.lower() == b.lower() or txt.lower() in b.lower():
                            chosen = b
                            break
                    state["brand"] = _normalize_brand(chosen or txt, offered)
                    state["stage"] = "budget"
                    refine_state_by_chat[chat_id] = state
                    await safe_send_message(
                        message,
                        f"OK, brend: <b>{state['brand']}</b>.\nEndi byudjetingizni USD da yozing (masalan: <b>300$ gacha</b>).",
                        parse_mode="HTML",
                    )
                    return

                filters = Filters()
                filters.brand = _normalize_brand(state.get("brand"), offered)
                min_usd, max_usd = (min_usd_try, max_usd_try) if looks_like_budget else parse_budget_usd(message.text)
                filters.min_usd = min_usd
                filters.max_usd = max_usd
                filters.size_token = parse_size_token(message.text) or state.get("size_token")
                smin, smax = parse_size_range(message.text)
                filters.size_min = smin
                filters.size_max = smax

                query = state.get("query") or message.text
                products_full = await asyncio.to_thread(search_products_filtered, query, filters, 10)
                refine_state_by_chat.pop(chat_id, None)

                if not products_full:
                    await safe_send_message(message, "Hech narsa topilmadi. Byudjet yoki brendni o'zgartirib ko'ring.")
                    return

                await safe_send_message(message, f"Topilgan mahsulotlar: <b>{len(products_full)}</b> ta", parse_mode="HTML")
                await asyncio.sleep(0.2)

                for p in products_full:
                    if p.id in product_list_summary_cache:
                        p = _with_description(p, product_list_summary_cache[p.id])
                    else:
                        short_desc = await asyncio.to_thread(summarize_product_50w, p)
                        short_desc = convert_markdown_to_html(short_desc)
                        short_desc = sanitize_telegram_html(short_desc)
                        product_list_summary_cache[p.id] = short_desc
                        p = _with_description(p, short_desc)
                    caption = format_product_caption(p)
                    kb = product_keyboard(p.id)
                    image_urls: list[str] = []
                    if p.image_urls:
                        tasks = [get_valid_image_url(u) for u in p.image_urls[:6]]
                        results = await asyncio.gather(*tasks)
                        image_urls = [u for u in results if u]

                    if image_urls:
                        # Keep the button on the product message: send only the first image with keyboard.
                        await safe_send_photo(message, image_urls[0], caption, reply_markup=kb)
                    else:
                        await safe_send_message(message, caption, reply_markup=kb)
                    await asyncio.sleep(0.35)
                return

            # Otherwise, treat as search query.
            routed = await asyncio.to_thread(route_user_message, message.text)

            if routed.get("type") == "greeting":
                await safe_send_message(message, routed.get("text", ""), parse_mode="HTML")
                return
            if routed.get("type") == "advice":
                await safe_send_message(message, routed.get("text", ""), parse_mode="HTML")
                return
            if routed.get("type") == "chat":
                # One more chance: if user message looks like shopping intent, try DB search before chatting.
                if _looks_like_shop_query(message.text):
                    q_text = (message.text or "").strip()
                    q_clean = normalize_query(q_text) or q_text
                    products_try = await asyncio.to_thread(search_products_filtered, q_clean, Filters(), 10)
                    if products_try:
                        last_search_context_by_chat[chat_id] = {"query": q_clean, "filters": {}}
                        await safe_send_message(
                            message, f"Topilgan mahsulotlar: <b>{len(products_try)}</b> ta", parse_mode="HTML"
                        )
                        await asyncio.sleep(0.2)
                        for p in products_try:
                            if p.id in product_list_summary_cache:
                                p = _with_description(p, product_list_summary_cache[p.id])
                            else:
                                short_desc = await asyncio.to_thread(summarize_product_50w, p)
                                short_desc = convert_markdown_to_html(short_desc)
                                short_desc = sanitize_telegram_html(short_desc)
                                product_list_summary_cache[p.id] = short_desc
                                p = _with_description(p, short_desc)
                            caption = format_product_caption(p)
                            kb = product_keyboard(p.id)
                            image_urls: list[str] = []
                            if p.image_urls:
                                tasks = [get_valid_image_url(u) for u in p.image_urls[:6]]
                                results = await asyncio.gather(*tasks)
                                image_urls = [u for u in results if u]
                            if image_urls:
                                await safe_send_photo(message, image_urls[0], caption, reply_markup=kb)
                            else:
                                await safe_send_message(message, caption, parse_mode="HTML", reply_markup=kb)
                            await asyncio.sleep(0.35)
                        return

                answer = await asyncio.to_thread(general_chat, chat_id, routed.get("text") or message.text)
                answer = convert_markdown_to_html(answer)
                answer = sanitize_telegram_html(answer)
                await safe_send_message(message, answer, parse_mode="HTML")
                return
            if routed.get("type") == "intent_search":
                q_text = (routed.get("query") or message.text or "").strip()
                f = routed.get("filters") or {}
                filters = Filters()
                # Normalize brand early using corrections; refined later against offered brands.
                filters.brand = apply_corrections(normalize_query(f.get("brand") or "")).strip() or None
                try:
                    filters.min_usd = int(f["min_usd"]) if f.get("min_usd") is not None else None
                except Exception:
                    filters.min_usd = None
                try:
                    filters.max_usd = int(f["max_usd"]) if f.get("max_usd") is not None else None
                except Exception:
                    filters.max_usd = None
                filters.size_token = f.get("size_token") or parse_size_token(q_text)

                q_clean = normalize_query(q_text) or q_text

                # If user didn't provide budget/brand and query is generic, keep the refine UX.
                missing_budget = filters.min_usd is None and filters.max_usd is None
                if missing_budget and not _looks_like_specific_product_query(q_text):
                    has_match = await asyncio.to_thread(search_products_filtered, q_clean, Filters(), 1)
                    if not has_match:
                        await safe_send_message(
                            message,
                            "Afsus, bu so'rov bo'yicha do'konimizda mahsulot topilmadi. Boshqa nom bilan urinib ko'ring.",
                        )
                        return
                    brands = await asyncio.to_thread(tool_brand_options_for_query, q_clean, 8)
                    refine_state_by_chat[chat_id] = {
                        "query": q_clean,
                        "brand": _normalize_brand(filters.brand, brands),
                        "size_token": parse_size_token(q_text),
                        "stage": "brand",
                        "brand_options": brands,
                    }
                    kb = brand_keyboard(brands)
                    brand_line = ""
                    if brands:
                        brand_line = "\n".join([f"• {b}" for b in brands[:8]])
                        brand_line = f"\n\n<b>Bu tur uchun mavjud brendlar:</b>\n{brand_line}"
                    extra = "" if kb else "\n\nByudjet (USD): masalan <b>300$ gacha</b>."
                    await message.answer(
                        "Zo'r! Bu turdagi mahsulotlar bizda bor.\n"
                        "Eng mos variantni topish uchun byudjetingizni aniqlashtiraylik.\n"
                        "Byudjet (USD): masalan <b>300$ gacha</b> yoki <b>300$ dan yuqori</b>."
                        + brand_line
                        + extra,
                        parse_mode="HTML",
                        reply_markup=kb,
                    )
                    return

                products = await asyncio.to_thread(search_products_filtered, q_clean, filters, 10)
                if not products:
                    await safe_send_message(
                        message,
                        "Afsus, bu so'rov bo'yicha do'konimizda mahsulot topilmadi. Boshqa nom bilan urinib ko'ring.",
                    )
                    return
                last_search_context_by_chat[chat_id] = {
                    "query": q_clean,
                    "filters": {"brand": filters.brand, "size_token": filters.size_token},
                }
                await safe_send_message(message, f"Topilgan mahsulotlar: <b>{len(products)}</b> ta", parse_mode="HTML")
                await asyncio.sleep(0.2)
                for p in products:
                    if p.id in product_list_summary_cache:
                        p = _with_description(p, product_list_summary_cache[p.id])
                    else:
                        short_desc = await asyncio.to_thread(summarize_product_50w, p)
                        short_desc = convert_markdown_to_html(short_desc)
                        short_desc = sanitize_telegram_html(short_desc)
                        product_list_summary_cache[p.id] = short_desc
                        p = _with_description(p, short_desc)
                    caption = format_product_caption(p)
                    kb = product_keyboard(p.id)
                    image_urls: list[str] = []
                    if p.image_urls:
                        tasks = [get_valid_image_url(u) for u in p.image_urls[:6]]
                        results = await asyncio.gather(*tasks)
                        image_urls = [u for u in results if u]
                    if image_urls:
                        # To keep the button on the product message, send only the first image with the keyboard.
                        await safe_send_photo(message, image_urls[0], caption, reply_markup=kb)
                    else:
                        await safe_send_message(message, caption, parse_mode="HTML", reply_markup=kb)
                    await asyncio.sleep(0.35)
                return

            if routed.get("type") in ("categories", "brands"):
                items = routed.get("data") or []
                if not items:
                    await safe_send_message(message, "Hech narsa topilmadi.")
                    return
                title = "Kategoriyalar" if routed["type"] == "categories" else "Brendlar"
                lines = [f"• {it.get('title','')}" for it in items if it.get("title")]
                header = f"<b>{title}:</b> (jami: <b>{len(lines)}</b>)"
                chunks = _chunk_lines(lines, header)
                for i, ch in enumerate(chunks):
                    tail = "\n\nMahsulot qidirish uchun nomini yozing." if i == len(chunks) - 1 else ""
                    await safe_send_message(message, ch + tail, parse_mode="HTML")
                    await asyncio.sleep(0.15)
                return

            # Products flow
            data = routed.get("data") or []
            if not data:
                # If this looks like a shop/product intent but nothing matched, say not found.
                if _looks_like_shop_query(message.text):
                    await safe_send_message(
                        message,
                        "Afsus, bu so'rov bo'yicha do'konimizda mahsulot topilmadi. Boshqa nom bilan urinib ko'ring.",
                    )
                else:
                    # Otherwise treat as free-form chat.
                    answer = await asyncio.to_thread(general_chat, chat_id, message.text)
                    answer = convert_markdown_to_html(answer)
                    answer = sanitize_telegram_html(answer)
                    await safe_send_message(message, answer, parse_mode="HTML")
                return

            # Before listing, ask brand + budget to refine (for broad queries).
            q_text = (message.text or "").strip()
            q_clean = normalize_query(q_text)

            # Brand-only follow-up: reuse last query/type from context
            if _looks_like_brand_only_message(q_text) and chat_id in last_search_context_by_chat:
                prev = last_search_context_by_chat[chat_id]
                base_query = prev.get("query") or ""
                if base_query:
                    brands = await asyncio.to_thread(tool_brand_options_for_query, base_query, 50)
                    detected = _normalize_brand(q_text, brands)
                    if detected:
                        refine_state_by_chat[chat_id] = {
                            "query": base_query,
                            "brand": detected,
                            "size_token": prev.get("filters", {}).get("size_token"),
                            "stage": "budget",
                            "brand_options": brands,
                        }
                        await safe_send_message(
                            message,
                            f"OK, brend: <b>{detected}</b>.\nByudjet (USD): masalan <b>300$ gacha</b>.",
                            parse_mode="HTML",
                        )
                        return
            if len(q_text.split()) <= 3:
                # Only start refinement if we actually have matches for this query.
                has_match = await asyncio.to_thread(search_products_filtered, q_clean or q_text, Filters(), 1)
                if not has_match:
                    await safe_send_message(
                        message,
                        "Afsus, bu so'rov bo'yicha do'konimizda mahsulot topilmadi. Boshqa nom bilan urinib ko'ring.",
                    )
                    return
                # If query looks like an exact/specific product, show results immediately (no brand/budget questions).
                if _looks_like_specific_product_query(q_text):
                    products_full = await asyncio.to_thread(search_products_filtered, q_text, Filters(), 1)
                    if not products_full:
                        await safe_send_message(message, "Hech narsa topilmadi. Boshqa so'z bilan urinib ko'ring.")
                        return
                    await safe_send_message(message, "<b>Topilgan mahsulot:</b>", parse_mode="HTML")
                    await asyncio.sleep(0.2)
                    for p in products_full:
                        if p.id in product_list_summary_cache:
                            p = _with_description(p, product_list_summary_cache[p.id])
                        else:
                            short_desc = await asyncio.to_thread(summarize_product_50w, p)
                            short_desc = convert_markdown_to_html(short_desc)
                            short_desc = sanitize_telegram_html(short_desc)
                            product_list_summary_cache[p.id] = short_desc
                            p = _with_description(p, short_desc)
                        caption = format_product_caption(p)
                        kb = product_keyboard(p.id)
                        image_urls: list[str] = []
                        if p.image_urls:
                            tasks = [get_valid_image_url(u) for u in p.image_urls[:6]]
                            results = await asyncio.gather(*tasks)
                            image_urls = [u for u in results if u]
                        if image_urls:
                            await safe_send_photo(message, image_urls[0], caption, reply_markup=kb)
                        else:
                            await safe_send_message(message, caption, parse_mode="HTML", reply_markup=kb)
                        await asyncio.sleep(0.35)
                    return
                else:
                    brands = await asyncio.to_thread(tool_brand_options_for_query, q_clean or q_text, 8)
                    refine_state_by_chat[chat_id] = {
                        "query": q_clean or q_text,
                        "brand": None,
                        "size_token": parse_size_token(q_text),
                        "stage": "brand",
                        "brand_options": brands,
                    }
                    kb = brand_keyboard(brands)
                    brand_line = ""
                    if brands:
                        brand_line = "\n".join([f"• {b}" for b in brands[:8]])
                        brand_line = f"\n\n<b>Bu tur uchun mavjud brendlar:</b>\n{brand_line}"
                    # Single message: brand is optional; user can just type budget.
                    extra = "" if kb else "\n\nByudjet (USD): masalan <b>300$ gacha</b>."
                    await message.answer(
                        "Zo'r! Bu turdagi mahsulotlar bizda bor.\n"
                        "Eng mos variantni topish uchun 2 ta narsani aniqlashtiraylik:\n"
                        "1) Qaysi <b>brend</b> xohlaysiz? (xohlasangiz tanlang, bo'lmasa o'tkazib yuboring)\n"
                        "2) Byudjet (USD): masalan <b>300$ gacha</b>\n\n"
                        "<i>Agar brend tanlamasangiz, byudjetni yozishingiz kifoya — qolganini o'zim topib beraman.</i>"
                        + brand_line
                        + extra,
                        parse_mode="HTML",
                        reply_markup=kb,
                    )
                    return

            # For listing, fetch full products (desc/specs) directly from DB (no AI).
            products = await asyncio.to_thread(search_products_filtered, q_clean or q_text, Filters(), 10)
            if not products:
                if _looks_like_shop_query(message.text):
                    await safe_send_message(
                        message,
                        "Afsus, bu so'rov bo'yicha do'konimizda mahsulot topilmadi. Boshqa nom bilan urinib ko'ring.",
                    )
                else:
                    answer = await asyncio.to_thread(general_chat, chat_id, message.text)
                    answer = convert_markdown_to_html(answer)
                    answer = sanitize_telegram_html(answer)
                    await safe_send_message(message, answer, parse_mode="HTML")
                return
            last_search_context_by_chat[chat_id] = {"query": q_clean or q_text, "filters": {}}

            await safe_send_message(message, f"Topilgan mahsulotlar: <b>{len(products)}</b> ta", parse_mode="HTML")
            await asyncio.sleep(0.2)

            for p in products:
                if p.id in product_list_summary_cache:
                    p = _with_description(p, product_list_summary_cache[p.id])
                else:
                    short_desc = await asyncio.to_thread(summarize_product_50w, p)
                    short_desc = convert_markdown_to_html(short_desc)
                    short_desc = sanitize_telegram_html(short_desc)
                    product_list_summary_cache[p.id] = short_desc
                    p = _with_description(p, short_desc)
                caption = format_product_caption(p)
                kb = product_keyboard(p.id)

                image_urls: list[str] = []
                if p.image_urls:
                    tasks = [get_valid_image_url(u) for u in p.image_urls[:6]]
                    results = await asyncio.gather(*tasks)
                    image_urls = [u for u in results if u]

                if image_urls:
                    await safe_send_photo(message, image_urls[0], caption, reply_markup=kb)
                else:
                    await safe_send_message(message, caption, reply_markup=kb)

                await asyncio.sleep(0.35)

        except Exception as e:
            logging.error(f"Error handling message: {e}")
            await message.answer("Kechirasiz, texnik nosozlik yuz berdi. Birozdan so'ng urinib ko'ring.")


async def main():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(cleanup_old_chats, "interval", minutes=5)
    scheduler.start()
    logging.info("🚀 v1 bot is starting...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
