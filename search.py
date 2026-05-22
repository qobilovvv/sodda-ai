import json
import os
import re
from difflib import SequenceMatcher
import logging

from db import fetch_all
from filters import Filters
from products import Product, clean_json_field, product_from_row

logger = logging.getLogger(__name__)

_CORRECTIONS_PATH = os.path.join(os.path.dirname(__file__), "search_corrections.json")


def _load_corrections() -> dict[str, str]:
    try:
        with open(_CORRECTIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k).lower(): str(v).lower() for k, v in data.items()}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return {}


_CORRECTIONS: dict[str, str] = {}
_CORRECTIONS_MTIME: float | None = None


def _get_corrections() -> dict[str, str]:
    global _CORRECTIONS, _CORRECTIONS_MTIME
    try:
        mtime = os.path.getmtime(_CORRECTIONS_PATH)
    except FileNotFoundError:
        _CORRECTIONS = {}
        _CORRECTIONS_MTIME = None
        return _CORRECTIONS
    except Exception:
        return _CORRECTIONS

    if _CORRECTIONS_MTIME != mtime:
        _CORRECTIONS = _load_corrections()
        _CORRECTIONS_MTIME = mtime
    return _CORRECTIONS


def normalize_query(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"[^\w\s\-]+", " ", q, flags=re.UNICODE)
    q = re.sub(r"\s+", " ", q).strip()
    # Drop common "shop intent" stopwords that shouldn't affect search matching.
    stop = {
        "bormi",
        "bormi?",
        "bor",
        "mi",
        "kerak",
        "narx",
        "qancha",
        "nechi",
        "olmoq",
        "sotib",
        "qidiryapman",
        "qidiraman",
        "menga",
        "menda",
        "bor-mi",
        "есть",
        "естьли",
        "сколько",
        "цена",
        "купить",
        "ищу",
        "нужен",
        "нужна",
    }
    raw_tokens = [t for t in q.split() if t and t not in stop]
    # Light Uzbek plural normalization: televizorlar/televizorlari -> televizor
    tokens: list[str] = []
    for t in raw_tokens:
        if len(t) > 6:
            if t.endswith("lari") and len(t) > 8:
                t = t[: -4]
            elif t.endswith("lar") and len(t) > 7:
                t = t[: -3]
        tokens.append(t)
    q = " ".join(tokens).strip()
    return q


def apply_corrections(q: str) -> str:
    corrections = _get_corrections()
    tokens = [t for t in q.split() if t]
    if corrections:
        tokens = [corrections.get(t, t) for t in tokens]
    return " ".join([t for t in tokens if t])


def expand_synonyms_tokens(tokens: list[str]) -> list[str]:
    """Expands tokens with built-in synonyms for ranking only (not SQL filtering)."""
    synonyms: dict[str, list[str]] = {
        # RU -> UZ/EN
        "пылесос": ["changyutgich", "vacuum", "pilesos"],
        "пылесоса": ["changyutgich", "vacuum", "pilesos"],
        "робот": ["robot"],
        "робот-пылесос": ["robot", "changyutgich", "vacuum", "pilesos"],
        # UZ/RU latin
        "pilesos": ["changyutgich", "vacuum", "пылесос"],
        "robot": ["robot"],
        "changyutgich": ["changyutgich", "pilesos", "пылесос", "vacuum"],
        # Common variants
        "robotpilesos": ["robot", "changyutgich", "pilesos", "пылесос"],
    }

    out: list[str] = []
    for t in tokens:
        if t not in out:
            out.append(t)
        for alt in synonyms.get(t, []):
            if alt and alt not in out:
                out.append(alt)
    return out


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _score_match(query: str, product: Product) -> float:
    """
    Token-average fuzzy match + field-weighted substring bonuses.
    - High bonus if token found in title/model (+2.0)
    - Low bonus if token found in description/keywords (+0.5)
    """
    q_tokens = [t for t in (query or "").split() if t]
    if not q_tokens:
        return 0.0

    title_l = (product.title or "").lower()
    model_l = (product.model or "").lower()
    desc_l = clean_json_field(product.description).lower()
    keywords_l = (product.keywords or "").lower()
    brand_l = (product.brand or "").lower()
    category_l = (product.category or "").lower()
    blob_l = " ".join([title_l, model_l, brand_l, category_l, keywords_l, desc_l]).strip()
    blob_tokens = re.findall(r"[\w\-]+", blob_l, flags=re.UNICODE) or [blob_l]

    scores: list[float] = []
    for t in q_tokens:
        if not t:
            continue
        best = 0.0
        for w in blob_tokens:
            if abs(len(w) - len(t)) > 8:
                continue
            best = max(best, _ratio(t, w))
            if best >= 0.92:
                break

        bonus = 0.0
        if t in title_l or t in model_l:
            bonus += 2.0
        elif t in desc_l or t in keywords_l:
            bonus += 0.5

        scores.append(best + bonus)

    base = sum(scores) / max(1, len(scores))
    return base + min(0.15, 0.02 * max(0, len(query) - 6))


def search_products(user_query: str, limit: int = 10) -> list[Product]:
    return search_products_filtered(user_query, Filters(), limit=limit)


def _passes_filters(p: Product, filters: Filters) -> bool:
    price_unit = (os.getenv("SODDA_PRICE_UNIT", "USD") or "USD").upper()
    price_value = int(p.price_uzs or 0)
    if price_unit == "USD":
        price_usd = price_value
    else:
        price_usd = round(price_value / 12000) if price_value else 0

    if filters.brand:
        if filters.brand.lower() not in (p.brand or "").lower():
            return False

    if filters.min_usd is not None:
        if price_usd < int(filters.min_usd):
            return False
    if filters.max_usd is not None:
        if price_usd > int(filters.max_usd):
            return False

    if filters.size_token:
        blob = " ".join([p.title, p.model, p.description, p.specs, p.keywords]).lower()
        if filters.size_token.lower() not in blob:
            return False

    if filters.size_min is not None or filters.size_max is not None:
        blob = " ".join([p.title, p.model, p.description, p.specs, p.keywords]).lower()
        nums = [int(n) for n in re.findall(r"\b(\d{2,3})\b", blob)]
        if not nums:
            return False
        if filters.size_min is not None and max(nums) < filters.size_min:
            return False
        if filters.size_max is not None and min(nums) > filters.size_max:
            return False

    return True


def search_products_filtered(user_query: str, filters: Filters, limit: int = 10) -> list[Product]:
    q_norm = normalize_query(user_query)
    q = apply_corrections(q_norm)
    if not q:
        return []

    tokens = [t for t in q.split() if t]
    tokens_rank = expand_synonyms_tokens(tokens)

    if os.getenv("DEBUG_SEARCH") == "1":
        logger.info("SEARCH q_raw=%r q_norm=%r q_corr=%r tokens_sql=%s tokens_rank=%s filters=%s", user_query, q_norm, q, tokens[:6], tokens_rank[:12], filters)
    # Broad pre-filter in SQL using token LIKEs, then rank in Python.
    where_parts: list[str] = []
    params: list[str] = []
    for t in tokens[:6]:
        like_t = f"%{t}%"
        where_parts.append(
            "(p.title LIKE %s OR p.keywords LIKE %s OR p.model LIKE %s OR c.title LIKE %s OR b.title LIKE %s)"
        )
        params.extend([like_t, like_t, like_t, like_t, like_t])

    where_sql = " AND ".join(where_parts) if where_parts else "1=1"
    rows = fetch_all(
        f"""
        SELECT
            p.id, p.title, p.sm_desc, p.spec, p.price, p.keywords, p.model, p.stock,
            p.images,
            c.title AS category_title,
            b.title AS brand_title
        FROM products p
        LEFT JOIN categories c ON p.category_id = c.id
        LEFT JOIN brands b ON p.brand_id = b.id
        WHERE p.deleted_at IS NULL AND p.status = 1
          AND ({where_sql})
        LIMIT 250
        """,
        tuple(params),
    )
    fallback_used = False

    if os.getenv("DEBUG_SEARCH") == "1":
        logger.info("SEARCH sql_rows=%d fallback=%s", len(rows), "no" if rows else "yes")

    # If SQL token filter returns nothing (typos, translit, etc.), do a small fallback scan
    # and rely on fuzzy ranking to find near matches (still limited to avoid heavy DB load).
    if not rows:
        fallback_used = True
        rows = fetch_all(
            """
            SELECT
                p.id, p.title, p.sm_desc, p.spec, p.price, p.keywords, p.model, p.stock,
                p.images,
                c.title AS category_title,
                b.title AS brand_title
            FROM products p
            LEFT JOIN categories c ON p.category_id = c.id
            LEFT JOIN brands b ON p.brand_id = b.id
            WHERE p.deleted_at IS NULL AND p.status = 1
            ORDER BY p.id DESC
            LIMIT 500
            """
        )

    products_all = [product_from_row(r) for r in rows]
    products = [p for p in products_all if _passes_filters(p, filters)]
    if os.getenv("DEBUG_SEARCH") == "1":
        logger.info("SEARCH post_filter_count=%d (from %d)", len(products), len(products_all))

    # Exact/substring match boost: if the normalized query is contained in title/model,
    # prefer those results (useful for concrete product names).
    exact: list[Product] = []
    for p in products:
        blob_nm = normalize_query(f"{p.title} {p.model}")
        if q and q in blob_nm:
            exact.append(p)
    if exact:
        return exact[:limit]
    ranked: list[tuple[float, Product]] = []
    for p in products:
        score = _score_match(" ".join(tokens_rank), p)
        ranked.append((score, p))

    if os.getenv("DEBUG_SEARCH") == "1" and ranked:
        logger.info("SEARCH top_score=%.3f top_title=%r top_price=%r", ranked[0][0], ranked[0][1].title, ranked[0][1].price_uzs)

    ranked.sort(key=lambda x: x[0], reverse=True)

    def _is_specific_query(text: str) -> bool:
        # Specific queries are usually model/SKU-like (letters+digits) or long exact names.
        t = text.lower().strip()
        if re.search(r"\b(?=[a-z0-9-]*\d)(?=[a-z0-9-]*[a-z])[a-z0-9-]{5,}\b", t):
            return True
        if re.search(r"\b[a-z0-9]{2,}-[a-z0-9-]{2,}\b", t):
            return True
        tokens = t.split()
        return len(t) >= 28 or len(tokens) >= 5

    # Hard guard against irrelevant queries (prevents random matches like "book").
    if ranked and ranked[0][0] < 0.65:
        return []

    # If we had to use broad fallback scan, require strong evidence to avoid random items.
    if fallback_used and ranked:
        qtoks = tokens[:2]
        top = ranked[0][1]
        blob = normalize_query(f"{top.title} {top.model} {top.brand} {top.category} {top.keywords}")
        has_substring = any(t in blob for t in qtoks if t)
        if not has_substring and ranked[0][0] < 0.8:
            return []

    # For single-word queries, be stricter: avoid returning unrelated products.
    if ranked and len(tokens) == 1:
        qt = tokens[0]
        top = ranked[0][1]
        blob = normalize_query(f"{top.title} {top.model} {top.brand} {top.category} {top.keywords}")
        has_substring = qt in blob
        if not has_substring and ranked[0][0] < 0.75:
            return []

    # For short 2-word queries, require a bit more confidence to avoid unrelated products.
    if ranked and len(tokens) <= 2 and ranked[0][0] < 0.6:
        return []

    # Stricter guard for "specific" queries: if user typed a concrete name/model and
    # we still didn't get a strong match, return nothing (avoid false positives).
    if ranked and _is_specific_query(q) and ranked[0][0] < 0.75:
        return []

    # Return only one item only for truly specific queries (model/name-like), not generic ones
    # like "Samsung TV" which should show multiple options.
    if ranked and _is_specific_query(q) and ranked[0][0] >= 0.92:
        return [ranked[0][1]]

    # Filter out very weak matches, but keep at least a few results if any exist.
    strong = [p for s, p in ranked if s >= 0.75]
    if strong:
        return strong[:limit]
    return [p for _, p in ranked[:limit]]


def get_product_by_id(product_id: int) -> Product | None:
    row = fetch_all(
        """
        SELECT
            p.id, p.title, p.sm_desc, p.spec, p.price, p.keywords, p.model, p.stock,
            p.images,
            c.title AS category_title,
            b.title AS brand_title
        FROM products p
        LEFT JOIN categories c ON p.category_id = c.id
        LEFT JOIN brands b ON p.brand_id = b.id
        WHERE p.id = %s AND p.deleted_at IS NULL
        LIMIT 1
        """,
        (int(product_id),),
    )
    if not row:
        return None
    return product_from_row(row[0])
