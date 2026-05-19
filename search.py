import json
import os
import re
from difflib import SequenceMatcher

from db import fetch_all
from filters import Filters
from products import Product, clean_json_field, product_from_row


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
    tokens = [t for t in q.split() if t and t not in stop]
    q = " ".join(tokens).strip()
    return q


def apply_corrections(q: str) -> str:
    corrections = _get_corrections()
    tokens = [t for t in q.split() if t]
    if corrections:
        tokens = [corrections.get(t, t) for t in tokens]

    # Built-in synonyms (so it doesn't have to be in search_corrections.json)
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

    expanded: list[str] = []
    for t in tokens:
        expanded.append(t)
        for alt in synonyms.get(t, []):
            if alt and alt not in expanded:
                expanded.append(alt)

    return " ".join([t for t in expanded if t])


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _score_match(query: str, blob: str) -> float:
    # Token-average best match + substring boosts
    q_tokens = [t for t in query.split() if t]
    if not q_tokens:
        return 0.0
    blob_l = blob.lower()
    blob_tokens = re.findall(r"[\w\-]+", blob_l, flags=re.UNICODE)
    if not blob_tokens:
        blob_tokens = [blob_l]

    scores: list[float] = []
    for t in q_tokens:
        if t in blob_l:
            scores.append(1.0)
            continue
        best = 0.0
        for w in blob_tokens:
            if abs(len(w) - len(t)) > 8:
                continue
            best = max(best, _ratio(t, w))
            if best >= 0.92:
                break
        scores.append(best)
    base = sum(scores) / max(1, len(scores))
    # Slight boost for longer (more specific) queries
    return base + min(0.15, 0.02 * max(0, len(query) - 6))


def search_products(user_query: str, limit: int = 10) -> list[Product]:
    return search_products_filtered(user_query, Filters(), limit=limit)


def _passes_filters(p: Product, filters: Filters) -> bool:
    price_unit = (os.getenv("SODDA_PRICE_UNIT", "USD") or "USD").upper()
    price_value = int(p.price_uzs or 0)
    price_usd = price_value if price_unit == "USD" else round(price_value / 12000) if price_value else 0

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
    q = apply_corrections(normalize_query(user_query))
    if not q:
        return []

    tokens = [t for t in q.split() if t]
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

    # If SQL token filter returns nothing (typos, translit, etc.), do a small fallback scan
    # and rely on fuzzy ranking to find near matches (still limited to avoid heavy DB load).
    if not rows:
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
        blob = " ".join(
            [
                p.title,
                p.model,
                p.brand,
                p.category,
                p.keywords,
                clean_json_field(p.description),
            ]
        )
        score = _score_match(q, blob)
        ranked.append((score, p))

    ranked.sort(key=lambda x: x[0], reverse=True)

    def _is_specific_query(text: str) -> bool:
        # Specific queries (model/name) should not return unrelated items.
        if any(ch.isdigit() for ch in text):
            return True
        tokens = text.split()
        if len(tokens) >= 3:
            return True
        if len(text) >= 16:
            return True
        return False

    # Hard guard against irrelevant queries (prevents random matches like "book").
    if ranked and ranked[0][0] < 0.45:
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
    strong = [p for s, p in ranked if s >= 0.55]
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
