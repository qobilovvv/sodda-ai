import json
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .db import fetch_all_async


def clean_json_field(data: Any) -> str:
    if not data:
        return ""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data

    if isinstance(data, list):
        # Common `spec` format: list of dicts with *_uz / *_ru keys.
        pairs: list[str] = []
        for item in data:
            if isinstance(item, dict):
                title = item.get("title_uz") or item.get("title_ru") or item.get("title") or ""
                value = item.get("value_uz") or item.get("value_ru") or item.get("value") or ""
                title = str(title).strip()
                value = str(value).strip()
                if title and value:
                    pairs.append(f"{title}: {value}")
            elif item:
                pairs.append(str(item))
        return ", ".join(pairs)

    if isinstance(data, dict):
        content = data.get("uz", data.get("ru", next(iter(data.values())) if data else ""))
        if isinstance(content, dict):
            return ", ".join([f"{k}: {v}" for k, v in content.items()])
        return str(content)
    return str(data)


def extract_image_urls(image_data: Any) -> list[str]:
    if not image_data:
        return []
    try:
        imgs = json.loads(image_data) if isinstance(image_data, str) else image_data
        if isinstance(imgs, list) and imgs:
            urls: list[str] = []
            for img in imgs:
                if not isinstance(img, str):
                    continue
                clean_name = img.replace("products/", "").replace("thumbs/", "").strip("/")
                urls.append(f"https://sodda.uz/storage/products/{clean_name}")
            return urls
    except Exception:
        return []
    return []


@dataclass(frozen=True)
class Product:
    id: int
    title: str
    description: str
    characteristics: str
    brand: str
    category: str
    price_usd: float
    price_uzs: int
    stock: int | None
    model: str
    images: list[str]


async def get_products_by_ids(settings: Settings, product_ids: list[int]) -> list[Product]:
    if not product_ids:
        return []

    placeholders = ",".join(["%s"] * len(product_ids))
    query = f"""
        SELECT
            p.id, p.title, p.sm_desc, p.spec, p.price, p.model, p.stock,
            p.images, p.images_thumb,
            c.title AS category_title,
            b.title AS brand_title
        FROM products p
        LEFT JOIN categories c ON p.category_id = c.id
        LEFT JOIN brands b ON p.brand_id = b.id
        WHERE p.deleted_at IS NULL AND p.status = 1 AND p.id IN ({placeholders})
    """
    rows = await fetch_all_async(settings, query, tuple(product_ids))

    by_id: dict[int, Product] = {}
    for r in rows:
        title = clean_json_field(r.get("title"))
        brand = clean_json_field(r.get("brand_title"))
        category = clean_json_field(r.get("category_title"))
        characteristics = clean_json_field(r.get("spec"))
        description = clean_json_field(r.get("sm_desc"))
        images = extract_image_urls(r.get("images"))

        price_usd = float(r.get("price") or 0.0)
        price_uzs = int(round(price_usd * settings.usd_to_uzs)) if price_usd else 0

        pid = int(r["id"])
        by_id[pid] = Product(
            id=pid,
            title=title,
            description=description,
            characteristics=characteristics,
            brand=brand,
            category=category,
            price_usd=price_usd,
            price_uzs=price_uzs,
            stock=r.get("stock"),
            model=str(r.get("model") or ""),
            images=images,
        )

    # Keep input order
    return [by_id[pid] for pid in product_ids if pid in by_id]


async def list_categories(settings: Settings, limit: int = 50) -> list[tuple[int, str]]:
    query = """
        SELECT id, title
        FROM categories
        WHERE status = 1
        ORDER BY id DESC
        LIMIT %s
    """
    rows = await fetch_all_async(settings, query, (limit,))
    result: list[tuple[int, str]] = []
    for r in rows:
        result.append((int(r["id"]), clean_json_field(r.get("title"))))
    return result


async def list_brands(settings: Settings, limit: int = 50) -> list[tuple[int, str]]:
    query = """
        SELECT id, title
        FROM brands
        WHERE status = 1
        ORDER BY id DESC
        LIMIT %s
    """
    rows = await fetch_all_async(settings, query, (limit,))
    result: list[tuple[int, str]] = []
    for r in rows:
        title = str(r.get("title") or "")
        result.append((int(r["id"]), title))
    return result


async def list_all_brand_names(settings: Settings) -> list[str]:
    rows = await fetch_all_async(
        settings,
        """
        SELECT title
        FROM brands
        WHERE status = 1
        ORDER BY title ASC
        """,
        None,
    )
    names: list[str] = []
    for r in rows:
        name = str(r.get("title") or "").strip()
        if name:
            names.append(name)
    return names


async def find_brand_names_like(settings: Settings, text: str, limit: int = 5) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    # Tokenize and try longer tokens first (works for "Samsung brandiga ...").
    parts = [p.strip(".,!?\"'()[]{}") for p in t.split() if p]
    parts = [p for p in parts if len(p) >= 3]
    parts = sorted(set(parts), key=len, reverse=True)
    candidates = parts[:8] or [t]
    seen: set[str] = set()
    out: list[str] = []
    for cand in candidates:
        like = f"%{cand}%"
        rows = await fetch_all_async(
            settings,
            """
            SELECT title
            FROM brands
            WHERE status = 1 AND title LIKE %s
            ORDER BY LENGTH(title) ASC
            LIMIT %s
            """,
            (like, int(limit)),
        )
        for r in rows:
            name = str(r.get("title") or "").strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                out.append(name)
        if len(out) >= limit:
            break
    return out[:limit]


async def search_product_ids_by_brand(settings: Settings, brand_name: str, limit: int = 80) -> list[int]:
    brand = (brand_name or "").strip()
    if not brand:
        return []
    rows = await fetch_all_async(
        settings,
        """
        SELECT p.id
        FROM products p
        JOIN brands b ON p.brand_id = b.id
        WHERE p.deleted_at IS NULL AND p.status = 1 AND b.status = 1 AND b.title = %s
        ORDER BY p.stock DESC, p.id DESC
        LIMIT %s
        """,
        (brand, int(limit)),
    )
    ids: list[int] = []
    for r in rows:
        try:
            ids.append(int(r["id"]))
        except Exception:
            continue
    return ids


async def search_product_ids_sql(settings: Settings, query: str, limit: int = 30) -> list[int]:
    """
    Fallback search when vector search is empty/unavailable.
    Uses simple LIKE matching across title/keywords/category/brand.
    """
    q = (query or "").strip()
    if not q:
        return []
    words = [w for w in q.split() if len(w) >= 3][:6]
    # Add a few common synonyms so users can find items even if DB uses another term.
    synonym_map = {
        "pilesos": ["pylesos", "changyutgich", "chang", "пылесос", "vacuum"],
        "pylesos": ["pilesos", "changyutgich", "пылесос", "vacuum"],
        "changyutgich": ["pylesos", "пылесос", "vacuum"],
        "konditsioner": ["conditioner", "кондиционер", "split"],
    }
    expanded: list[str] = []
    for w in words:
        expanded.append(w)
        for s in synonym_map.get(w.lower(), []):
            expanded.append(s)
    words = expanded[:12]
    if not words:
        words = [q[:20]]

    where_parts: list[str] = []
    args: list[str] = []
    for w in words:
        like = f"%{w}%"
        where_parts.append(
            "(p.title LIKE %s OR p.keywords LIKE %s OR p.model LIKE %s OR c.title LIKE %s OR b.title LIKE %s)"
        )
        args.extend([like, like, like, like, like])

    where_sql = " OR ".join(where_parts) if where_parts else "1=0"
    sql = f"""
        SELECT p.id
        FROM products p
        LEFT JOIN categories c ON p.category_id = c.id
        LEFT JOIN brands b ON p.brand_id = b.id
        WHERE p.deleted_at IS NULL AND p.status = 1 AND ({where_sql})
        ORDER BY p.stock DESC, p.id DESC
        LIMIT %s
    """
    args.append(str(int(limit)))
    rows = await fetch_all_async(settings, sql, tuple(args))
    ids: list[int] = []
    for r in rows:
        try:
            ids.append(int(r["id"]))
        except Exception:
            continue
    return ids
