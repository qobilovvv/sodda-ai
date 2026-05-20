from __future__ import annotations

from typing import Any

from db import fetch_all, fetch_one
from products import Product, clean_json_field, product_from_row
from search import apply_corrections, normalize_query, search_products


def tool_search_products(query: str, limit: int = 10) -> list[Product]:
    return search_products(query, limit=limit)


def tool_get_product(product_id: int) -> Product | None:
    row = fetch_one(
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
    return product_from_row(row)


def tool_list_categories(limit: int = 50) -> list[dict[str, Any]]:
    rows = fetch_all(
        "SELECT id, title FROM categories WHERE status = 1 ORDER BY id DESC LIMIT %s",
        (int(limit),),
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({"id": int(r["id"]), "title": clean_json_field(r.get("title"))})
    return out


def tool_list_brands(limit: int = 50) -> list[dict[str, Any]]:
    rows = fetch_all(
        "SELECT id, title FROM brands WHERE status = 1 ORDER BY id DESC LIMIT %s",
        (int(limit),),
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({"id": int(r["id"]), "title": clean_json_field(r.get("title"))})
    return out


def tool_brand_options_for_query(query: str, limit: int = 10) -> list[str]:
    """
    Returns distinct brand titles for products matching the query.
    This is used to let user choose a brand before listing products.
    """
    q_norm = normalize_query(query)
    q_corr = apply_corrections(q_norm) or q_norm or query
    like = f"%{q_corr}%"
    rows = fetch_all(
        """
        SELECT DISTINCT b.title AS brand_title
        FROM products p
        LEFT JOIN brands b ON p.brand_id = b.id
        LEFT JOIN categories c ON p.category_id = c.id
        WHERE p.deleted_at IS NULL AND p.status = 1
          AND (
            p.title LIKE %s OR p.keywords LIKE %s OR p.model LIKE %s OR
            c.title LIKE %s OR b.title LIKE %s
          )
          AND b.title IS NOT NULL AND b.title <> ''
        LIMIT 50
        """,
        (like, like, like, like, like),
    )
    brands: list[str] = []
    for r in rows:
        title = clean_json_field(r.get("brand_title")).strip()
        if title and title not in brands:
            brands.append(title)
        if len(brands) >= limit:
            break
    return brands
