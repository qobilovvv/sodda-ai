import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal


def clean_json_field(data, prefer_lang: str = "uz") -> str:
    if data is None:
        return ""
    if isinstance(data, (int, float)):
        return str(data)
    if isinstance(data, str):
        s = data.strip()
        if not s:
            return ""
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            return data

    if isinstance(data, dict):
        content = data.get(prefer_lang) or data.get("ru") or next(iter(data.values()), "")
        if isinstance(content, dict):
            return ", ".join([f"{k}: {v}" for k, v in content.items()])
        return str(content)

    return str(data)


def format_specs(spec_data, prefer_lang: str = "uz") -> str:
    """
    Normalizes product `spec` into a human-readable string.
    Handles cases where spec is:
      - JSON string of list[{"title_uz","value_uz",...}]
      - list/dict already parsed by driver
      - arbitrary string
    """
    if not spec_data:
        return ""

    data = spec_data
    if isinstance(spec_data, str):
        s = spec_data.strip()
        if not s:
            return ""
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            return spec_data

    # Common case: list of {title_uz/value_uz/...}
    if isinstance(data, list):
        lines: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            title = (
                item.get(f"title_{prefer_lang}")
                or item.get("title_uz")
                or item.get("title_ru")
                or item.get("title")
                or ""
            )
            value = (
                item.get(f"value_{prefer_lang}")
                or item.get("value_uz")
                or item.get("value_ru")
                or item.get("value")
                or ""
            )
            title_s = clean_json_field(title, prefer_lang=prefer_lang).strip()
            value_s = clean_json_field(value, prefer_lang=prefer_lang).strip()
            if title_s and value_s:
                lines.append(f"{title_s}: {value_s}")
            elif title_s:
                lines.append(title_s)
        return "\n".join(lines)

    if isinstance(data, dict):
        # Sometimes it's a lang map; reuse clean_json_field
        return clean_json_field(data, prefer_lang=prefer_lang)

    return str(spec_data)

def extract_image_urls(image_data) -> list[str]:
    if not image_data:
        return []
    try:
        imgs = json.loads(image_data) if isinstance(image_data, str) else image_data
        if not isinstance(imgs, list):
            return []
        urls: list[str] = []
        for img in imgs:
            if not isinstance(img, str):
                continue
            clean_name = img.replace("products/", "").replace("thumbs/", "").strip("/")
            if clean_name:
                urls.append(f"https://sodda.uz/storage/products/{clean_name}")
        return urls
    except Exception:
        return []


def format_soum(amount: int) -> str:
    # Use dot as thousands separator: 1200000 -> 1.200.000
    s = f"{amount:,}".replace(",", ".")
    return s


def format_usd(amount: int) -> str:
    # 2600 -> 2,600
    return f"{amount:,}"


def format_price_dual(uzs_price) -> tuple[str, str]:
    try:
        uzs_int = int(float(uzs_price or 0))
    except Exception:
        uzs_int = 0
    usd = round(uzs_int / 12000) if uzs_int else 0
    return f"{usd}$", f"{format_soum(uzs_int)} so'm"


def format_price_usd(uzs_price) -> str:
    try:
        uzs_int = int(float(uzs_price or 0))
    except Exception:
        uzs_int = 0
    usd = round(uzs_int / 12000) if uzs_int else 0
    return f"{usd}$"


def format_price_ui(price_value: int) -> str:
    """
    Price display for UI.
    If DB price is stored in USD (common in some catalogs), set `SODDA_PRICE_UNIT=USD`
    and we will show it as `$` without converting or adding so'm.
    Default is USD to match current expectations.
    """
    unit = (os.getenv("SODDA_PRICE_UNIT", "USD") or "USD").upper()
    v = int(price_value or 0)

    if unit == "UZS":
        usd, uzs = format_price_dual(v)
        return f"{usd} | {uzs}"

    # USD
    return f"{format_usd(v)}$"


def parse_price_uzs(value) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, Decimal):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return 0
        # Remove common formatting: spaces, commas, currency symbols
        s = re.sub(r"[^\d.]+", "", s)
        if not s:
            return 0
        try:
            return int(float(s))
        except Exception:
            return 0
    try:
        return int(value)
    except Exception:
        return 0


def truncate(text: str, max_len: int) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


@dataclass(frozen=True)
class Product:
    id: int
    title: str
    model: str
    brand: str
    category: str
    price_uzs: int
    description: str
    specs: str
    keywords: str
    image_urls: list[str]


def product_from_row(row: dict) -> Product:
    title = clean_json_field(row.get("title"))
    brand = clean_json_field(row.get("brand_title"))
    category = clean_json_field(row.get("category_title"))
    specs = format_specs(row.get("spec"))
    description = clean_json_field(row.get("sm_desc"))
    keywords = clean_json_field(row.get("keywords"))
    model = clean_json_field(row.get("model"))
    price_val = row.get("price")
    if price_val is None:
        price_val = row.get("price_uzs")
    if price_val is None:
        price_val = row.get("price_value")
    price_uzs = parse_price_uzs(price_val)
    image_urls = extract_image_urls(row.get("images"))
    return Product(
        id=int(row["id"]),
        title=title,
        model=model,
        brand=brand,
        category=category,
        price_uzs=price_uzs,
        description=description,
        specs=specs,
        keywords=keywords,
        image_urls=image_urls,
    )
