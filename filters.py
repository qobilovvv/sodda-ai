from __future__ import annotations

import re
from dataclasses import dataclass


USD_TO_UZS = 12000


@dataclass
class Filters:
    min_usd: int | None = None
    max_usd: int | None = None
    brand: str | None = None
    size_token: str | None = None  # e.g. "55", "6/128" (exact token)
    size_min: int | None = None
    size_max: int | None = None


def parse_budget_usd(text: str) -> tuple[int | None, int | None]:
    """
    Parses patterns like:
      - "300$" (treated as max=300)
      - "300$ gacha", "<=300$", "do 300"
      - "300$ dan yuqori", ">=300$", "ot 300"
      - "200-300$"
    """
    t = (text or "").lower()
    t = t.replace("usd", "$").replace("dollar", "$").replace("dol", "$")

    range_m = re.search(r"(\d{2,5})\s*[-–]\s*(\d{2,5})\s*\$?", t)
    if range_m:
        a, b = int(range_m.group(1)), int(range_m.group(2))
        return (min(a, b), max(a, b))

    nums = re.findall(r"(\d{2,5})\s*\$?", t)
    if not nums:
        return (None, None)
    val = int(nums[0])

    if any(k in t for k in ["dan yuqori", "yuqori", ">= ", ">=", "ko'proq", "more", "higher", "ot "]):
        return (val, None)
    if any(k in t for k in ["dan past", "past", "<= ", "<=", "kamroq", "less", "lower", "gacha", "do "]):
        return (None, val)

    # Default: treat as max budget
    return (None, val)


def parse_size_token(text: str) -> str | None:
    """
    Very lightweight "razmer/size" extraction:
      - '55' / '55 inch' / '55"' -> '55'
      - '6/128' -> '6/128'
    """
    t = (text or "").lower()
    if "razmer" not in t and "size" not in t and "\"" not in t and "inch" not in t and "dyuym" not in t:
        # Still allow common 6/128 pattern for phones
        m = re.search(r"\b(\d{1,2}\s*/\s*\d{2,4})\b", t)
        if m:
            return m.group(1).replace(" ", "")
        return None

    m = re.search(r"\b(\d{2,3})\s*(?:\"|inch|dyuym)?\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{1,2}\s*/\s*\d{2,4})\b", t)
    if m:
        return m.group(1).replace(" ", "")
    return None


def parse_size_range(text: str) -> tuple[int | None, int | None]:
    """
    Parses simple size constraints:
      - '55"' -> (55,55) handled by caller if needed
      - '55 dan katta/yuqori' -> (55,None)
      - '55 dan kichik/past' -> (None,55)
      - '50-55' -> (50,55)
    """
    t = (text or "").lower()
    m = re.search(r"\b(\d{2,3})\s*[-–]\s*(\d{2,3})\b", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    m = re.search(r"\b(\d{2,3})\b", t)
    if not m:
        return (None, None)
    val = int(m.group(1))
    if any(k in t for k in ["dan katta", "kattaroq", "yuqori", ">= ", ">=", "bigger", "larger"]):
        return (val, None)
    if any(k in t for k in ["dan kichik", "kichikroq", "past", "<= ", "<=", "smaller", "lower"]):
        return (None, val)
    return (None, None)


def usd_to_uzs(usd: int) -> int:
    return int(usd) * USD_TO_UZS
