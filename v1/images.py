from __future__ import annotations

import asyncio

import aiohttp


async def resolve_product_image_url(url: str, timeout_s: float = 2.0) -> str | None:
    """
    Matches existing behavior in `bot.py`:
    - Prefer `/thumbs/` if it exists
    - Fallback to original `/products/`
    - Return None if not reachable
    """
    if not url:
        return None

    filename = url.split("/")[-1].strip()
    if not filename:
        return None

    thumb_url = f"https://sodda.uz/storage/products/thumbs/{filename}"
    main_url = f"https://sodda.uz/storage/products/{filename}"

    async with aiohttp.ClientSession() as session:
        for candidate in (thumb_url, main_url):
            try:
                async with session.head(candidate, timeout=timeout_s) as resp:
                    if resp.status == 200:
                        return candidate
            except Exception:
                continue
    return None


async def resolve_first_image(urls: list[str]) -> str | None:
    if not urls:
        return None
    # Try first 3 quickly in parallel; pick first valid.
    tasks = [resolve_product_image_url(u) for u in urls[:3]]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, str) and r:
            return r
    return None

