 Inline keyboard flows (buttons): after “qaysi brend/kategoriya/byudjet?” show buttons like Brendlar, Kategoriyalar, Byudjet:
    <300$ / 300–600$ / 600$+>, No Frost / Inverter / Quiet, plus Show more for pagination. (Right now everything is free-text, which
    increases confusion/loops.)
  - Product “focus” mode (pick one): when you send 6–10 products, add “Tanlash: 1–6” buttons (or a #2 hint) so users can easily ask
    about one item, then keep conversation scoped to that selected product.
  - Pagination + “Top N”: users get overwhelmed; add Keyingi 6 ta / Oldingi and optionally Faqat eng arzon / eng yaxshi toggles.
  - Direct product link + CTA: include a website link (product page) and 1–2 clear next actions: Buyurtma berish, Operator, Yetkazib
    berish, Kafolat. (Currently the prompt has a static phone; a real link/button usually converts better.)
  - “Compare” UX: let users select 2 products and tap Taqqoslash to get a compact comparison table (price/specs/warranty/stock).
  - Stock + delivery expectations: show “Mavjud: ha/yo’q” and “Yetkazib berish: 1–2 kun” if you have it in DB (or add it). These are
    the most common follow-ups.
  - Better “unknown” handling: instead of generic fallback, offer 2–3 suggested intents as buttons: Mahsulot qidirish, Brendlar,
    Kategoriyalar, Operator.
  - Performance UX: add a “searching” typing indicator is good; also consider caching image HEAD checks and reusing one aiohttp
    session so albums load faster (right now get_valid_image_url() opens a new session per image).
  - Conversation memory UX: expose /last (last shown products), /brands, /categories, /compare, and show these in /help so users
    discover capabilities quickly.
