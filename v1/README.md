# v1 Telegram Manager Bot

This is a lightweight Telegram “seller/manager” bot that answers using your catalog (brands/categories/products) and sends product cards with image + a static manager phone number.

## Requirements

- Run `sync.py` at least once to build `./chroma_db` (product/brand/category vectors).
- Environment variables (see `.env.example`):
  - `TELEGRAM_TOKEN`
  - `OPENAI_API_KEY` (needed for Chroma search; optional for AI intro text)
  - `DB_HOST`, `DB_USERNAME`, `DB_PASSWORD`, `DB_DATABASE`
  - Optional: `MANAGER_PHONE`, `USD_TO_UZS`

## Run

From repo root:

`python -m v1.bot_main`

## Low-token mode

- By default, the bot does **no chat generation**. It only does retrieval + deterministic formatting.
- To enable a short AI “manager-style” intro message, set:
  - `AI_ENABLED=1`
  - Optional: `OPENAI_MODEL=gpt-4o-mini`

