import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    openai_api_key: str | None
    openai_model: str
    ai_enabled: bool

    db_host: str
    db_username: str
    db_password: str
    db_database: str

    chroma_dir: str
    usd_to_uzs: float
    manager_phone: str

    search_k: int
    max_products_per_reply: int
    max_caption_chars: int


def _get_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    return _get_settings()


@lru_cache(maxsize=1)
def _get_settings() -> Settings:
    load_dotenv()

    telegram_token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if not telegram_token:
        raise RuntimeError("Missing TELEGRAM_TOKEN in environment")

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    # Force gpt-4o-mini as requested.
    openai_model = "gpt-4o-mini"
    # Enable AI intro by default when API key exists.
    ai_enabled = _get_bool("AI_ENABLED", "1")

    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment (AI replies are required).")

    return Settings(
        telegram_token=telegram_token,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        ai_enabled=ai_enabled and (openai_api_key is not None),
        db_host=os.getenv("DB_HOST", "127.0.0.1").strip(),
        db_username=os.getenv("DB_USERNAME", "").strip(),
        db_password=os.getenv("DB_PASSWORD", "").strip(),
        db_database=os.getenv("DB_DATABASE", "").strip(),
        chroma_dir=os.getenv("CHROMA_DIR", "./chroma_db").strip(),
        usd_to_uzs=float(os.getenv("USD_TO_UZS", "12000")),
        manager_phone=os.getenv("MANAGER_PHONE", "+998950001234").strip(),
        search_k=int(os.getenv("SEARCH_K", "8")),
        max_products_per_reply=int(os.getenv("MAX_PRODUCTS_PER_REPLY", "10")),
        max_caption_chars=int(os.getenv("MAX_CAPTION_CHARS", "900")),
    )
