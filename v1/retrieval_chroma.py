from __future__ import annotations

from functools import lru_cache

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from .config import Settings


@lru_cache(maxsize=1)
def _get_vector_db(chroma_dir: str, openai_api_key: str | None) -> Chroma:
    # Chroma needs an embedding function at query time.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    return Chroma(persist_directory=chroma_dir, embedding_function=embeddings)


def search_product_ids(settings: Settings, query: str, k: int | None = None) -> list[int]:
    if not settings.openai_api_key:
        return []

    try:
        vector_db = _get_vector_db(settings.chroma_dir, settings.openai_api_key)
        docs_with_scores = vector_db.similarity_search_with_score(
            query=query,
            k=k or settings.search_k,
            filter={"entity_type": "product"},
        )
    except Exception:
        return []

    ids: list[int] = []
    for doc, _score in docs_with_scores:
        pid = doc.metadata.get("product_id")
        if pid is None:
            continue
        try:
            pid_int = int(pid)
        except Exception:
            continue
        if pid_int not in ids:
            ids.append(pid_int)
    return ids


def search_product_ids_with_scores(settings: Settings, query: str, k: int | None = None) -> list[tuple[int, float]]:
    if not settings.openai_api_key:
        return []
    try:
        vector_db = _get_vector_db(settings.chroma_dir, settings.openai_api_key)
        docs_with_scores = vector_db.similarity_search_with_score(
            query=query,
            k=k or settings.search_k,
            filter={"entity_type": "product"},
        )
    except Exception:
        return []

    out: list[tuple[int, float]] = []
    seen: set[int] = set()
    for doc, score in docs_with_scores:
        pid = doc.metadata.get("product_id")
        if pid is None:
            continue
        try:
            pid_int = int(pid)
        except Exception:
            continue
        if pid_int in seen:
            continue
        seen.add(pid_int)
        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0
        out.append((pid_int, score_f))
    return out


def search_brand_names(settings: Settings, query: str, k: int = 5) -> list[str]:
    if not settings.openai_api_key:
        return []

    try:
        vector_db = _get_vector_db(settings.chroma_dir, settings.openai_api_key)
        docs = vector_db.similarity_search(
            query=query,
            k=k,
            filter={"entity_type": "brand"},
        )
    except Exception:
        return []
    names: list[str] = []
    for d in docs:
        text = d.page_content
        for line in text.splitlines():
            if line.startswith("BRAND_NAME:"):
                name = line.split(":", 1)[1].strip()
                if name and name not in names:
                    names.append(name)
    return names


def search_category_names(settings: Settings, query: str, k: int = 5) -> list[str]:
    if not settings.openai_api_key:
        return []

    try:
        vector_db = _get_vector_db(settings.chroma_dir, settings.openai_api_key)
        docs = vector_db.similarity_search(
            query=query,
            k=k,
            filter={"entity_type": "category"},
        )
    except Exception:
        return []
    names: list[str] = []
    for d in docs:
        text = d.page_content
        for line in text.splitlines():
            if line.startswith("CATEGORY_NAME:"):
                name = line.split(":", 1)[1].strip()
                if name and name not in names:
                    names.append(name)
    return names
