import asyncio
from typing import Any

import pymysql
import pymysql.cursors

from .config import Settings


def fetch_all(settings: Settings, query: str, args: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
    connection = pymysql.connect(
        host=settings.db_host,
        user=settings.db_username,
        password=settings.db_password,
        database=settings.db_database,
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, args)
            return list(cursor.fetchall())
    finally:
        connection.close()


async def fetch_all_async(settings: Settings, query: str, args: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
    return await asyncio.to_thread(fetch_all, settings, query, args)

