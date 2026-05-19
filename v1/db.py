import os
from typing import Any, Iterable

import pymysql
import pymysql.cursors
from dotenv import load_dotenv


load_dotenv()


def _connect():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_DATABASE"),
        cursorclass=pymysql.cursors.DictCursor,
    )


def fetch_all(query: str, params: Iterable[Any] | None = None) -> list[dict[str, Any]]:
    connection = _connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            return list(cursor.fetchall())
    finally:
        connection.close()


def fetch_one(query: str, params: Iterable[Any] | None = None) -> dict[str, Any] | None:
    rows = fetch_all(query, params)
    return rows[0] if rows else None

