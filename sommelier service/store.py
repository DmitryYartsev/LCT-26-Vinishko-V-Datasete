# -*- coding: utf-8 -*-
"""Postgres-слой сомелье: таблица wine_profiles (атрибуты вин для подбора) + FTS по описаниям.

Живёт в той же БД, что и индекс `ml`, но в своей таблице — сервисы друг от друга не зависят
(slug — общий ключ, без FK: порядок старта контейнеров не важен).
"""
import os
import psycopg
from psycopg.rows import dict_row

DB_URL = os.environ.get("DATABASE_URL", "postgresql://vino:vino@db:5432/vino")
FIELDS = ["slug", "site_slug", "in_catalog", "title", "manufacturer", "category", "color", "sweetness",
          "sparkling", "alcohol", "temperature", "grapes", "region", "dishes", "rating",
          "description", "url", "image"]


def connect():
    conn = psycopg.connect(DB_URL, autocommit=True, row_factory=dict_row)
    ensure_schema(conn)
    return conn


def ensure_schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS wine_profiles (
        slug TEXT PRIMARY KEY, site_slug TEXT, in_catalog BOOLEAN, title TEXT, manufacturer TEXT,
        category TEXT, color TEXT, sweetness TEXT, sparkling BOOLEAN, alcohol REAL, temperature TEXT,
        grapes TEXT[], region TEXT, dishes TEXT[], rating REAL, description TEXT, url TEXT, image TEXT,
        tsv tsvector GENERATED ALWAYS AS (
            to_tsvector('russian', coalesce(title,'') || ' ' || coalesce(description,''))) STORED)""")
    conn.execute("CREATE INDEX IF NOT EXISTS wine_profiles_tsv_idx ON wine_profiles USING gin(tsv)")


def count(conn) -> int:
    return conn.execute("SELECT count(*) AS n FROM wine_profiles").fetchone()["n"]


def replace_all(conn, rows):
    cols = ",".join(FIELDS)
    vals = ",".join(f"%({f})s" for f in FIELDS)
    with conn.transaction():
        conn.execute("DELETE FROM wine_profiles")
        with conn.cursor() as cur:
            cur.executemany(f"INSERT INTO wine_profiles ({cols}) VALUES ({vals})", rows)


def load_all(conn) -> dict[str, dict]:
    rows = conn.execute(f"SELECT {','.join(FIELDS)} FROM wine_profiles").fetchall()
    return {r["slug"]: r for r in rows}


def note_hits(conn, notes: list[str], slugs: list[str] | None = None) -> dict[str, set]:
    """{заметка: slug'и, в описании/названии которых она встречается} — русский стемминг Postgres."""
    out = {}
    for n in notes:
        q = "SELECT slug FROM wine_profiles WHERE tsv @@ plainto_tsquery('russian', %s)"
        args = [n]
        if slugs is not None:
            q += " AND slug = ANY(%s)"
            args.append(slugs)
        out[n] = {r["slug"] for r in conn.execute(q, args).fetchall()}
    return out
