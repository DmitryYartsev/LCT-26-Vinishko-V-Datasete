# -*- coding: utf-8 -*-
"""pgvector-слой: схема, апсерт каталога/векторов, косинусный поиск.

Мультивектор: на slug несколько строк в wine_vectors, матч = max cosine по slug.
Мульти-МОДЕЛЬ: у каждого вектора есть `model`; вектора разных моделей сосуществуют,
поиск фильтрует по текущей модели. Колонка emb — dimensionless `vector` (разные dim ок;
ANN-индекс не строим — каталог маленький, full scan быстрый). Мета моделей — в `models`.
"""
import os
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

DB_URL = os.environ.get("DATABASE_URL", "postgresql://vino:vino@db:5432/vino")
CARD_FIELDS = ["slug", "name", "winery", "category", "color", "region", "grape", "description", "rating"]


def connect():
    conn = psycopg.connect(DB_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    ensure_schema(conn)
    return conn


def ensure_schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS wines (
        slug TEXT PRIMARY KEY, name TEXT, winery TEXT, category TEXT, color TEXT,
        region TEXT, grape TEXT, description TEXT, rating TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS wine_vectors (
        id SERIAL PRIMARY KEY, slug TEXT REFERENCES wines(slug) ON DELETE CASCADE,
        model TEXT NOT NULL, emb vector)""")
    conn.execute("CREATE INDEX IF NOT EXISTS wine_vectors_model_idx ON wine_vectors(model)")
    conn.execute("""CREATE TABLE IF NOT EXISTS models (
        model TEXT PRIMARY KEY, dim INT, backend TEXT, n_vectors INT,
        built_at TIMESTAMPTZ DEFAULT now())""")


def has_vectors(conn, model: str) -> bool:
    return conn.execute("SELECT EXISTS(SELECT 1 FROM wine_vectors WHERE model=%s)", (model,)).fetchone()[0]


def upsert_wines(conn, rows):
    with conn.cursor() as cur:
        cur.executemany("""INSERT INTO wines (slug,name,winery,category,color,region,grape,description,rating)
            VALUES (%(slug)s,%(name)s,%(winery)s,%(category)s,%(color)s,%(region)s,%(grape)s,%(description)s,%(rating)s)
            ON CONFLICT (slug) DO UPDATE SET name=EXCLUDED.name, winery=EXCLUDED.winery,
              category=EXCLUDED.category, color=EXCLUDED.color, region=EXCLUDED.region,
              grape=EXCLUDED.grape, description=EXCLUDED.description, rating=EXCLUDED.rating""",
            [{k: r.get(k, "") for k in CARD_FIELDS} for r in rows])


def replace_vectors(conn, model: str, pairs, dim: int, backend: str):
    """pairs: список (slug, emb[np.float32]). Перезаливает вектора ТОЛЬКО этой модели."""
    conn.execute("DELETE FROM wine_vectors WHERE model=%s", (model,))
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO wine_vectors (slug, model, emb) VALUES (%s, %s, %s)",
                        [(s, model, v) for s, v in pairs])
    conn.execute("""INSERT INTO models (model,dim,backend,n_vectors,built_at)
        VALUES (%s,%s,%s,%s,now())
        ON CONFLICT (model) DO UPDATE SET dim=EXCLUDED.dim, backend=EXCLUDED.backend,
          n_vectors=EXCLUDED.n_vectors, built_at=now()""", (model, dim, backend, len(pairs)))


def search(conn, qvec: np.ndarray, model: str, k: int = 5):
    rows = conn.execute("""SELECT slug, MIN(emb <=> %s) AS dist FROM wine_vectors
        WHERE model=%s GROUP BY slug ORDER BY dist LIMIT %s""", (qvec, model, k)).fetchall()
    return [{"slug": s, "score": round(1.0 - float(d), 4), "card": get_card(conn, s)} for s, d in rows]


def get_card(conn, slug: str) -> dict:
    r = conn.execute("SELECT slug,name,winery,category,color,region,grape,description,rating "
                     "FROM wines WHERE slug=%s", (slug,)).fetchone()
    return dict(zip(CARD_FIELDS, r)) if r else {"slug": slug}


def list_models(conn):
    rows = conn.execute("SELECT model,dim,backend,n_vectors,built_at FROM models ORDER BY built_at DESC").fetchall()
    return [{"model": m, "dim": d, "backend": b, "n_vectors": n, "built_at": str(t)} for m, d, b, n, t in rows]
