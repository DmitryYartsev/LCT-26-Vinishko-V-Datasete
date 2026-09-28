# -*- coding: utf-8 -*-
"""Реранк top-K кандидатов мультимодальным реранкером: фото запроса vs эталонные фото вин.

Этап после векторного поиска: pgvector отдаёт top-K по косинусу (SigLIP), реранкер попарно
сравнивает фото запроса (после кропа) с эталоном каждого кандидата. Итоговый скор — смесь
реранкера и визуального скора, обе величины min-max нормированы внутри top-K:

  final = RERANK_WEIGHT * norm(rerank) + (1 - RERANK_WEIGHT) * norm(visual)

Любая ошибка/таймаут реранкера -> возвращается исходный порядок (сервис не падает).

Бэкенды (API у обоих одинаковой формы: query+documents -> index/relevance_score):
  jina       jina-reranker-m0 (api.jina.ai), image-query поддерживается
  dashscope  qwen3-vl-rerank (Alibaba Model Studio, DashScope API)

ENV:
  RERANK_BACKEND   none | jina | dashscope          (деф none — реранк выключен)
  RERANK_TOP_K     сколько кандидатов реранкать     (деф 10)
  RERANK_WEIGHT    доля реранкера в итоговом скоре  (деф 0.5; 1.0 — чистый порядок реранкера)
  RERANK_TIMEOUT   таймаут запроса, с               (деф 5)
  RERANK_IMG_SIDE  макс. сторона картинок, px       (деф 448; больше — точнее, но дороже в токенах)
  JINA_API_KEY, JINA_RERANK_MODEL (jina-reranker-m0), JINA_RERANK_URL
  DASHSCOPE_API_KEY, DASHSCOPE_RERANK_URL (полный URL с WorkspaceId и регионом),
  DASHSCOPE_RERANK_MODEL (qwen3-vl-rerank), DASHSCOPE_INSTRUCT
"""
import os, io, sys, time, base64
from functools import lru_cache
from pathlib import Path
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import FILTERED

BACKEND = os.environ.get("RERANK_BACKEND", "none").lower()
TOP_K = int(os.environ.get("RERANK_TOP_K", "10"))
WEIGHT = float(os.environ.get("RERANK_WEIGHT", "0.5"))
TIMEOUT = float(os.environ.get("RERANK_TIMEOUT", "5"))
IMG_SIDE = int(os.environ.get("RERANK_IMG_SIDE", "448"))

JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_MODEL = os.environ.get("JINA_RERANK_MODEL", "jina-reranker-m0")
JINA_URL = os.environ.get("JINA_RERANK_URL", "https://api.jina.ai/v1/rerank")

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = os.environ.get("DASHSCOPE_RERANK_URL", "")
DASHSCOPE_MODEL = os.environ.get("DASHSCOPE_RERANK_MODEL", "qwen3-vl-rerank")
DASHSCOPE_INSTRUCT = os.environ.get(
    "DASHSCOPE_INSTRUCT",
    "Given a photo of a wine bottle, retrieve the catalog photo of exactly the same wine: "
    "same producer, name, color, sweetness and vintage.")

ENABLED = BACKEND in ("jina", "dashscope")


def _b64(img: Image.Image) -> str:
    img = img.convert("RGB")
    img.thumbnail((IMG_SIDE, IMG_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


@lru_cache(maxsize=4096)
def _ref_b64(slug: str) -> str | None:
    """Эталон вина (первое фото из filtered/<slug>/, как и /ref) в base64 — кэшируется."""
    d = FILTERED / slug
    imgs = sorted(p for p in d.iterdir() if p.is_file()) if d.is_dir() else []
    return _b64(Image.open(imgs[0])) if imgs else None


def _jina(q: str, docs: list[str]) -> list[float]:
    r = requests.post(JINA_URL, timeout=TIMEOUT,
                      headers={"Authorization": f"Bearer {JINA_API_KEY}"},
                      json={"model": JINA_MODEL, "query": {"image": q},
                            "documents": [{"image": d} for d in docs], "return_documents": False})
    r.raise_for_status()
    return _scores(r.json()["results"], len(docs))


def _dashscope(q: str, docs: list[str]) -> list[float]:
    uri = lambda b: f"data:image/jpeg;base64,{b}"
    r = requests.post(DASHSCOPE_URL, timeout=TIMEOUT,
                      headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
                      json={"model": DASHSCOPE_MODEL,
                            "input": {"query": {"image": uri(q)}, "documents": [{"image": uri(d)} for d in docs]},
                            "parameters": {"return_documents": False, "top_n": len(docs),
                                           "instruct": DASHSCOPE_INSTRUCT}})
    r.raise_for_status()
    return _scores(r.json()["output"]["results"], len(docs))


def _scores(results: list[dict], n: int) -> list[float]:
    out = [0.0] * n
    for x in results:
        out[x["index"]] = float(x["relevance_score"])
    return out


def _norm(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 1.0 for x in xs]


def check_config() -> str | None:
    """Проблема конфигурации (для лога на старте и /health) или None."""
    if BACKEND == "jina" and not JINA_API_KEY:
        return "JINA_API_KEY не задан"
    if BACKEND == "dashscope" and not (DASHSCOPE_API_KEY and DASHSCOPE_URL):
        return "DASHSCOPE_API_KEY / DASHSCOPE_RERANK_URL не заданы"
    if BACKEND not in ("none", "jina", "dashscope"):
        return f"неизвестный RERANK_BACKEND={BACKEND}"
    return None


def rerank(query_img: Image.Image, results: list[dict]) -> tuple[list[dict], dict]:
    """results: [{slug, score, card}] по убыванию визуального скора.
    -> (переупорядоченные results с полями rerank_score/final_score, info для ответа API)."""
    info = {"backend": BACKEND, "applied": False,
            "visual_top1": results[0]["slug"] if results else None}   # для харнесса: recall@1 до реранка
    if not ENABLED or len(results) < 2:
        return results, info
    t = time.perf_counter()
    try:
        if err := check_config():
            raise RuntimeError(err)
        cand = [r for r in results if _ref_b64(r["slug"])]   # без эталона сравнивать не с чем
        if len(cand) < 2:
            return results, info
        call = _jina if BACKEND == "jina" else _dashscope
        rr = call(_b64(query_img), [_ref_b64(r["slug"]) for r in cand])
    except Exception as e:
        info.update(error=str(e)[:300], ms=round(1000 * (time.perf_counter() - t)))
        print(f"[rerank] {BACKEND} ошибка, оставляю визуальный порядок: {e}")
        return results, info
    vis = _norm([r["score"] for r in cand])
    rrn = _norm(rr)
    for r, s, v, n in zip(cand, rr, vis, rrn):
        r["rerank_score"] = round(s, 4)
        r["final_score"] = round(WEIGHT * n + (1 - WEIGHT) * v, 4)
    cand.sort(key=lambda r: r["final_score"], reverse=True)
    rest = [r for r in results if r not in cand]
    info.update(applied=True, ms=round(1000 * (time.perf_counter() - t)), weight=WEIGHT,
                n_candidates=len(cand), changed_top1=cand[0]["slug"] != results[0]["slug"])
    return cand + rest, info
