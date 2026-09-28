# -*- coding: utf-8 -*-
"""FastAPI-сервис сканера вин (pgvector).

Эндпоинты:
  POST /v1/eval/predict  — для скрипта-оценщика: multipart `image` -> {"slug": "..."}
  POST /v1/search        — богатый ответ: top-5 + score/margin + карточка
  GET  /wine/{slug}      — карточка по slug
  GET  /ref/{slug}       — эталонное фото
  GET  /health           — статус

На старте: warm-up (сид каталога + построение индекса в pgvector, если пусто).
Модели и пороги — через env (см. README).
"""
import os, io, sys, time
from functools import lru_cache
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import FILTERED
import db
from encoder import get_encoder
from crop import maybe_crop, CROP_ENABLED
from build_index import warmup

THRESH_SCORE = float(os.environ.get("THRESH_SCORE", "0.75"))
THRESH_MARGIN = float(os.environ.get("THRESH_MARGIN", "0.015"))
EVAL_ABSTAIN = os.environ.get("EVAL_ABSTAIN", "0") == "1"
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Своё Вино — сканер", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_enc = None
_conn = None


@app.on_event("startup")
def _load():
    global _enc, _conn
    t = time.time()
    warmup()
    _enc = get_encoder()
    _conn = db.connect()
    if CROP_ENABLED:
        from crop import get_cropper; get_cropper()
    print(f"[startup] готово за {time.time()-t:.1f}с")


def _read_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="bad image")


def _rank(img: Image.Image, k: int = 5):
    img, _ = maybe_crop(img)
    q = _enc.embed([img])[0]
    res = db.search(_conn, q, _enc.model_name, k=k)
    top1 = res[0]["score"] if res else 0.0
    top2 = res[1]["score"] if len(res) > 1 else 0.0
    margin = round(top1 - top2, 4)
    in_catalog = bool(top1 >= THRESH_SCORE)
    return res, top1, margin, in_catalog


@app.post("/v1/eval/predict")
async def eval_predict(image: UploadFile = File(...)):
    res, top1, margin, in_catalog = _rank(_read_image(await image.read()))
    if EVAL_ABSTAIN and not in_catalog:
        return JSONResponse({"slug": None})
    return JSONResponse({"slug": res[0]["slug"] if res else None, "score": top1, "margin": margin})


@app.post("/v1/search")
async def search(image: UploadFile = File(...)):
    t = time.perf_counter()
    res, top1, margin, in_catalog = _rank(_read_image(await image.read()))
    return {"in_catalog": in_catalog, "elapsed_ms": round(1000 * (time.perf_counter() - t)),
            "confidence": {"top1_score": top1, "margin": margin,
                           "thresholds": {"score": THRESH_SCORE, "margin": THRESH_MARGIN}},
            "top1": res[0] if res else None, "results": res}


@app.get("/wine/{slug}")
def wine(slug: str):
    card = db.get_card(_conn, slug)
    if "name" not in card:
        raise HTTPException(status_code=404, detail="slug not found")
    return card


@lru_cache(maxsize=512)
def _thumb(path: str, h: int) -> bytes:
    img = Image.open(path)
    img.thumbnail((h * 2, h))                       # по высоте: бутылки вытянуты вертикально
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=85)
    return buf.getvalue()


@app.get("/ref/{slug}")
def ref_image(slug: str, h: int | None = None):
    """Эталонное фото вина. `h` — превью заданной высоты (для UI: оригиналы бывают по 2-3 тыс. px)."""
    d = FILTERED / slug
    if d.is_dir():
        imgs = sorted(p for p in d.iterdir() if p.is_file())
        if imgs:
            if h:
                return Response(_thumb(str(imgs[0]), max(64, min(h, 1600))), media_type="image/webp")
            return FileResponse(imgs[0])
    raise HTTPException(status_code=404, detail="no image")


@app.get("/health")
def health():
    return {"status": "ok",
            "model": _enc.model_name if _enc else None,
            "crop": CROP_ENABLED,
            "wines": db.count_wines(_conn) if _conn else 0,
            "models": db.list_models(_conn) if _conn else []}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
