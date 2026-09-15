# -*- coding: utf-8 -*-
"""FastAPI-сервис сканера вин.

Эндпоинты:
  POST /v1/eval/predict  — для скрипта-оценщика: multipart `image` -> {"slug": "..."}
  POST /v1/search        — богатый ответ: top-5, score, margin, confidence, in_catalog, карточка
  GET  /health           — статус
  GET  /wine/{slug}      — карточка по slug

Запуск:
  uv run uvicorn solution.service.app:app --host 0.0.0.0 --port 8080
  (или из папки service:  uv run uvicorn app:app --port 8080)

Пороги in_catalog — ЧЕРНОВЫЕ, калибруются на валидационном harness (env ниже).
"""
import os, io, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import FILTERED
from encoder import get_encoder
from search import CatalogIndex
from crop import maybe_crop, CROP_ENABLED

HERE = Path(__file__).parent
STATIC = HERE / "static"

# --- пороги отсечки «нет в каталоге» (черновые, калибровать!) ---
THRESH_SCORE = float(os.environ.get("THRESH_SCORE", "0.75"))   # мин. косинус top-1
THRESH_MARGIN = float(os.environ.get("THRESH_MARGIN", "0.015")) # мин. отрыв top1-top2
EVAL_ABSTAIN = os.environ.get("EVAL_ABSTAIN", "0") == "1"       # отказываться ли в eval-эндпоинте

app = FastAPI(title="Своё Вино — сканер", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_enc = None
_idx = None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/ref/{slug}")
def ref_image(slug: str):
    """Первое (эталонное) фото вина из filtered/<slug>/."""
    d = FILTERED / slug
    if d.is_dir():
        imgs = sorted(p for p in d.iterdir() if p.is_file())
        if imgs:
            return FileResponse(imgs[0])
    raise HTTPException(status_code=404, detail="no image")


@app.on_event("startup")
def _load():
    global _enc, _idx
    t = time.time()
    _enc = get_encoder()
    _idx = CatalogIndex()
    if CROP_ENABLED:                          # прогреть YOLO
        from crop import get_cropper
        get_cropper()
    print(f"[startup] готово за {time.time()-t:.1f}с (crop={CROP_ENABLED})")


def _read_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="bad image")


def _rank(img: Image.Image, k: int = 5):
    img, _ = maybe_crop(img)                 # кроп бутылки (если CROP_ENABLED)
    q = _enc.embed([img])[0]
    res = _idx.search(q, k=k)
    top1 = res[0]["score"] if res else 0.0
    top2 = res[1]["score"] if len(res) > 1 else 0.0
    margin = round(top1 - top2, 4)
    in_catalog = bool(top1 >= THRESH_SCORE and margin >= THRESH_MARGIN)
    return res, top1, margin, in_catalog


@app.post("/v1/eval/predict")
async def eval_predict(image: UploadFile = File(...)):
    """Плоский ответ для скрипта-оценщика."""
    res, top1, margin, in_catalog = _rank(_read_image(await image.read()), k=5)
    if EVAL_ABSTAIN and not in_catalog:
        return JSONResponse({"slug": None})
    return JSONResponse({"slug": res[0]["slug"] if res else None,
                         "score": top1, "margin": margin})


@app.post("/v1/search")
async def search(image: UploadFile = File(...)):
    """Богатый ответ для UI / отладки / метрик."""
    res, top1, margin, in_catalog = _rank(_read_image(await image.read()), k=5)
    return {
        "in_catalog": in_catalog,
        "confidence": {"top1_score": top1, "margin": margin,
                       "thresholds": {"score": THRESH_SCORE, "margin": THRESH_MARGIN}},
        "top1": res[0] if res else None,
        "results": res,
    }


@app.get("/wine/{slug}")
def wine(slug: str):
    card = _idx.card(slug)
    if "name" not in card:
        raise HTTPException(status_code=404, detail="slug not found")
    return card


@app.get("/health")
def health():
    return {"status": "ok", "model": _idx.meta.get("model"),
            "catalog_size": _idx.emb.shape[0], "crop": CROP_ENABLED}
