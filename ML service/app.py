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
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import FILTERED
import db
from encoder import get_encoder
from crop import (maybe_crop, maybe_label_crop, get_cropper, get_label_cropper,
                  CROP_ENABLED, USE_LABEL_BRANCH, LABEL_SUFFIX)
from build_index import warmup

THRESH_SCORE = float(os.environ.get("THRESH_SCORE", "0.75"))
THRESH_MARGIN = float(os.environ.get("THRESH_MARGIN", "0.015"))
# SEARCH_PIPELINE: combined = бутылка + этикетка (макс. score), bottle = только бутылка,
# label = только этикетка. Отбор ветки — по максимальному top-1 score (см. norm_exp).
SEARCH_PIPELINE = os.environ.get("SEARCH_PIPELINE", "combined").lower()
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
    if USE_LABEL_BRANCH:
        get_label_cropper()
    print(f"[startup] готово за {time.time()-t:.1f}с (pipeline={SEARCH_PIPELINE})")


def _read_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="bad image")


def _rank(img: Image.Image, k: int = 5):
    """Скоринг по веткам: A = кроп бутылки (индекс model), B = кроп этикетки (model#label).
    Ответ — ветка с максимальным top-1 score; margin считается внутри выбранной ветки."""
    model = _enc.model_name
    bottle, _ = maybe_crop(img)                       # бутылочный кроп общий для обеих веток
    want_a = SEARCH_PIPELINE in ("bottle", "combined")
    want_b = USE_LABEL_BRANCH and SEARCH_PIPELINE in ("label", "combined")
    label, found, _ = maybe_label_crop(bottle)
    if want_b and not found and USE_LABEL_BRANCH:
        label = bottle                                # этикетка не найдена — fallback на бутылку
    res_a = db.search(_conn, _enc.embed([bottle])[0], model, k=k) if want_a else None
    res_b = db.search(_conn, _enc.embed([label])[0], model + LABEL_SUFFIX, k=k) if want_b else None
    if res_a is None:
        res, branch = res_b, "label"
    elif res_b is None:
        res, branch = res_a, "bottle"
    elif res_a[0]["score"] >= res_b[0]["score"]:
        res, branch = res_a, "bottle"
    else:
        res, branch = res_b, "label"
    _rank.branch = branch
    _rank.branches = {
        "bottle": {"top1": res_a[0]["slug"], "score": res_a[0]["score"], "results": res_a} if res_a else None,
        "label": {"top1": res_b[0]["slug"], "score": res_b[0]["score"], "results": res_b} if res_b else None,
    }
    top1 = res[0]["score"] if res else 0.0
    top2 = res[1]["score"] if len(res) > 1 else 0.0
    margin = round(top1 - top2, 4)
    in_catalog = bool(top1 >= THRESH_SCORE and margin >= THRESH_MARGIN)
    return res, top1, margin, in_catalog


@app.post("/v1/eval/predict")
async def eval_predict(image: UploadFile = File(...)):
    res, top1, margin, in_catalog = _rank(_read_image(await image.read()))
    if EVAL_ABSTAIN and not in_catalog:
        return JSONResponse({"slug": None})
    return JSONResponse({"slug": res[0]["slug"] if res else None, "score": top1,
                         "margin": margin, "branch": getattr(_rank, "branch", None)})


@app.post("/v1/search")
async def search(image: UploadFile = File(...)):
    res, top1, margin, in_catalog = _rank(_read_image(await image.read()))
    return {"in_catalog": in_catalog,
            "pipeline": SEARCH_PIPELINE, "branch": getattr(_rank, "branch", None),
            "branches": getattr(_rank, "branches", None),
            "confidence": {"top1_score": top1, "margin": margin,
                           "thresholds": {"score": THRESH_SCORE, "margin": THRESH_MARGIN}},
            "top1": res[0] if res else None, "results": res}


@app.get("/wine/{slug}")
def wine(slug: str):
    card = db.get_card(_conn, slug)
    if "name" not in card:
        raise HTTPException(status_code=404, detail="slug not found")
    return card


@app.get("/ref/{slug}")
def ref_image(slug: str):
    d = FILTERED / slug
    if d.is_dir():
        imgs = sorted(p for p in d.iterdir() if p.is_file())
        if imgs:
            return FileResponse(imgs[0])
    raise HTTPException(status_code=404, detail="no image")


@app.get("/health")
def health():
    return {"status": "ok",
            "model": _enc.model_name if _enc else None,
            "crop": CROP_ENABLED,
            "pipeline": SEARCH_PIPELINE,
            "label_branch": USE_LABEL_BRANCH,
            "models": db.list_models(_conn) if _conn else []}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
