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
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- конфиг пайплайна (OmegaConf) читается ДО импорта env-зависимых модулей ---
from pipeline_config import load_config, apply_retrieval_env
import ocr_rerank

_cfg = load_config()
apply_retrieval_env(_cfg)

from paths import FILTERED
import db
from encoder import get_encoder
from crop import (maybe_crop, maybe_label_crop, ocr_label_crop, get_cropper,
                  get_label_cropper, use_query_policy,
                  CROP_ENABLED, USE_LABEL_BRANCH, LABEL_SUFFIX)
from build_index import warmup
import preflight

THRESH_SCORE = float(os.environ.get("THRESH_SCORE", "0.81"))       # HI: карточка сразу
THRESH_SCORE_LO = float(os.environ.get("THRESH_SCORE_LO", "0.70"))  # LO: ниже — «нет в каталоге»
OCR_CONFIRM_CONF = float(os.environ.get("OCR_CONFIRM_CONF", "0.80"))  # уверенность OCR-переранка
OCR_AGREE_CONF = float(os.environ.get("OCR_AGREE_CONF", "0.80"))      # уверенность «OCR согласен»
THRESH_MARGIN = float(os.environ.get("THRESH_MARGIN", "0.015"))
# SEARCH_PIPELINE: combined = бутылка + этикетка (макс. score), bottle = только бутылка,
# label = только этикетка. Отбор ветки — по максимальному top-1 score.
SEARCH_PIPELINE = os.environ.get("SEARCH_PIPELINE", "combined").lower()
EVAL_ABSTAIN = os.environ.get("EVAL_ABSTAIN", "0") == "1"
# ширина short-list: и ответ сервиса, и пул кандидатов для OCR-rerank берутся из
# конфига (retrieval.top_k) — иначе сервис и standalone-прогон pipeline.py
# расходятся (в pipeline.py k читается из конфига).
TOP_K = int(_cfg.retrieval.get("top_k", 5))
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Своё Вино — сканер", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_enc = None
_conn = None
# OCR-rerank (строится в startup, если ocr.enabled)
_ocr_enabled = bool(_cfg.ocr.enabled)
_ocr_extractor = None
_ocr_matcher = None
_ocr_verifier = None
_fusion = None


@app.on_event("startup")
def _load():
    global _enc, _conn, _ocr_extractor, _ocr_matcher, _ocr_verifier, _fusion
    t = time.time()
    if preflight.check(_cfg, where='контейнер ml'):     # понятное сообщение, а не traceback
        raise SystemExit(3)
    warmup()
    _enc = get_encoder()
    _conn = db.connect()
    if CROP_ENABLED:
        from crop import get_cropper; get_cropper()
    if USE_LABEL_BRANCH:
        get_label_cropper()
    if _ocr_enabled:
        _ocr_extractor = ocr_rerank.OcrExtractor(_cfg)
        _ocr_matcher = ocr_rerank.CsvMatcher(_cfg)
        crop_fn = None
        if _cfg.visual.enabled:
            from crop import maybe_crop as _mc, maybe_label_crop as _mlc, index_policy

            def _ref_crop(img):
                # эталон каталога: индексная политика, даже если запрос идёт query_crop_*
                with index_policy():
                    bottle, _ = _mc(img)
                    label, found, _ = _mlc(bottle)
                return label if found else bottle

            crop_fn = _ref_crop
        _ocr_verifier = ocr_rerank.VisualVerifier(_cfg, _enc.embed, crop_fn)
        print(f"[startup] OCR-rerank: model={_cfg.ocr.model}, refs={len(_ocr_matcher.entries)}")
        if getattr(_cfg, 'fusion', None) is not None and bool(_cfg.fusion.get('enabled', False)):
            from pipeline import build_fusion
            _fusion = build_fusion(_cfg, _ocr_matcher)
            print(f"[startup] fusion-rerank: {'on' if _fusion else 'off'}")
    print(f"[startup] готово за {time.time()-t:.1f}с (pipeline={SEARCH_PIPELINE}, ocr={_ocr_enabled})")


def _read_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="bad image")


def _rank(img: Image.Image, k: int | None = None):
    """Скоринг по веткам: A = кроп бутылки (индекс model), B = кроп этикетки (model#label).
    Ответ — ветка с максимальным top-1 score; margin считается внутри выбранной ветки."""
    k = TOP_K if k is None else k
    model = _enc.model_name
    use_query_policy()                                # кроп ЗАПРОСА: политика query_crop_*
    bottle, _ = maybe_crop(img)                       # бутылочный кроп общий для обеих веток
    want_a = SEARCH_PIPELINE in ("bottle", "combined")
    want_b = USE_LABEL_BRANCH and SEARCH_PIPELINE in ("label", "combined")
    label, found, _ = maybe_label_crop(bottle)
    if want_b and not found and USE_LABEL_BRANCH:
        label = bottle                                # этикетка не найдена — fallback на бутылку
    _rank.bottle = bottle
    _rank.label = label
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
    _rank.res = res
    _rank.branches = {
        "bottle": {"top1": res_a[0]["slug"], "score": res_a[0]["score"], "results": res_a} if res_a else None,
        "label": {"top1": res_b[0]["slug"], "score": res_b[0]["score"], "results": res_b} if res_b else None,
    }
    top1 = res[0]["score"] if res else 0.0
    top2 = res[1]["score"] if len(res) > 1 else 0.0
    margin = round(top1 - top2, 4)
    # база гейта: «явно ниже LO — вина нет в каталоге»; точное решение принимает _gate()
    # после OCR-реранка (там доступны stage/csv_confidence).
    in_catalog = bool(top1 >= THRESH_SCORE_LO)
    return res, top1, margin, in_catalog


def _rerank(img: Image.Image, pre_ocr_slug, pre_ocr_score) -> dict:
    """OCR-rerank поверх retrieval-ответа (см. ocr_rerank.rerank).

    VLM подаётся ВЫПРЯМЛЕННЫЙ кроп этикетки (``ocr.use_label_crop``), а не сырое
    полочное фото; ``_rank.label`` остаётся для визуальной проверки (препроцесс
    совпадает с индексом).
    """
    if not _ocr_enabled:
        return {"final_slug": pre_ocr_slug, "stage": "retrieval", "reason": "ocr_disabled"}
    ocr_crop = None
    if bool(getattr(_cfg.ocr, "use_label_crop", True)):
        bottle = getattr(_rank, "bottle", None)
        if bottle is not None:
            try:
                ocr_crop, _ = ocr_label_crop(bottle)
            except Exception as e:  # noqa: BLE001 — OCR-кроп не должен ронять запрос
                print(f"[ocr] ocr_label_crop failed: {e}", flush=True)
    rr = ocr_rerank.rerank(_cfg, img, getattr(_rank, "label", img),
                           pre_ocr_slug, pre_ocr_score,
                           _ocr_extractor, _ocr_matcher, _ocr_verifier,
                           ocr_image=ocr_crop,
                           pre_ocr_candidates=[r["slug"] for r in (getattr(_rank, "res", None) or [])],
                           bottle_crop=getattr(_rank, "bottle", None))
    if _fusion is not None and rr.get("ocr_fields"):
        res = getattr(_rank, "res", None) or []
        try:
            fr = _fusion.rerank(rr["ocr_fields"],
                                [{"slug": r["slug"], "score": r["score"]} for r in res],
                                query_crop=getattr(_rank, "label", img),
                                verifier=_ocr_verifier,
                                image_k=int(_cfg.fusion.get("image_k", 30)),
                                text_k=int(_cfg.fusion.get("text_k", 10)))
            if fr.get("final_slug"):
                rr["final_slug"] = fr["final_slug"]
                rr["stage"] = "fusion"
                rr["fusion_pool_size"] = fr.get("pool_size")
                rr["fusion_top_candidates"] = fr.get("top_candidates")
        except Exception as e:  # noqa: BLE001 — fusion не должен ронять запрос
            print(f"[fusion] {e}", flush=True)
    return rr


def _gate(top1: float, rr: dict) -> tuple:
    """Решение «есть в каталоге»: показать карточку или сценарий «похожие вина».

    * ``top1 >= THRESH_SCORE`` (HI) — сильное визуальное совпадение, карточка сразу;
    * ``top1 <  THRESH_SCORE_LO`` (LO) — слишком слабо, вина в каталоге нет;
    * между LO и HI — только если OCR уверенно ПОДТВЕРДИЛ карточку:
      либо переставил ответ (``stage=ocr_rerank``, ``csv_confidence >= OCR_CONFIRM_CONF``),
      либо согласился с кандидатом retrieval (``reason=csv_agrees``, ``>= OCR_AGREE_CONF``).

    Калибровка на 185 позитивах и 11 негативах — ``reports/27_gate_negatives.md``.
    Возвращает ``(in_catalog, reason)`` — reason попадает в ответ как ``gate_reason``.
    """
    if top1 >= THRESH_SCORE:
        return True, "visual_hi"
    if top1 < THRESH_SCORE_LO:
        return False, "low_score"
    conf = float(rr.get("csv_confidence") or 0.0)
    if rr.get("stage") == "ocr_rerank" and conf >= OCR_CONFIRM_CONF:
        return True, "ocr_confirms"
    if rr.get("reason") == "csv_agrees" and conf >= OCR_AGREE_CONF:
        return True, "ocr_agrees"
    return False, "ocr_not_confirmed"


@app.post("/v1/eval/predict")
async def eval_predict(image: UploadFile = File(...)):
    img = _read_image(await image.read())
    res, top1, margin, in_catalog = _rank(img)
    pre_ocr_slug = res[0]["slug"] if res else None
    rr = _rerank(img, pre_ocr_slug, top1)
    final_slug = rr["final_slug"]
    if EVAL_ABSTAIN and not in_catalog:
        return JSONResponse({"slug": None})
    return JSONResponse({"slug": final_slug, "score": top1,
                         "margin": margin, "branch": getattr(_rank, "branch", None),
                         "stage": rr["stage"], "pre_ocr_slug": pre_ocr_slug,
                         "csv_confidence": rr.get("csv_confidence"),
                         "csv_margin": rr.get("csv_margin"),
                         "visual_similarity": rr.get("visual_similarity")})


@app.post("/v1/search")
async def search(image: UploadFile = File(...)):
    t = time.perf_counter()
    img = _read_image(await image.read())
    res, top1, margin, in_catalog = _rank(img)
    pre_ocr_slug = res[0]["slug"] if res else None
    rr = _rerank(img, pre_ocr_slug, top1)
    in_catalog, gate_reason = _gate(top1, rr)
    return {"in_catalog": in_catalog,
            "gate_reason": gate_reason,
            "elapsed_ms": round(1000 * (time.perf_counter() - t)),
            "pipeline": SEARCH_PIPELINE, "branch": getattr(_rank, "branch", None),
            "branches": getattr(_rank, "branches", None),
            "confidence": {"top1_score": top1, "margin": margin,
                           "thresholds": {"score": THRESH_SCORE, "margin": THRESH_MARGIN}},
            "top1": res[0] if res else None, "results": res,
            "final_slug": rr["final_slug"], "stage": rr["stage"],
            "pre_ocr_slug": pre_ocr_slug,
            "csv_confidence": rr.get("csv_confidence"),
            "csv_margin": rr.get("csv_margin"),
            "visual_similarity": rr.get("visual_similarity"),
            "ocr_input": rr.get("ocr_input"),
            "ocr_fields": rr.get("ocr_fields"),
            "top_candidates": rr.get("top_candidates")}


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
            "pipeline": SEARCH_PIPELINE,
            "label_branch": USE_LABEL_BRANCH,
            "ocr_rerank": _ocr_enabled,
            "ocr_model": str(_cfg.ocr.model) if _ocr_enabled else None,
            "models": db.list_models(_conn) if _conn else []}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
