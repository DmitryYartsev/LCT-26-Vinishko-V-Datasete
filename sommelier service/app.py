# -*- coding: utf-8 -*-
"""FastAPI-сервис «цифровой сомелье». Stateless: переписка и профиль живут на фронте и
приходят в каждом запросе; на сервере — только логи (JSON-строки в stdout / LOG_FILE).

Эндпоинты:
  POST /v1/chat       {messages, profile} -> {reply, profile, picks, suggestions}   (LLM обновляет профиль)
  POST /v1/recommend  {profile, k}        -> {picks}                   (без LLM)
  POST /v1/match      {slug, profile}     -> {wine, score, reasons, checks, verdict}  (для отсканированной карточки)
  POST /v1/match_many {slugs, profile}    -> {items: [...как /v1/match]} (сканы за сессию, похожие)
  POST /v1/analogs    {slug, k}           -> {picks}                   (похожие по стилю вина других виноделен)
  POST /v1/stt        multipart `audio`   -> {text}
  GET  /v1/wine/{slug}, /v1/vocab, /health
"""
import os, sys, json, time, uuid, logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import store, llm, prefs
from prefs import COLORS, SWEETNESS, DISH_GROUPS
from seed import warmup

MAX_PER_MAKER = 2                      # разнообразие подборки: не больше N вин одного производителя
MAX_AUDIO = 10 * 1024 * 1024

log = logging.getLogger("sommelier")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))
if os.environ.get("LOG_FILE"):
    log.addHandler(logging.FileHandler(os.environ["LOG_FILE"], encoding="utf-8"))

app = FastAPI(title="Своё Вино — сомелье", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_conn = None
_wines: dict[str, dict] = {}
_regions: list[str] = []


def _log(event: str, **kw):
    log.info(json.dumps({"ts": round(time.time(), 3), "event": event, **kw}, ensure_ascii=False, default=str))


@app.on_event("startup")
def _load():
    global _conn, _wines, _regions
    warmup()
    _conn = store.connect()
    _wines = store.load_all(_conn)
    for w in _wines.values():
        w["dish_groups"] = prefs.dish_groups(w["dishes"])
    _regions = sorted({w["region"] for w in _wines.values() if w["region"]})
    print(f"[startup] сомелье: {len(_wines)} вин, LLM={llm.LLM_MODEL}, ключ={'есть' if llm.API_KEY else 'НЕТ'}")


class ChatIn(BaseModel):
    messages: list[dict] = Field(default_factory=list)
    profile: dict = Field(default_factory=dict)


class RecommendIn(BaseModel):
    profile: dict = Field(default_factory=dict)
    k: int = 5
    exclude: list[str] = Field(default_factory=list)


class MatchIn(BaseModel):
    slug: str
    profile: dict = Field(default_factory=dict)


class MatchManyIn(BaseModel):
    slugs: list[str] = Field(default_factory=list)
    profile: dict = Field(default_factory=dict)


class AnalogsIn(BaseModel):
    slug: str
    k: int = 4


def _public(w: dict) -> dict:
    return {k: w[k] for k in ("slug", "in_catalog", "title", "manufacturer", "category", "color", "sweetness",
                              "sparkling", "alcohol", "temperature", "grapes", "region", "dishes",
                              "rating", "url", "image")} | {"tagline": prefs.tagline(w), "sugar": prefs.sugar_label(w)}


def _jaccard(a, b) -> float:
    a, b = {x.lower() for x in a or []}, {x.lower() for x in b or []}
    return len(a & b) / len(a | b) if a and b else 0.0


def _closeness(ref: dict):
    """Тай-брейк для аналогов: у многих вин score=100 по стилю — ближе тот, у кого тот же
    набор сортов, регион и блюда."""
    return lambda w: (_jaccard(ref.get("grapes"), w.get("grapes")) + 0.5 * (w.get("region") == ref.get("region"))
                      + 0.3 * _jaccard(ref.get("dish_groups"), w.get("dish_groups")))


def _recommend(p: dict, k: int = 5, exclude=(), skip_maker: str | None = None, tiebreak=None) -> list[dict]:
    if prefs.is_empty(p):
        return []
    hits = store.note_hits(_conn, p["notes"]) if p["notes"] else None
    scored = []
    for w in _wines.values():
        if w["slug"] in exclude or (skip_maker and w.get("manufacturer") == skip_maker):
            continue
        m = prefs.score(w, p, hits)
        if m["score"] is not None:
            scored.append((m["score"], tiebreak(w) if tiebreak else 0, w.get("rating") or 0, w, m))
    scored.sort(key=lambda t: t[:3], reverse=True)
    out, per_maker = [], {}
    for s, _, _, w, m in scored:
        mk = w.get("manufacturer") or w["slug"]
        if per_maker.get(mk, 0) >= MAX_PER_MAKER:
            continue
        per_maker[mk] = per_maker.get(mk, 0) + 1
        out.append({**_public(w), "score": s, "reasons": [r for r in m["reasons"] if r["ok"]][:3]})
        if len(out) >= k:
            break
    return out


@app.post("/v1/chat")
def chat(body: ChatIn):
    rid = uuid.uuid4().hex[:12]
    t = time.time()
    prev = prefs.normalize(body.profile, _regions)
    try:
        turn = llm.sommelier_turn(body.messages, prev, _regions)
    except llm.LLMError as e:
        _log("chat_error", rid=rid, error=str(e))
        raise HTTPException(status_code=502, detail=str(e))
    p = prefs.normalize(turn["profile"], _regions)
    picks = _recommend(p)
    last_user = next((m.get("content") for m in reversed(body.messages) if m.get("role") == "user"), None)
    _log("chat", rid=rid, ms=round(1000 * (time.time() - t)), n_messages=len(body.messages),
         user=last_user, reply=turn["reply"], profile=p, picks=[x["slug"] for x in picks])
    return {"reply": turn["reply"], "profile": p, "picks": picks, "suggestions": turn["suggestions"]}


@app.post("/v1/recommend")
def recommend(body: RecommendIn):
    p = prefs.normalize(body.profile, _regions)
    picks = _recommend(p, max(1, min(body.k, 20)), set(body.exclude))
    _log("recommend", profile=p, picks=[x["slug"] for x in picks])
    return {"profile": p, "picks": picks}


@app.post("/v1/match")
def match(body: MatchIn):
    w = _wines.get(body.slug)
    if not w:
        raise HTTPException(status_code=404, detail="slug not found")
    p = prefs.normalize(body.profile, _regions)
    hits = store.note_hits(_conn, p["notes"], [w["slug"]]) if p["notes"] else None
    m = prefs.score(w, p, hits)
    _log("match", slug=body.slug, profile=p, score=m["score"])
    return {"wine": _public(w), **m}


@app.post("/v1/match_many")
def match_many(body: MatchManyIn):
    """Пакетный /v1/match (порядок slugs сохраняется, неизвестные пропускаются)."""
    p = prefs.normalize(body.profile, _regions)
    slugs = [s for s in dict.fromkeys(body.slugs[:50]) if s in _wines]
    hits = store.note_hits(_conn, p["notes"], slugs) if p["notes"] else None
    return {"items": [{"wine": _public(_wines[s]), **prefs.score(_wines[s], p, hits)} for s in slugs]}


@app.post("/v1/analogs")
def analogs(body: AnalogsIn):
    """Аналоги из других виноделен: профиль строится из самого вина (цвет, сладость, игристость,
    сорт, блюда) и скорится тем же детерминированным скорингом, что и запросы сомелье."""
    w = _wines.get(body.slug)
    if not w:
        raise HTTPException(status_code=404, detail="slug not found")
    p = prefs.normalize({"colors": [w["color"]] if w.get("color") else [],
                         "sweetness": [w["sweetness"]] if w.get("sweetness") else [],
                         "sparkling": w.get("sparkling"), "grapes": (w.get("grapes") or [])[:2],
                         "dishes": w.get("dish_groups") or []}, _regions)
    picks = _recommend(p, max(1, min(body.k, 12)), {w["slug"]}, w.get("manufacturer"), _closeness(w))
    _log("analogs", slug=body.slug, picks=[x["slug"] for x in picks])
    return {"picks": picks}


@app.post("/v1/stt")
async def stt(audio: UploadFile = File(...)):
    raw = await audio.read()
    if not raw or len(raw) > MAX_AUDIO:
        raise HTTPException(status_code=400, detail="empty or too large audio")
    t = time.time()
    try:
        text = llm.transcribe(raw, audio.filename or "voice.webm")
    except llm.LLMError as e:
        _log("stt_error", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))
    # само аудио не храним и не логируем — только расшифровку
    _log("stt", ms=round(1000 * (time.time() - t)), bytes=len(raw), text=text)
    return {"text": text}


@app.get("/v1/wine/{slug}")
def wine(slug: str):
    w = _wines.get(slug)
    if not w:
        raise HTTPException(status_code=404, detail="slug not found")
    return {**_public(w), "description": w["description"]}


@app.get("/v1/vocab")
def vocab():
    return {"colors": COLORS, "sweetness": SWEETNESS, "dishes": list(DISH_GROUPS), "regions": _regions}


@app.get("/health")
def health():
    return {"status": "ok", "wines": len(_wines), "llm_model": llm.LLM_MODEL, "stt_model": llm.STT_MODEL,
            "api_key": bool(llm.API_KEY)}
