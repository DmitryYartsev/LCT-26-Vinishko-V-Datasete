# -*- coding: utf-8 -*-
"""OpenRouter (OpenAI-совместимый chat/completions): диалог сомелье -> {reply, profile} и STT.

STT — тем же chat/completions с аудио-частью `input_audio` (отдельного transcription-эндпоинта
у OpenRouter нет). Модели меняются через env без правки кода.
"""
import os, re, json, base64, subprocess, tempfile
from pathlib import Path
import httpx
from prefs import COLORS, SWEETNESS, DISH_GROUPS

BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")
STT_MODEL = os.environ.get("STT_MODEL", "google/gemini-2.5-flash")
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))
HISTORY_TURNS = int(os.environ.get("HISTORY_TURNS", "12"))   # сколько последних реплик слать в LLM

SYSTEM = """Ты — цифровой сомелье сервиса «Своё Вино» (российские вина, проект Россельхозбанка).
Задача: в живом коротком диалоге понять, какое вино нужно человеку, и вести его профиль предпочтений.

Правила:
- Отвечай по-русски, тепло и кратко (1–3 предложения). Не больше одного уточняющего вопроса за раз, не устраивай анкету.
- Не называй конкретные вина и производителей: подборку из каталога интерфейс покажет сам под твоим ответом.
- Если человек говорит о блюде или поводе — сам выведи, как сомелье, подходящие цвет / сладость / игристость
  (стейк → красное сухое; устрицы → белое сухое или брют; десерт → сладкое; праздник → игристое),
  но явные пожелания человека важнее.
- Профиль — накопленное состояние всего диалога: сохраняй прежние поля, если человек их не менял; убирай, если передумал.
- На темы, не связанные с вином, едой и поводами, мягко возвращай разговор к выбору вина.

Верни СТРОГО JSON без markdown: {"reply": "...", "profile": {...}, "suggestions": [...]}
suggestions — 0–4 коротких (2–4 слова) готовых ответа человека на твой уточняющий вопрос
(«К стейку», «Сухое», «Не важно»); если вопроса нет — [].
Поля profile (где указан список — только значения из него):
- colors: список из %(colors)s
- sweetness: список из %(sweetness)s (брют и экстра брют по сахару = «сухое»)
- sparkling: true | false | null (null — не важно)
- dishes: список из %(dishes)s
- regions: список из %(regions)s
- grapes: сорта винограда по-русски («Саперави», «Каберне Совиньон», «Рислинг»…)
- notes: до 5 вкусо-ароматических слов для поиска по описаниям («вишня», «минеральность», «дуб», «цитрус»)
- exclude_colors / exclude_sweetness / exclude_grapes: чего человек НЕ хочет
- alcohol_max: число (%% об.) или null
- occasion: повод коротко или null
- summary: одна строка — суть запроса («Красное сухое к стейку на день рождения»)"""


class LLMError(RuntimeError):
    pass


def _post(model: str, messages: list, **extra) -> str:
    if not API_KEY:
        raise LLMError("OPENROUTER_API_KEY не задан")
    r = httpx.post(f"{BASE_URL}/chat/completions", timeout=TIMEOUT,
                   headers={"Authorization": f"Bearer {API_KEY}", "X-Title": "Svoe Vino Sommelier"},
                   json={"model": model, "messages": messages, **extra})
    if r.status_code != 200:
        raise LLMError(f"OpenRouter {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"] or ""


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)          # модель добавила текст вокруг JSON
        if m:
            return json.loads(m.group(0))
        raise LLMError(f"LLM вернула не JSON: {text[:200]}")


def sommelier_turn(messages: list[dict], profile: dict, regions: list[str]) -> dict:
    """messages: [{role: user|assistant, content}] — вся переписка с фронта. -> {reply, profile}."""
    system = SYSTEM % {"colors": COLORS, "sweetness": SWEETNESS, "dishes": list(DISH_GROUPS), "regions": regions}
    history = [{"role": m["role"], "content": str(m["content"])[:2000]}
               for m in messages[-HISTORY_TURNS:] if m.get("role") in ("user", "assistant") and m.get("content")]
    msgs = [{"role": "system", "content": system},
            {"role": "system", "content": "Текущий профиль: " + json.dumps(profile, ensure_ascii=False)},
            *history]
    out = _parse_json(_post(LLM_MODEL, msgs, temperature=0.4, response_format={"type": "json_object"}))
    sugg = out.get("suggestions") if isinstance(out.get("suggestions"), list) else []
    return {"reply": str(out.get("reply") or "").strip(), "profile": out.get("profile") or {},
            "suggestions": [str(x).strip()[:40] for x in sugg if str(x).strip()][:4]}


def _to_wav(raw: bytes, filename: str) -> bytes:
    """Любой контейнер браузера (webm/opus, mp4/aac, ogg) -> wav 16 кГц моно через ffmpeg."""
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / ("in" + (Path(filename).suffix or ".webm"))
        dst = Path(d) / "out.wav"
        src.write_bytes(raw)
        p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                            "-ac", "1", "-ar", "16000", str(dst)], capture_output=True)
        if p.returncode != 0:
            raise LLMError(f"ffmpeg: {p.stderr.decode(errors='ignore')[:200]}")
        return dst.read_bytes()


def transcribe(raw: bytes, filename: str = "voice.webm") -> str:
    wav = _to_wav(raw, filename)
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Дословно расшифруй эту русскую речь. Верни только текст расшифровки, "
                                 "без кавычек и комментариев. Если речи нет — верни пустую строку."},
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
    ]}]
    return _post(STT_MODEL, msgs, temperature=0).strip()
