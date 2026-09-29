# -*- coding: utf-8 -*-
"""OCR-rerank поверх image retrieval (SigLIP2 + pgvector).

  1. ``OcrExtractor`` — вызов VLM (``ocr.model`` через OpenRouter) на выровненном
     кропе этикетки: поля ``winery/grape/color/sugar/sparkling/line/year/additional_text``.
     Промпт — ``paths.prompt_file``.
  2. ``CsvMatcher`` — взвешенное сопоставление полей с записями каталога
     (``found_in_catalog_corrected.csv`` + OCR-разметка каталожных фото). Веса —
     ``csv_match.weights``: сорт и линейка — главные различители вин одной линейки.
  3. ``decide`` — ответ retrieval заменяется, только если CSV-уверенность лучшего
     кандидата выше CSV-уверенности ответа retrieval на ``decision_margin``.
  4. ``VisualVerifier`` — косинус запроса с эталонным фото нового кандидата
     (мягкий сигнал, блокирует только при ``visual.block_on_mismatch``).

``rerank()`` возвращает финальный slug, stage и диагностику.

Параметры читаются из OmegaConf-конфига (см. ``config/pipeline.yaml``).
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import requests
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]   # корень репозитория (для относительных путей)

# ---------------------------------------------------------------- транслит / канонизация
_RU_LAT = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
}

_LAT_CANON = [('kh', 'h'), ('x', 'h'), ('ts', 'c'), ('ya', 'a'), ('yu', 'u'),
              ('yo', 'e'), ('zh', 'j'), ('shch', 'sh'), ('sch', 'sh'),
              ('sh', 's'), ('y', 'i'), ('w', 'v'), ('ck', 'k'), ('q', 'k')]


def _canon(word: str) -> str:
    w = word.lower().strip()
    w = unicodedata.normalize('NFKD', w)
    w = ''.join(ch for ch in w if not unicodedata.combining(ch))
    for a, b in _LAT_CANON:
        w = w.replace(a, b)
    return re.sub(r'[^a-z0-9]', '', w)


def _translit(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch in _RU_LAT:
            out.append(_RU_LAT[ch])
        elif ch.isascii():
            out.append(ch)
    return ''.join(out)


def canon_word(word: str) -> str:
    return _canon(_translit(word))


def parse_slug(slug: str):
    """'...krasnoe-suhoe-145' -> (['...','krasnoe','suhoe'], '145')."""
    parts = slug.lower().split('-')
    digits = ''
    while parts and parts[-1].isdigit():
        digits = parts.pop() + digits
    return [p for p in parts if p], digits


# год в свободном тексте названия/описания: 1950-1999 и 2000-2039
_YEAR_TEXT_RE = re.compile(r'\b(19[5-9]\d|20[0-3]\d)\b')


def year_code(raw) -> str:
    """Год -> 4 цифры ('2013'); '' — не распознан.

    Принимаем ТОЛЬКО явный 4-значный год в диапазоне 1950-2039. Короткие
    (2-3-значные) суффиксы слагов — это НЕ винтаж, а крепость/объём
    (``-12`` -> 12%, ``-135`` -> 13.5%, ``-750`` -> 0.75 л): в каталоге таких
    суффиксов ~1800 из 2108, и год в названии записи совпадал с ними лишь в
    72 случаях. Раньше эти суффиксы декодировались в «год», из-за чего год,
    прочитанный VLM с этикетки, в 96% случаев «не совпадал» с записью и
    штрафовал правильного кандидата. Настоящий винтаж в слаге пишется
    полностью (``aligote-barrel-2024``), его и берём.
    """
    d = re.sub(r'\D', '', str(raw or ''))
    if len(d) == 4 and 1950 <= int(d) <= 2039:
        return d
    return ''


def year_from_text(*texts) -> str:
    """Первый 4-значный год (1950-2039) среди текстов или ''."""
    for t in texts:
        m = _YEAR_TEXT_RE.search(str(t or ''))
        if m:
            return m.group(0)
    return ''


def year_sim(entry_year: str, query_year: str) -> float:
    """Совпадение винтажей: оба нормализованы к 4 цифрам -> 1.0 / 0.0."""
    if not entry_year or not query_year:
        return 0.0
    return 1.0 if entry_year == query_year else 0.0


# ---------------------------------------------------------------- синонимы сортов / типа
_GRAPE_ALIAS = {
    'kaberne': 'cabernet', 'cabernet': 'cabernet', 'sauvignon': 'sauvignon',
    'sovinon': 'sauvignon', 'shardone': 'chardonnay', 'chardonnay': 'chardonnay',
    'pino': 'pinot', 'pinot': 'pinot', 'nuar': 'noir', 'noir': 'noir',
    'merlo': 'merlot', 'merlot': 'merlot', 'risling': 'riesling',
    'riesling': 'riesling', 'saperavi': 'saperavi', 'rkaciteli': 'rkatsiteli',
    'rkatsiteli': 'rkatsiteli', 'muskat': 'muscat', 'muscat': 'muscat',
    'sira': 'syrah', 'sirah': 'syrah', 'syrah': 'syrah', 'aligote': 'aligote',
    'kokur': 'kokur', 'traminer': 'traminer', 'gewurz': 'gewurztraminer',
    'gevyurc': 'gewurztraminer', 'gevyurctraminer': 'gewurztraminer',
    'malbek': 'malbec', 'malbec': 'malbec',
    # доп. сорта/формы, встречающиеся в каталоге и на этикетках
    'blan': 'blanc', 'blanc': 'blanc', 'gri': 'gris', 'gris': 'gris',
    'ottonel': 'ottonel', 'krasnostop': 'krasnostop', 'bastardo': 'bastardo',
    'cimlyanskij': 'tsimlyanskij', 'cimlyanskiy': 'tsimlyanskij',
    'tsimlyanskij': 'tsimlyanskij', 'tsimlyanskiy': 'tsimlyanskij',
    'aleatiko': 'aleatico', 'kefesiya': 'kefessiya', 'sabate': 'sabat',
    'sovinonblan': 'sauvignonblanc', 'shiraz': 'syrah',
    # склеенные canon-формы (canon_word убирает пробелы)
    'kabernesovinon': 'cabernetsauvignon', 'kabernefran': 'cabernetfranc',
    'pinonuair': 'pinotnoir', 'pinonuar': 'pinotnoir', 'pinogri': 'pinotgris',
    'sovinonblan': 'sauvignonblanc', 'myuskatelye': 'muscat',
    'krasnostopzolotovskiy': 'krasnostop', 'cimlyanskiycherniy': 'tsimlyanskij',
    'tsimlyanskiycherniy': 'tsimlyanskij',
}

_COLOR_WORDS = {
    'белое': 'white', 'белый': 'white', 'белая': 'white', 'белые': 'white',
    'белого': 'white', 'white': 'white',
    'красное': 'red', 'красный': 'red', 'красная': 'red', 'красные': 'red',
    'красного': 'red', 'red': 'red',
    'розовое': 'rose', 'розовый': 'rose', 'розовая': 'rose', 'розовые': 'rose',
    'розового': 'rose', 'розе': 'rose', 'rose': 'rose', 'pink': 'rose',
    # ВАЖНО: «игристое/шампанское» — это НЕ цвет, а отдельный КЛАСС вина
    # (см. _SPARKLING_WORDS и поле sparkling).
    # оранжевое (skin-contact) — отдельная категория в каталоге (16 записей);
    # без маппинга такие записи вовсе не имели токена цвета и не штрафовались.
    'оранжевое': 'orange', 'оранжевый': 'orange', 'оранжевые': 'orange',
    'оранжевого': 'orange', 'оранж': 'orange', 'orange': 'orange',
}

# Оттенок из колонки ``Цвет`` (Светло-соломенный / Тёмно-рубиновый / ...)
# -> категория цвета. Нужен как fallback, когда ``Категория`` пуста/нестандартна.
_SHADE_COLOR = [
    ('розов', 'rose'), ('розе', 'rose'), ('лососев', 'rose'),
    ('рубин', 'red'), ('гранат', 'red'), ('вишн', 'red'), ('фиолет', 'red'),
    ('пурпур', 'red'), ('красн', 'red'), ('кирпичн', 'red'),
    ('солом', 'white'), ('золот', 'white'), ('лимон', 'white'),
    ('зеленоват', 'white'), ('прозрачн', 'white'), ('кристальн', 'white'),
    ('светло-желт', 'white'), ('бледно-желт', 'white'), ('желт', 'white'),
    ('оранж', 'orange'), ('янтар', 'orange'), ('мед', 'orange'), ('лукович', 'orange'),
]


def _shade_color_tokens(text: str) -> list:
    """Оттенок (``Светло-соломенный``/``Тёмно-рубиновый``) -> категория цвета."""
    s = (text or '').lower()
    return sorted({v for k, v in _SHADE_COLOR if k in s})
_SWEET_WORDS = {
    'сухое': 'dry', 'сухой': 'dry', 'сухая': 'dry', 'сухие': 'dry',
    'сухого': 'dry', 'dry': 'dry',
    'брют': 'brut', 'brut': 'brut',
    'полусухое': 'semidry', 'полусухой': 'semidry', 'полусухие': 'semidry',
    'полусухого': 'semidry', 'semi-dry': 'semidry', 'semidry': 'semidry',
    'полусладкое': 'semisweet', 'полусладкий': 'semisweet', 'полусладкие': 'semisweet',
    'полусладкого': 'semisweet', 'semi-sweet': 'semisweet', 'semisweet': 'semisweet',
    'сладкое': 'sweet', 'сладкий': 'sweet', 'сладкие': 'sweet',
    'сладкого': 'sweet', 'sweet': 'sweet',
}
# транслит-формы цвета/сладости, как они встречаются в слагах каталога
_SLUG_COLOR = {'beloe': 'white', 'krasnoe': 'red', 'rozovoe': 'rose'}
_SLUG_SWEET = {'suhoe': 'dry', 'polusuhoe': 'semidry', 'polusladkoe': 'semisweet',
               'sladkoe': 'sweet', 'bryut': 'brut', 'ekstrabryut': 'extrabrut',
               'extrabrut': 'extrabrut'}

# «игристое/шампанское» — отдельный КЛАСС вина (НЕ цвет). Тихие = без этих надписей.
# Для мэтчинга игристое и шампанское сводим к одному классу 'sparkling'.
_SPARKLING_WORDS = {
    'игристое': 'sparkling', 'игристые': 'sparkling', 'игристый': 'sparkling',
    'игристого': 'sparkling', 'игрист': 'sparkling', 'sparkling': 'sparkling',
    'шампанское': 'sparkling', 'шампанский': 'sparkling', 'шампанского': 'sparkling',
    'champagne': 'sparkling',
    # брют встречается только у игристых — это тоже маркер класса
    'брют': 'sparkling', 'brut': 'sparkling',
    'тихое': 'still', 'тихие': 'still', 'тихий': 'still', 'still': 'still',
}
# маркеры игристого в слаге каталога (bryut/brut/spumante/...)
_SLUG_SPARK = ('bryut', 'brut', 'ekstrabryut', 'extrabrut', 'igristoe',
               'spumante', 'shampanskoe', 'shampan')


def _wine_sparkling_tokens(*texts) -> list:
    """Тексты -> токен класса: 'sparkling' (игристое/шампанское) или 'still'."""
    out = set()
    for text in texts:
        s = (text or '').lower()
        for k, v in _SPARKLING_WORDS.items():
            if re.search(r'\b' + re.escape(k) + r'\b', s):
                out.add(v)
    return sorted(out)


def _slug_spark_tokens(slug: str) -> list:
    toks = set((slug or '').lower().split('-'))
    return ['sparkling'] if any(t in _SLUG_SPARK for t in toks) else []


def _norm_tokens(text: str) -> list:
    """Текст -> канонизированные токены (латиница, без стоп-мусора)."""
    if not text:
        return []
    w = canon_word(text)
    return [t for t in re.split(r'[^a-z0-9]+', w) if len(t) >= 2]


def _grape_tokens(text: str) -> list:
    """Сорт(а) -> канонические токены. Купаж делится по запятым/«и»/слэшу.

    ВАЖНО: ``canon_word`` склеивает слова («Каберне Совиньон» -> ``kabernesovinon``),
    поэтому КУПАЖ («Каберне Совиньон, Мерло») сначала режется на части, и уже
    каждая часть канонизируется как единый сорт.
    """
    if not text:
        return []
    parts = re.split(r'[,;/+]|\\bи\\b', str(text).lower())
    out = []
    for part in parts:
        for tok in _norm_tokens(part):
            out.append(_GRAPE_ALIAS.get(tok, tok))
    return [t for t in out if t]


def _wine_color_tokens(*texts) -> list:
    """Тексты (``wine_type``/``color``/доп.) -> канонические токены цвета."""
    out = set()
    for text in texts:
        s = (text or '').lower()
        for k, v in _COLOR_WORDS.items():
            if re.search(r'\b' + re.escape(k) + r'\b', s):
                out.add(v)
    return sorted(out)


def _wine_sweet_tokens(*texts) -> list:
    """Тексты (``wine_type``/``sugar``/доп.) -> токены сладости/типа (сухое/брют/...)."""
    out = set()
    for text in texts:
        s = (text or '').lower()
        for k, v in _SWEET_WORDS.items():
            if re.search(r'\b' + re.escape(k) + r'\b', s):
                out.add(v)
    return sorted(out)


def _wine_type_tokens(wine_type: str) -> list:
    """wine_type (LLM) -> канонические токены цвета + сладости (объединение)."""
    return sorted(set(_wine_color_tokens(wine_type)) | set(_wine_sweet_tokens(wine_type)))


def _slug_color_tokens(slug: str) -> list:
    """Из слага достаём канонические токены цвета."""
    return sorted({_SLUG_COLOR[t] for t in slug.lower().split('-') if t in _SLUG_COLOR})


def _slug_sweet_tokens(slug: str) -> list:
    """Из слага достаём канонические токены сладости/типа."""
    return sorted({_SLUG_SWEET[t] for t in slug.lower().split('-') if t in _SLUG_SWEET})


def _slug_category_tokens(slug: str) -> list:
    """Цвет + сладость из слага (обратная совместимость с прежним полем category)."""
    return sorted(set(_slug_color_tokens(slug)) | set(_slug_sweet_tokens(slug)))


def _best_sim(tok: str, targets: list) -> float:
    best = 0.0
    for t in targets:
        if tok == t:
            return 1.0
        if len(tok) >= 4 and len(t) >= 4 and (tok in t or t in tok):
            best = max(best, 0.94)
            continue
        best = max(best, SequenceMatcher(None, tok, t).ratio())
    return best


def best_sim_ck(tok: str, targets: list, thresh: float) -> float:
    """Как ``_best_sim``, но при провале сравнивает c/k-нормализованные формы.

    В каталоге сорт пишется склейкой слов («Мускат Белый» -> ``muskatbelii``), а в
    алиасах сортов — «английской» формой («Мускат» -> ``muscat``): токены не совпадают
    ни точь-в-точь, ни подстрокой, и сорт обнулялся (кейс 96.61, отчёт 20).
    """
    best = _best_sim(tok, targets)
    if best >= thresh:
        return best
    a = tok.replace('c', 'k')
    for t in targets:
        b = t.replace('c', 'k')
        if a == b:
            return 1.0
        if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
            best = max(best, 0.94)
    return best


def _token_cover(qtok: list, ttok: list, thresh: float = 0.72) -> float:
    if not qtok or not ttok:
        return 0.0
    vals = []
    for q in qtok:
        s = _best_sim(q, ttok)
        vals.append(s if s >= thresh else 0.0)
    return sum(vals) / len(vals)


def _sym_cover(a: list, b: list) -> float:
    return 0.5 * (_token_cover(a, b) + _token_cover(b, a))


def _grape_overlap(qtok: list, ttok: list) -> float:
    """F1 по МНОЖЕСТВУ сортов (купаж): exact set -> 1.0, 1 из 2 vs 2 -> 0.667.

    Токены уже канонизированы через ``_GRAPE_ALIAS``.
    """
    if not qtok or not ttok:
        return 0.0
    sq, st = set(qtok), set(ttok)
    inter = len(sq & st)
    if not inter:
        return 0.0
    return 2.0 * inter / (len(sq) + len(st))


# ---------------------------------------------------------------- промт извлечения
_EXTRACTION_SCHEMA = (
    "\n\nПрочитай этикетку МАКСИМАЛЬНО внимательно. КЛЮЧЕВЫЕ поля — сорт винограда "
    "(grape, включая купаж), цвет (color), сахар (sugar), название ЛИНЕЙКИ (line) и "
    "класс вина (sparkling): именно по ним различаются вина одной винодельни. "
    "Год (year) — вторичен, его можно не указывать, если не виден. "
    "Не угадывай: если не видно — null.\n"
    "ВАЖНО: «игристое/шампанское» — это НЕ цвет, а КЛАСС вина (см. sparkling). "
    "Цвет (color) — только красное/белое/розовое (даже у игристого: брют розовый -> розовое).\n"
    "Верни СТРОГО JSON (без markdown, без текста вокруг) в формате:\n"
    '{"winery": "<производитель (винодельня/бренд), ОБЯЗАТЕЛЬНО, или null>",\n'
    ' "grape": "<сорта через запятую, напр. \\"Каберне Совиньон, Мерло\\", или null>",\n'
    ' "color": "<красное|белое|розовое или null>",\n'
    ' "sugar": "<сухое|полусухое|полусладкое|сладкое|брют или null>",\n'
    ' "sparkling": "<игристое|шампанское|тихое>",\n'
    ' "line": "<название линейки/серии, напр. Viva / Velvet Season, или null>",\n'
    ' "wine_type": "<цвет и тип одной строкой, напр. \\"красное сухое\\", или null>",\n'
    ' "additional_text": "<прочий текст (объём, крепость) или null>",\n'
    ' "year": "<год урожая, 4 цифры, или null>",\n'
    ' "raw_text": "<вся читаемая транскрипция этикетки одной строкой>",\n'
    ' "field_confidence": {"winery": <0..1>, "grape": <0..1>, "color": <0..1>,'
    ' "sugar": <0..1>, "sparkling": <0..1>, "line": <0..1>}}\n'
)


def build_extraction_prompt(prompt_file) -> str:
    """Достаёт из prompts_wine_match.txt ТОЛЬКО секцию извлечения признаков
    (до инструкции сравнивать с CSV) и приклеивает схему JSON-ответа."""
    path = Path(prompt_file)
    text = path.read_text(encoding='utf-8') if path.is_file() else ''
    if not text:
        text = ("Прочитай этикетку винной бутылки на ТЕКУЩЕМ фото и извлеки из неё данные: "
                "сорт винограда (включая купаж), цвет (красное/белое/розовое), "
                "сахар (сухое/полусухое/полусладкое/сладкое/брют), "
                "класс вина (игристое/шампанское/тихое — это НЕ цвет), "
                "название линейки, производитель, год урожая (необязательно).")
    for marker in ('Сравни извлечённые', 'Сравни извлеченные', '1. Сравни'):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()
            break
    return text + _EXTRACTION_SCHEMA


# ---------------------------------------------------------------- OpenRouter (VLM)
class OcrExtractor:
    """Один вызов внешней vision-модели: фото -> поля этикетки."""

    def __init__(self, cfg):
        self.cfg = cfg.ocr
        self.api_key = os.environ.get(self.cfg.api_key_env, '')
        self.prompt = build_extraction_prompt(cfg.paths.prompt_file)
        if not self.api_key:
            print(f'[ocr] ВНИМАНИЕ: env "{self.cfg.api_key_env}" не задан — OCR пропустится')

    @staticmethod
    def _data_url(img) -> str:
        buf = io.BytesIO()
        img.convert('RGB').save(buf, 'JPEG', quality=92)
        return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

    def prepare(self, img):
        """Кроп этикетки -> изображение для VLM: апскейл мелкого текста + ограничение размера.

        Наклонный/мелкий текст этикетки читается только после апскейла, но
        гигантский JPEG бьёт по токенам и латентности — держим окно
        ``min_side`` .. ``max_side`` (0 — не трогать).
        """
        img = img.convert('RGB')
        w, h = img.size
        if not w or not h:
            return img
        min_side = float(getattr(self.cfg, 'min_side', 0) or 0)
        max_side = float(getattr(self.cfg, 'max_side', 0) or 0)
        side = min(w, h)
        if min_side and side < min_side:
            scale = min_side / float(side)
        elif max_side and max(w, h) > max_side:
            scale = max_side / float(max(w, h))
        else:
            return img
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return img.resize(size, Image.LANCZOS)

    def extract(self, img) -> dict:
        if not self.api_key:
            return {'error': 'no_api_key', 'fields': {}}
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': self._data_url(self.prepare(img))}},
                {'type': 'text', 'text': self.prompt},
            ],
        }]
        payload = {
            'model': self.cfg.model,
            'messages': messages,
            'temperature': float(self.cfg.temperature),
            'max_tokens': int(self.cfg.max_tokens),
            'response_format': {'type': 'json_object'},
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://vino-svoe.ru',
            'X-Title': 'svoe-vino-scanner',
        }
        last = None
        for attempt in range(1, int(self.cfg.retries) + 1):
            try:
                r = requests.post(self.cfg.api_url, headers=headers,
                                  json=payload, timeout=int(self.cfg.timeout))
                if r.status_code in (429, 500, 502, 503, 529):
                    time.sleep(2 ** attempt)
                    last = RuntimeError(f'HTTP {r.status_code}: {r.text[:300]}')
                    continue
                if r.status_code in (400, 401, 402, 403, 404):
                    # 402 — кончились кредиты OpenRouter, 401/403 — ключ/доступ: ретраить
                    # бессмысленно (раньше 402 попадал в retry и жёг время на каждом фото).
                    last = RuntimeError(f'OpenRouter HTTP {r.status_code}: {r.text[:300]}')
                    break
                r.raise_for_status()
                data = r.json()
                content = data['choices'][0]['message']['content']
                usage = data.get('usage') or {}
                return {'fields': self._parse(content), 'raw': content,
                        'usage': {'prompt_tokens': usage.get('prompt_tokens', 0),
                                  'completion_tokens': usage.get('completion_tokens', 0),
                                  'total_tokens': usage.get('total_tokens', 0)}}
            except requests.RequestException as e:  # noqa: BLE001
                last = e
                time.sleep(2 ** attempt)
        return {'error': f'openrouter_failed: {last}', 'fields': {}}

    @staticmethod
    def _parse(content: str) -> dict:
        t = (content or '').strip()
        m = re.search(r'```(?:json)?\s*(.*?)```', t, re.DOTALL | re.IGNORECASE)
        if m:
            t = m.group(1).strip()
        try:
            data = json.loads(t)
        except json.JSONDecodeError:
            s = t.find('{'); e = t.rfind('}')
            if s != -1 and e > s:
                data = json.loads(t[s:e + 1])
            else:
                data = {}
        keys = ('year', 'winery', 'grape', 'color', 'sugar', 'line', 'sparkling',
                'wine_type', 'additional_text', 'raw_text', 'field_confidence')
        out = {}
        for k in keys:
            v = data.get(k)
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            s = '' if v is None else str(v).strip()
            if s.lower() in ('null', 'none', 'nan', 'нет', 'не указано', 'не найдено', '-'):
                s = ''
            out[k] = s
        return out


# ---------------------------------------------------------------- мэтчинг по референсному CSV
def _cfg_weight(weights, key: str, *aliases, default: float = 0.0) -> float:
    """Вес поля из конфига с алиасами (совместимость старых конфигов).

    Напр. ``color``/``wine_type`` могут браться из старого общего ``category``.
    """
    for k in (key,) + aliases:
        try:
            if k in weights:
                return float(weights[k])
        except TypeError:                       # plain-dict без __contains__-семантики
            if isinstance(weights, dict) and k in weights:
                return float(weights[k])
    return default


class CsvMatcher:
    """Сопоставление полей OCR с записями found_in_catalog_corrected.csv (без модели).

    Уверенность записи — взвешенная доля совпавших полей (``csv_match.weights``);
    поле, которого нет у записи, даёт ``missing_credit``.
    """

    FIELDS = ('Название вина', 'Категория', 'Цвет', 'Регион', 'Сорт винограда',
              'Описание', 'Винодельня', 'Slug', 'photo_file')

    def __init__(self, cfg):
        self.cfg = cfg.csv_match
        self._root_cfg = cfg            # корневой конфиг: путь к файлу обогащения (paths.*)
        with open(cfg.paths.reference_csv, encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f, delimiter=';'))
        self.entries = []
        self._by_base = {}
        self._by_slug = {}
        seen = set()
        for r in rows:
            slug = (r.get('Slug') or '').strip()
            if not slug or slug in seen:
                continue
            seen.add(slug)
            base, year_digit = parse_slug(slug)
            # год может быть и в названии, напр. "Алиготе Баррель, 2024"
            title = (r.get('Название вина') or '').strip()
            m = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', title)
            title_year = m.group(0) if m else ''
            e = {
                'slug': slug,
                'title': title,
                'winery': (r.get('Винодельня') or '').strip(),
                'grape': (r.get('Сорт винограда') or '').strip(),
                'category': (r.get('Категория') or '').strip(),
                'color': (r.get('Цвет') or '').strip(),
                'photo_file': (r.get('photo_file') or '').strip(),
                'base_key': '-'.join(base),
                'year': year_code(year_digit) or year_code(title_year) or year_from_text(title),
                'winery_tok': _norm_tokens(r.get('Винодельня')),
                'grape_tok': _grape_tokens(r.get('Сорт винограда')),
                'title_tok': _norm_tokens(title),
                'slug_tok': _norm_tokens(slug.replace('-', ' ')),
                'color_tok': sorted(set(_wine_color_tokens(r.get('Цвет'), r.get('Категория'))
                                       + _shade_color_tokens(r.get('Цвет'))
                                       + _slug_color_tokens(slug))),
                'sweet_tok': sorted(set(_wine_sweet_tokens(r.get('Категория'), title)
                                        + _slug_sweet_tokens(slug))),
                'cat_tok': _slug_category_tokens(slug),
                # класс вина: игристое/шампанское ⟷ тихое (маркеры в названии/слаге)
                'spark_tok': (sorted(set(_wine_sparkling_tokens(title, slug.replace('-', ' '))
                                         + _slug_spark_tokens(slug))) or ['still']),
            }
            self.entries.append(e)
            self._by_slug[slug] = e
            self._by_base.setdefault(e['base_key'], []).append(e['slug'])
        # словарь сортов каталога: по нему фильтруем сорт, вытащенный из доп. текста
        self.grape_vocab = {t for e in self.entries for t in e['grape_tok']}
        self.grape_from_text = bool(getattr(self.cfg, 'grape_from_text', False))
        self.grape_contain = bool(getattr(self.cfg, 'grape_contain', False))
        self.fuzzy_ck_norm = bool(getattr(self.cfg, 'fuzzy_ck_norm', False))
        self._apply_enrichment(str(getattr(self.cfg, 'enrichment_file', '') or ''))

    def _enr_path(self, path: str):
        """Какой файл обогащения читать: auto / выключено / явный путь.

        - ``auto`` (значение конфига) — ``paths.catalog_ocr_fields``: в Docker это
          ``/app/ref/...``, на хосте ``apply_host_paths`` переписывает его в
          ``<repo>/data/...`` (как ``reference_csv``);
        - ``""`` / ``none`` — обогащение выключено;
        - иначе — путь как есть, либо относительно КОРНЯ репозитория (удобно для
          харнессов: ``--enrich data/catalog_ocr_fields.csv``).
        """
        if path in ('', 'none'):
            return None
        if path == 'auto':
            paths = getattr(self._root_cfg, 'paths', None)
            cand = str(getattr(paths, 'catalog_ocr_fields', '') or '') if paths else ''
            if not cand:
                print('[csv_match] enrichment_file=auto, но paths.catalog_ocr_fields пуст',
                      flush=True)
                return None
            p = Path(cand)
        else:
            p = Path(path)
        if not p.is_absolute():
            p = REPO / p
        if not p.is_file():
            print(f'[csv_match] enrichment_file не найден: {p}', flush=True)
            return None
        return p

    def _enr_consistent(self, e: dict, r: dict) -> bool:
        """Согласован ли прочитанный бренд с записью каталога (защита от мусора).

        Часть каталожных фото VLM читает неверно (`pobeda` -> «Mancopia/Riesling»,
        `rubin-golodrigi` -> «CR»): если бренд не подтверждается ни полем «Винодельня»,
        ни названием, ни слагом записи — данные такого фото не подмешиваем.
        """
        w = _norm_tokens(r.get('winery'))
        if not w:
            return True
        ref = sorted(set(e['winery_tok']) | set(e['title_tok']) | set(e['slug_tok']))
        return _token_cover(w, ref, 0.72) >= 0.5

    def _apply_enrichment(self, path: str) -> int:
        """Дополнить записи каталога полями OCR-разметки каталожных фото.

        Исходную разметку организаторов НЕ меняем: поля только добавляются к токенам
        записи (сорт/линейка/сахар) и заполняется пустой год. Файл —
        `data/catalog_ocr_fields.csv` (приезжает в архиве `data`).
        """
        if not path:
            return 0
        p = self._enr_path(path)
        if p is None:
            return 0
        n, skipped = 0, 0
        with open(p, encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f, delimiter=';'):
                e = self._by_slug.get((r.get('slug') or '').strip())
                if e is None:
                    continue
                if not self._enr_consistent(e, r):
                    skipped += 1
                    continue
                grape = _grape_tokens(r.get('grape'))
                if grape:
                    e['grape_tok'] = sorted(set(e['grape_tok']) | set(grape))
                sweet = _wine_sweet_tokens(r.get('sugar'), r.get('wine_type'))
                if sweet:
                    e['sweet_tok'] = sorted(set(e['sweet_tok']) | set(sweet))
                line = _norm_tokens(r.get('line'))
                if line:
                    e['title_tok'] = sorted(set(e['title_tok']) | set(line))
                if not e['year']:
                    e['year'] = year_code(r.get('year'))
                n += 1
        self.grape_vocab = {t for e in self.entries for t in e['grape_tok']}
        print(f'[csv_match] обогащение из {p.name}: {n} записей '
              f'(пропущено {skipped} — бренд не подтвердился)', flush=True)
        return n

    def _year_conflict(self, e: dict) -> bool:
        """True, если в базе есть то же вино в разных винтажах (одна основа слага)."""
        return len(self._by_base.get(e['base_key'], [])) > 1

    # Поля-дискриминаторы в порядке значимости для near-dup «сестёр».
    _FIELDS = ('winery', 'grape', 'color', 'wine_type', 'line', 'sparkling',
               'year', 'additional_text')

    def _query(self, fields: dict) -> dict:
        """Нормализованные значения запроса по полям каталога (пустые — None).

        При ``grape_from_text`` сорт добирается из ``additional_text``/``raw_text``
        (VLM часто пишет туда сорт, который не опознал как поле ``grape``: «Рубин»,
        «Цитрон 2024»). Добираем ТОЛЬКО токены, которые есть в словаре сортов
        каталога, — иначе в сорт попадал бы мусор («Тираж», «Год урожая»).
        """
        q = {
            'winery': _norm_tokens(fields.get('winery')) or None,
            'grape': _grape_tokens(fields.get('grape')) or None,
            'color': _wine_color_tokens(fields.get('color'), fields.get('wine_type')) or None,
            'wine_type': _wine_sweet_tokens(fields.get('sugar'), fields.get('wine_type')) or None,
            # класс вина: игристое/шампанское ⟷ тихое (НЕ цвет!)
            'sparkling': _wine_sparkling_tokens(fields.get('sparkling')) or None,
            # линейка/название: явное поле line, иначе доп. текст (напр. "Velvet Season")
            'line': (_norm_tokens(fields.get('line'))
                     or _norm_tokens(fields.get('additional_text'))) or None,
            # год: явное поле LLM, иначе 4-значный год в доп. тексте / типе
            'year': (year_code(fields.get('year'))
                     or year_from_text(fields.get('additional_text'),
                                       fields.get('wine_type'))) or None,
            'additional_text': _norm_tokens(fields.get('additional_text')) or None,
        }
        if self.grape_from_text and not q.get('grape'):
            extra = []
            for key in ('additional_text', 'raw_text'):
                extra += _grape_tokens(fields.get(key))
            extra = sorted({t for t in extra if t in self.grape_vocab})
            if extra:
                q['grape'] = extra
        return q

    def _field_sim(self, key: str, qval, e: dict, thresh: float,
                   only_if_conflict: bool):
        """Сходство запроса и записи по одному полю.

        Возвращает float либо ``None``, если у записи НЕТ данных по этому полю
        (тогда поле считается нейтральным: ``missing_credit``, а не 0 и не 1).
        """
        if key == 'winery':
            # в каталоге иногда винодельня не заполнена, но бренд есть в слаге
            return max(_sym_cover(qval, e['winery_tok']),
                       0.85 * _sym_cover(qval, e['slug_tok']))
        if key == 'grape':
            if not e['grape_tok']:
                return None
            # купаж: F1 по множеству сортов (exact set -> 1.0, частичное -> ниже),
            # плюс (grape_contain) «покрытие запроса»: VLM читает 1-2 сорта из купажа,
            # и F1 штрафовал правильную запись за «лишние» сорта сильнее, чем односортную
            # запись-«близнеца» (отчёты 19/20: 84.73, 87.09, 96.61).
            sim = max(_grape_overlap(qval, e['grape_tok']),
                      0.8 * _sym_cover(qval, e['slug_tok']))
            if self.grape_contain:
                cov = (max(best_sim_ck(q, e['grape_tok'], thresh) for q in qval)
                       if self.fuzzy_ck_norm else _token_cover(qval, e['grape_tok'], thresh))
                sim = max(sim, cov)
            return sim
        if key == 'color':
            return _token_cover(qval, e['color_tok'], thresh) if e['color_tok'] else None
        if key == 'wine_type':
            return _token_cover(qval, e['sweet_tok'], thresh) if e['sweet_tok'] else None
        if key == 'sparkling':
            # класс: игристое/шампанское ⟷ тихое (spark_tok всегда непустой)
            return _token_cover(qval, e.get('spark_tok') or [], thresh)
        if key == 'year':
            if not e['year']:
                return None
            # в режиме конфликта год учитываем только у «сестёр» по основе слага
            if only_if_conflict and not self._year_conflict(e):
                return None
            return year_sim(e['year'], qval)
        if key == 'additional_text':
            if not (e['title_tok'] or e['slug_tok']):
                return None
            return max(_token_cover(qval, e['title_tok'], thresh),
                       0.7 * _token_cover(qval, e['slug_tok'], thresh))
        if key == 'line':
            # линейка/название: токены line ⟷ title+slug записи
            if not (e['title_tok'] or e['slug_tok']):
                return None
            return max(_token_cover(qval, e['title_tok'], thresh),
                       0.7 * _token_cover(qval, e['slug_tok'], thresh))
        return None

    def match(self, fields: dict, only=None) -> list:
        """fields -> отсортированный список [{slug, confidence, breakdown}].

        ``only`` — необязательный список slug'ов: считать только по ним
        (``csv_match.shortlist_only``); по умолчанию — весь каталог.

        Ключевое отличие от «наивного» fuzzy-матчинга: нормировка идёт на
        ФИКСИРОВАННЫЙ набор полей, заданный ЗАПРОСОМ (одинаковый для всех
        кандидатов), а не на «поля, которые нашлись у записи». Иначе запись без
        данных (напр. ``balaklava-muskat`` без цвета/типа) получала искусственно
        высокую уверенность только потому, что штрафные поля у неё отсутствуют.
        Отсутствующее у записи поле даёт ``missing_credit`` (по умолчанию 0.5) —
        нейтрально: не награждает и не обнуляет.
        """
        w = self.cfg.weights
        thresh = float(self.cfg.token_thresh)
        credit = float(getattr(self.cfg, 'missing_credit', 0.5))
        only_if_conflict = bool(getattr(self.cfg, 'year_only_if_vintage_conflict', False))
        q = self._query(fields)
        active = [k for k in self._FIELDS if q.get(k)]
        if not active:
            return []
        weights = {k: _cfg_weight(w, k, 'category' if k in ('color', 'wine_type') else k)
                   for k in active}
        tot_w = sum(weights.values())
        if tot_w <= 0:
            return []

        if only is None:
            entries = self.entries
        else:
            entries = [self._by_slug[s] for s in dict.fromkeys(only) if s in self._by_slug]

        results = []
        for e in entries:
            part, detail = {}, {}
            for k in active:
                sim = self._field_sim(k, q[k], e, thresh, only_if_conflict)
                part[k] = credit if sim is None else sim
                detail[k] = round(part[k], 3)
            wsum = sum(weights[k] * part[k] for k in active)
            results.append({'slug': e['slug'], 'confidence': round(wsum / tot_w, 4),
                            'photo_file': e['photo_file'],
                            'year': e['year'], 'breakdown': detail})
        results.sort(key=lambda x: -x['confidence'])
        return results


# ---------------------------------------------------------------- визуальная проверка
class VisualVerifier:
    """Косинусная близость (SigLIP2) фото каталога (start_photos) и текущего фото.

    ``embed_fn`` — функция [PIL, ...] -> np.ndarray[N, dim] (L2-нормированный).
    ``crop_fn`` — функция PIL -> PIL, применяемая к референсному фото (запрос уже кропнут).
    """

    def __init__(self, cfg, embed_fn, crop_fn=None):
        self.cfg = cfg.visual
        self.start_photos = Path(cfg.paths.start_photos)
        self.embed_fn = embed_fn
        self.crop_fn = crop_fn

    def _ref_path(self, photo_file: str, slug: str) -> Path:
        name = Path(photo_file).name if photo_file else ''
        if name and (self.start_photos / name).exists():
            return self.start_photos / name
        return self.start_photos / f'{slug}.webp'

    def verify(self, query_crop, photo_file: str, slug: str) -> float:
        ref_path = self._ref_path(photo_file, slug)
        if not ref_path.exists():
            return 0.0
        ref = Image.open(ref_path).convert('RGB')
        if self.crop_fn is not None:
            try:
                ref = self.crop_fn(ref)
            except Exception:  # noqa: BLE001 — не ронять из-за одного референса
                pass
        v = self.embed_fn([query_crop, ref])
        a, b = v[0], v[1]
        return float(np.dot(a, b))


# ---------------------------------------------------------------- оркестрация rerank
def _candidates_payload(candidates: list, n: int = 3) -> list:
    """Компактный top-N кандидатов для отчёта/ответа API."""
    return [{'slug': c['slug'], 'confidence': c['confidence'],
             'year': c.get('year'), 'breakdown': c.get('breakdown')}
            for c in candidates[:n]]


def decide(cfg, candidates: list, pre_ocr_slug, min_conf: float) -> dict:
    """Гейт «оставить retrieval или заменить на CSV-ответ» (чистая функция).

    Сравнение идёт в ОДНОЙ шкале: CSV-уверенность лучшего кандидата против
    CSV-уверенности действующего ответа retrieval (обе — доля покрытия полей от
    фиксированного набора, заданного запросом; см. ``CsvMatcher.match``).

    Прежний гейт (``csv_conf`` против косинуса SigLIP 0.7–0.85) сравнивал разные
    шкалы и блокировал OCR почти всегда; гейт «по разрыву top1/top2 внутри CSV»
    мог выбрать «сестру» вместо ответа retrieval. Здесь нужен явный ЗАПАС над
    incumbent'ом.

    Возвращает dict с ``final_slug``/``reason``/``switched`` и диагностикой.
    """
    if not candidates:
        return {'final_slug': pre_ocr_slug, 'reason': 'no_csv_match', 'switched': False,
                'top_candidates': []}
    top = candidates[0]
    csv_conf = float(top['confidence'])
    retr_conf = next((float(c['confidence']) for c in candidates
                      if c['slug'] == pre_ocr_slug), 0.0)
    margin = round(csv_conf - retr_conf, 4)
    second_conf = float(candidates[1]['confidence']) if len(candidates) > 1 else 0.0
    extra = {'csv_confidence': csv_conf, 'csv_margin': margin,
             'retrieval_csv_confidence': round(retr_conf, 4),
             'second_csv_confidence': second_conf,
             'top_candidates': _candidates_payload(candidates)}

    if top['slug'] == pre_ocr_slug:
        return {'final_slug': pre_ocr_slug, 'reason': 'csv_agrees', 'switched': False, **extra}

    need = float(getattr(cfg.csv_match, 'decision_margin', 0.05))
    if csv_conf < float(min_conf):
        return {'final_slug': pre_ocr_slug, 'reason': 'low_csv_confidence',
                'switched': False, **extra}
    if margin < need:
        return {'final_slug': pre_ocr_slug, 'reason': 'csv_ambiguous',
                'switched': False, **extra}
    return {'final_slug': top['slug'], 'reason': 'csv_decisive', 'switched': True, **extra}


def _shortlist(matcher, pre_ocr_slug, pre_ocr_candidates) -> list:
    """Шорт-лист для ``csv_match.shortlist_only``: ответ retrieval + его top-k +
    «сёстры» по основе слага."""
    out = []
    if pre_ocr_slug:
        out.append(pre_ocr_slug)
    for s in (pre_ocr_candidates or []):
        if s:
            out.append(s)
    if pre_ocr_slug:
        base, _ = parse_slug(pre_ocr_slug)
        out.extend(matcher._by_base.get('-'.join(base), []))
    return list(dict.fromkeys(out))


def _fields_poor(fields: dict, vocab: set) -> bool:
    """Похоже, что VLM прочитал кроп плохо -> стоит перечитать по кропу бутылки.

    Триггеры (отчёты 19/20): детектор этикетки обрезал надписи, rectify «зуммит» и
    теряет строку сорта, либо в кропе вообще нет текста и модель выдумывает сорт
    (`94.02`, `95.63`, `87.54`).
    """
    winery = (fields.get('winery') or '').strip()
    grape = (fields.get('grape') or '').strip()
    line = (fields.get('line') or '').strip()
    if not winery and not grape:
        return True
    if not grape:
        return True                      # сорт не прочитан -> кроп, вероятно, обрезан
    toks = _grape_tokens(grape)
    if toks and vocab and not any(_best_sim(t, list(vocab)) >= 0.8 for t in toks):
        return True                      # сорт, которого нет в каталоге (галлюцинация)
    return False


def _merge_fields(a: dict, b: dict, vocab: set) -> dict:
    """Слияние полей двух проходов: непустые из первого, добор из второго.

    Сорт: если в первом проходе его нет или он не из словаря каталога, берём вариант
    второго прохода (он читался по кропу бутылки).
    """
    out = dict(a)
    for k, v in (b or {}).items():
        if not (v or '').strip():
            continue
        if not (out.get(k) or '').strip():
            out[k] = v
    ga = _grape_tokens(out.get('grape'))
    gb = _grape_tokens((b or {}).get('grape'))
    if gb and (not ga or not any(_best_sim(t, list(vocab)) >= 0.8 for t in ga)):
        out['grape'] = (b or {}).get('grape')
    return out


def rerank(cfg, query_image, query_crop, pre_ocr_slug, pre_ocr_score,
           extractor, matcher, verifier, ocr_image=None, pre_ocr_candidates=None,
           bottle_crop=None) -> dict:
    """Решение: оставить retrieval-ответ или заменить его на OCR/CSV-ответ.

    Порядок: извлечь поля (1 вызов модели по ВЫПРЯМЛЕННОМУ кропу этикетки) ->
    самописный мэтчинг по CSV в пределах короткого списка retrieval -> решение
    «CSV-кандидат против CSV-ответа retrieval».

    Гейт сравнивает ДВЕ CSV-уверенности (обе — доля покрытия токенов, 0–1) на
    одной шкале: OCR-ответ побеждает, только если его CSV-скор выше скора
    текущего retrieval-ответа на ``decision_margin``. Прежний гейт
    (``csv_conf`` против косинуса SigLIP 0.7–0.85) сравнивал разные шкалы и
    блокировал OCR почти всегда; гейт по разрыву внутри CSV top1/top2 всё ещё
    мог выбрать «сестру» вместо ответа retrieval.

    ``ocr_image`` — изображение для VLM (выпрямленный кроп этикетки); если не
    задано, берётся ``query_crop`` при ``ocr.use_label_crop`` иначе ``query_image``.
    ``query_crop`` по-прежнему идёт в визуальную проверку (как препроцесс индекса).
    ``pre_ocr_candidates`` — slug'и short-list'а retrieval (best-first).
    """
    base = {'pre_ocr_slug': pre_ocr_slug, 'pre_ocr_score': round(float(pre_ocr_score), 4)}
    if not cfg.ocr.enabled:
        return {**base, 'final_slug': pre_ocr_slug, 'stage': 'retrieval',
                'reason': 'ocr_disabled'}

    if ocr_image is None and bool(getattr(cfg.ocr, 'use_label_crop', True)):
        ocr_image = query_crop
    if ocr_image is None:
        ocr_image = query_image
    ocr_input = 'label_crop' if ocr_image is not query_image else 'full_image'

    try:
        ocr = extractor.extract(ocr_image)
    except Exception as e:  # noqa: BLE001 — OCR не должен ронять пайплайн
        ocr = {'error': f'ocr_exception: {e}', 'fields': {}}
    fields = ocr.get('fields', {})
    if not fields or not any((v or '').strip() for v in fields.values()):
        return {**base, 'final_slug': pre_ocr_slug, 'stage': 'retrieval',
                'reason': 'ocr_empty', 'ocr_error': ocr.get('error'),
                'ocr_input': ocr_input}

    # Страховка на случай ошибки детектора этикетки / rectify: если поля бедные или
    # сорт не из каталога — читаем ещё раз по КРОПУ БУТЫЛКИ и сливаем поля.
    retry_info = None
    if (bottle_crop is not None and bool(getattr(cfg.ocr, 'retry_on_poor_fields', False))
            and _fields_poor(fields, matcher.grape_vocab)):
        try:
            second = extractor.extract(bottle_crop)
            fields2 = second.get('fields', {})
            if fields2:
                fields = _merge_fields(fields, fields2, matcher.grape_vocab)
                ocr_input = f'{ocr_input}+bottle_crop'
                retry_info = {'retry_reason': 'poor_fields',
                              'retry_fields': fields2}
        except Exception as e:  # noqa: BLE001 — страховка не должна ронять пайплайн
            print(f'[ocr] retry on bottle crop failed: {e}', flush=True)

    shortlist = _shortlist(matcher, pre_ocr_slug, pre_ocr_candidates)
    only = shortlist if (shortlist and bool(getattr(cfg.csv_match, 'shortlist_only', True))) \
        else None
    candidates = matcher.match(fields, only=only)
    min_conf = max(float(cfg.ocr.min_confidence), float(cfg.csv_match.min_score))
    if not candidates:
        return {**base, 'final_slug': pre_ocr_slug, 'stage': 'retrieval',
                'reason': 'no_csv_match', 'ocr_fields': fields, 'ocr_input': ocr_input,
                'shortlist_size': len(shortlist), 'top_candidates': [],
                **(retry_info or {})}

    decision = decide(cfg, candidates, pre_ocr_slug, min_conf)
    csv_extra = {'shortlist_size': len(shortlist), 'ocr_fields': fields,
                 'ocr_input': ocr_input, **(retry_info or {}),
                 **{k: v for k, v in decision.items()
                    if k not in ('final_slug', 'switched')}}

    if not decision['switched']:
        return {**base, 'final_slug': pre_ocr_slug, 'stage': 'retrieval',
                'reason': decision['reason'], **csv_extra}

    # ---- OCR предлагает ДРУГОГО кандидата и выиграл гейт: мягкая визуальная проверка
    top = candidates[0]
    vis = 1.0
    if cfg.visual.enabled:
        try:
            vis = verifier.verify(query_crop, top['photo_file'], top['slug'])
        except Exception as e:  # noqa: BLE001 — визуал не должен ронять пайплайн
            print(f'[ocr] visual verify failed: {e}', flush=True)
            vis = 1.0
    vis_ok = (not cfg.visual.enabled) or (vis >= float(cfg.visual.min_similarity))
    csv_extra['visual_similarity'] = round(vis, 4)
    if not vis_ok and bool(getattr(cfg.visual, 'block_on_mismatch', False)):
        return {**base, 'final_slug': pre_ocr_slug, 'stage': 'retrieval',
                'reason': 'visual_mismatch', **csv_extra}

    return {**base, 'final_slug': top['slug'], 'stage': 'ocr_rerank',
            'reason': 'visual_ok' if vis_ok else 'csv_decisive_visual_weak',
            **csv_extra}
