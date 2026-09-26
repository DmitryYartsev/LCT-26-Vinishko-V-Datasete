# -*- coding: utf-8 -*-
"""OCR-rerank для пайплайна предсказаний вина.

Добавляет поверх image retrieval (SigLIP2 + pgvector) второй проход:

  1. ``OcrExtractor`` — ОДИН вызов внешней vision-модели (gpt-4o-mini через
     OpenRouter) на фото: извлекает с этикетки поля
     ``year/winery/grape/wine_type/additional_text``. Промт извлекается из
     ``paths.prompt_file`` (берётся только секция извлечения признаков).
  2. ``CsvMatcher`` — самописный (без модели) мэтчинг извлечённых параметров
     по референсному CSV (``found_in_catalog_corrected.csv``). Важность полей
     (по убыванию): винодельня > категория (цвет+сладость) > сорт винограда >
     год (только при конфликте винтажей) > дополнительный текст.
  3. ``VisualVerifier`` — если уверенность CSV-мэтча выше уверенности ответа
     модели до OCR, проверяем косинусную близость фото каталога (start_photos)
     и текущего фото через тот же энкодер.

Итог: ``rerank()`` возвращает финальный slug + какая ветка победила + скоринг.

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


# ---------------------------------------------------------------- синонимы сортов / типа
_GRAPE_ALIAS = {
    'kaberne': 'cabernet', 'cabernet': 'cabernet', 'sauvignon': 'sauvignon',
    'sovinon': 'sauvignon', 'shardone': 'chardonnay', 'chardonnay': 'chardonnay',
    'pino': 'pinot', 'pinot': 'pinot', 'nuar': 'noir', 'noir': 'noir',
    'merlo': 'merlot', 'merlot': 'merlot', 'risling': 'riesling',
    'riesling': 'riesling', 'saperavi': 'saperavi', 'rkaciteli': 'rkatsiteli',
    'rkatsiteli': 'rkatsiteli', 'muskat': 'muscat', 'muscat': 'muscat',
    'sira': 'syrah', 'syrah': 'syrah', 'aligote': 'aligote',
    'kokur': 'kokur', 'traminer': 'traminer', 'gewurz': 'gewurztraminer',
    'gevyurc': 'gewurztraminer', 'malbek': 'malbec', 'malbec': 'malbec',
}

_COLOR_WORDS = {
    'белое': 'white', 'white': 'white', 'красное': 'red', 'red': 'red',
    'розовое': 'rose', 'rose': 'rose', 'розе': 'rose', 'pink': 'rose',
    'игристое': 'sparkling', 'sparkling': 'sparkling',
}
_SWEET_WORDS = {
    'сухое': 'dry', 'dry': 'dry', 'брют': 'brut', 'brut': 'brut',
    'полусухое': 'semidry', 'semi-dry': 'semidry', 'semidry': 'semidry',
    'полусладкое': 'semisweet', 'semi-sweet': 'semisweet', 'semisweet': 'semisweet',
    'сладкое': 'sweet', 'sweet': 'sweet',
}
# транслит-формы цвета/сладости, как они встречаются в слагах каталога
_SLUG_COLOR = {'beloe': 'white', 'krasnoe': 'red', 'rozovoe': 'rose',
               'igristoe': 'sparkling'}
_SLUG_SWEET = {'suhoe': 'dry', 'polusuhoe': 'semidry', 'polusladkoe': 'semisweet',
               'sladkoe': 'sweet', 'bryut': 'brut', 'ekstrabryut': 'extrabrut',
               'extrabrut': 'extrabrut'}


def _norm_tokens(text: str) -> list:
    """Текст -> канонизированные токены (латиница, без стоп-мусора)."""
    if not text:
        return []
    w = canon_word(text)
    return [t for t in re.split(r'[^a-z0-9]+', w) if len(t) >= 2]


def _grape_tokens(text: str) -> list:
    return [_GRAPE_ALIAS.get(t, t) for t in _norm_tokens(text)]


def _wine_type_tokens(wine_type: str) -> list:
    """wine_type (LLM) -> канонические токены цвета + сладости."""
    s = (wine_type or '').lower()
    out = set()
    for k, v in _COLOR_WORDS.items():
        if re.search(r'\b' + re.escape(k) + r'\b', s):
            out.add(v)
    for k, v in _SWEET_WORDS.items():
        if re.search(r'\b' + re.escape(k) + r'\b', s):
            out.add(v)
    return sorted(out)


def _slug_category_tokens(slug: str) -> list:
    """Из слага достаём канонические токены цвета/сладости."""
    out = set()
    for t in slug.lower().split('-'):
        if t in _SLUG_COLOR:
            out.add(_SLUG_COLOR[t])
        if t in _SLUG_SWEET:
            out.add(_SLUG_SWEET[t])
    return sorted(out)


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


# ---------------------------------------------------------------- промт извлечения
_EXTRACTION_SCHEMA = (
    "\n\nВерни СТРОГО JSON (без markdown, без текста вокруг) в формате:\n"
    '{"year": "<год или null>", "winery": "<производитель или null>",\n'
    ' "grape": "<сорта через запятую или null>", "wine_type": "<тип или null>",\n'
    ' "additional_text": "<доп. текст или null>"}\n'
)


def build_extraction_prompt(prompt_file) -> str:
    """Достаёт из prompts_wine_match.txt ТОЛЬКО секцию извлечения признаков
    (до инструкции сравнивать с CSV) и приклеивает схему JSON-ответа."""
    path = Path(prompt_file)
    text = path.read_text(encoding='utf-8') if path.is_file() else ''
    if not text:
        text = ("Прочитай этикетку винной бутылки на ТЕКУЩЕМ фото и извлеки из неё данные: "
                "год, производитель и марка вина, сорт винограда, тип вина "
                "(сухое/брют/полусладкое/полусухое), дополнительный текст.")
    for marker in ('Сравни извлечённые', 'Сравни извлеченные', '1. Сравни'):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()
            break
    return text + _EXTRACTION_SCHEMA


# ---------------------------------------------------------------- OpenRouter (gpt-4o-mini)
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

    def extract(self, img) -> dict:
        if not self.api_key:
            return {'error': 'no_api_key', 'fields': {}}
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': self._data_url(img)}},
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
                if r.status_code in (400, 401, 403, 404):
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
        keys = ('year', 'winery', 'grape', 'wine_type', 'additional_text')
        return {k: ('' if data.get(k) is None else str(data[k]).strip()) for k in keys}


# ---------------------------------------------------------------- мэтчинг по референсному CSV
class CsvMatcher:
    """Самописный (без модели) поиск по параметрам вина в found_in_catalog_corrected.csv.

    Важность полей (веса, по убыванию): винодельня > категория > сорт > год > доп.текст.
    Год учитывается только когда у одной основы слага есть несколько винтажей.
    """

    FIELDS = ('Название вина', 'Категория', 'Цвет', 'Регион', 'Сорт винограда',
              'Описание', 'Винодельня', 'Slug', 'photo_file')

    def __init__(self, cfg):
        self.cfg = cfg.csv_match
        with open(cfg.paths.reference_csv, encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f, delimiter=';'))
        self.entries = []
        self._by_base = {}
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
                'year': year_digit or title_year,
                'winery_tok': _norm_tokens(r.get('Винодельня')),
                'grape_tok': _grape_tokens(r.get('Сорт винограда')),
                'title_tok': _norm_tokens(title),
                'slug_tok': _norm_tokens(slug.replace('-', ' ')),
                'cat_tok': _wine_type_tokens((r.get('Категория') or '') + ' ' +
                                             (r.get('Цвет') or ''))
                             + _slug_category_tokens(slug),
            }
            self.entries.append(e)
            self._by_base.setdefault(e['base_key'], []).append(e['slug'])

    def _year_conflict(self, e: dict) -> bool:
        """True, если в базе есть то же вино в разных винтажах (одна основа слага)."""
        return len(self._by_base.get(e['base_key'], [])) > 1

    def match(self, fields: dict) -> list:
        """fields -> отсортированный список [{slug, confidence, breakdown}]."""
        w = self.cfg.weights
        thresh = float(self.cfg.token_thresh)
        winery_q = _norm_tokens(fields.get('winery'))
        grape_q = _grape_tokens(fields.get('grape'))
        add_q = _norm_tokens(fields.get('additional_text'))
        wtype_q = _wine_type_tokens(fields.get('wine_type'))
        year_q = re.sub(r'\D', '', str(fields.get('year') or ''))
        year_q = year_q if len(year_q) == 4 else ''

        results = []
        for e in self.entries:
            part, detail = {}, {}
            if winery_q:
                part['winery'] = max(_sym_cover(winery_q, e['winery_tok']),
                                     0.85 * _sym_cover(winery_q, e['slug_tok']))
                detail['winery'] = round(part['winery'], 3)
            if wtype_q:
                pool = e['cat_tok'] + e['slug_tok']
                cover = _token_cover(wtype_q, pool, thresh)
                part['category'] = cover
                detail['category'] = round(cover, 3)
            if grape_q:
                part['grape'] = max(_sym_cover(grape_q, e['grape_tok']),
                                    0.8 * _sym_cover(grape_q, e['slug_tok']))
                detail['grape'] = round(part['grape'], 3)
            if add_q:
                part['additional_text'] = max(
                    _token_cover(add_q, e['title_tok'], thresh),
                    0.7 * _token_cover(add_q, e['slug_tok'], thresh))
                detail['additional_text'] = round(part['additional_text'], 3)
            # год — только при конфликте винтажей
            if year_q and (not self.cfg.year_only_if_vintage_conflict
                           or self._year_conflict(e)):
                part['year'] = 1.0 if year_q == e['year'] else 0.0
                detail['year'] = part['year']
            if not part:
                continue
            tot_w = sum(float(w[k]) for k in part)
            conf = sum(float(w[k]) * v for k, v in part.items()) / tot_w
            results.append({'slug': e['slug'], 'confidence': round(conf, 4),
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
        from PIL import Image
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
def rerank(cfg, query_image, query_crop, pre_ocr_slug, pre_ocr_score,
           extractor, matcher, verifier) -> dict:
    """Решение: оставить retrieval-ответ или заменить его на OCR/CSV-ответ.

    Порядок: извлечь поля (1 вызов модели) -> самописный мэтчинг по CSV -> если
    уверенность CSV > уверенности модели до OCR -> визуальная проверка фото каталога.
    """
    base = {'pre_ocr_slug': pre_ocr_slug, 'pre_ocr_score': round(float(pre_ocr_score), 4)}
    if not cfg.ocr.enabled:
        return {**base, 'final_slug': pre_ocr_slug, 'stage': 'retrieval',
                'reason': 'ocr_disabled'}

    try:
        ocr = extractor.extract(query_image)
    except Exception as e:  # noqa: BLE001 — OCR не должен ронять пайплайн
        ocr = {'error': f'ocr_exception: {e}', 'fields': {}}
    fields = ocr.get('fields', {})
    if not fields or not any((v or '').strip() for v in fields.values()):
        return {**base, 'final_slug': pre_ocr_slug, 'stage': 'retrieval',
                'reason': 'ocr_empty', 'ocr_error': ocr.get('error')}

    candidates = matcher.match(fields)
    if not candidates:
        return {**base, 'final_slug': pre_ocr_slug, 'stage': 'retrieval',
                'reason': 'no_csv_match', 'ocr_fields': fields, 'top_candidates': []}

    top = candidates[0]
    csv_conf = top['confidence']
    min_conf = max(float(cfg.ocr.min_confidence), float(cfg.csv_match.min_score))
    if csv_conf < min_conf:
        return {**base, 'final_slug': pre_ocr_slug, 'stage': 'retrieval',
                'reason': 'low_csv_confidence', 'csv_confidence': csv_conf,
                'ocr_fields': fields, 'top_candidates': candidates[:3]}

    # «скор поиска по csv более значим, чем уверенность ответа модели до OCR»
    if csv_conf <= pre_ocr_score + float(cfg.visual.confidence_margin):
        return {**base, 'final_slug': pre_ocr_slug, 'stage': 'retrieval',
                'reason': 'csv_conf_not_higher', 'csv_confidence': csv_conf,
                'ocr_fields': fields, 'top_candidates': candidates[:3]}

    vis = 1.0
    if cfg.visual.enabled:
        vis = verifier.verify(query_crop, top['photo_file'], top['slug'])

    if vis >= float(cfg.visual.min_similarity):
        return {**base, 'final_slug': top['slug'], 'stage': 'ocr_rerank',
                'reason': 'visual_ok', 'csv_confidence': csv_conf,
                'visual_similarity': round(vis, 4), 'ocr_fields': fields,
                'top_candidates': candidates[:3]}

    return {**base, 'final_slug': pre_ocr_slug, 'stage': 'retrieval',
            'reason': 'visual_mismatch', 'csv_confidence': csv_conf,
            'visual_similarity': round(vis, 4), 'ocr_fields': fields,
            'top_candidates': candidates[:3]}
