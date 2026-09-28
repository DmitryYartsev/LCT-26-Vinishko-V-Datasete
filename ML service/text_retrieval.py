# -*- coding: utf-8 -*-
"""Полнокаталожный ТЕКСТОВЫЙ путь (P0-2).

Параллельно image-retrieval ищет кандидатов **по всему каталогу** (2108 slug)
по текстовым признакам, прочитанным VLM/OCR с этикетки. Нужен потому, что
правильный SKU near-dup «сестры» часто отсутствует даже в top-30 визуального
шортлиста (см. ``Plans/План до 90.md`` §0), а текст его различает.

Два взаимодополняющих сигнала (оба — по всем записям каталога):
  * ``csv_conf``  — самописный ``CsvMatcher`` (год/цвет/тип/сорт/винодельня);
  * ``e5_cos``    — косинус текстового эмбеддера ``multilingual-e5``
                    (запрос «OCR-поля» ⟷ строка карточки каталога).

Текстовый путь сам по себе слабее визуального top-1, но **дополняет** его:
на 63 реальных фото union(image top-30, text top-10) покрывает 94% истины
(против 87.6% у одного image top-30). Поэтому он отдаёт ПУЛ кандидатов,
а финальное решение принимает fusion-ранкер (``rerank_fusion.py``).

Параметры — из секции ``text:`` конфига (``config/pipeline.yaml``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import ocr_rerank

_NULL = {'', 'null', 'none', 'nan', 'нет', 'не указано', 'не найдено'}


def clean(v) -> str:
    s = str(v or '').strip()
    return '' if s.lower() in _NULL else s


def query_text(fields: dict) -> str:
    """OCR-поля запроса -> одна строка для эмбеддера (год впереди как дискриминатор)."""
    order = ('year', 'winery', 'grape', 'color', 'sugar', 'wine_type', 'additional_text')
    return ' '.join(clean(fields.get(k)) for k in order if clean(fields.get(k)))


def catalog_text(e: dict) -> str:
    """Запись каталога (``CsvMatcher.entries``) -> одна строка для эмбеддера."""
    return ' '.join(clean(e.get(k)) for k in
                    ('title', 'winery', 'grape', 'category', 'color', 'slug') if clean(e.get(k)))


class E5Embedder:
    """Текстовый эмбеддер ``intfloat/multilingual-e5-*`` на transformers (mean-pool + L2).

    Используем transformers напрямую (не sentence-transformers): меньше зависимостей,
    а префиксы e5 (``query:`` / ``passage:``) ставим руками.
    """

    def __init__(self, model_name: str = 'intfloat/multilingual-e5-small', device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def embed(self, texts, prefix: str = 'passage', batch_size: int = 64) -> np.ndarray:
        pre = 'query: ' if prefix == 'query' else 'passage: '
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [pre + (t or '') for t in texts[i:i + batch_size]]
                enc = self.tok(batch, return_tensors='pt', padding=True,
                               truncation=True, max_length=256).to(self.device)
                h = self.model(**enc).last_hidden_state           # [B, T, H]
                mask = enc['attention_mask'].unsqueeze(-1).float()  # [B, T, 1]
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                out.append(pooled.float().cpu().numpy())
        v = np.concatenate(out, axis=0)
        return (v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)).astype(np.float32)

class TextRetriever:
    """Текстовый поиск по ВСЕМУ каталогу: ``csv_conf`` (CsvMatcher) + ``e5_cos``.

    Возвращает список кандидатов с раздельными признаками (``csv_conf``,
    ``e5_cos``) — чтобы fusion-ранкер сам взвешивал сигналы, а не полагался на
    единственную комбинированную оценку.
    """

    def __init__(self, cfg, use_e5: bool = True, e5_model: str | None = None):
        self.cfg = cfg
        self.matcher = ocr_rerank.CsvMatcher(cfg)
        self.entries = {e['slug']: e for e in self.matcher.entries}
        self.slugs = [e['slug'] for e in self.matcher.entries]
        self.use_e5 = bool(getattr(cfg.text, 'use_e5', use_e5)) if hasattr(cfg, 'text') else use_e5
        self._cat_text = [catalog_text(self.entries[s]) for s in self.slugs]
        self.e5 = None
        self._cat_emb = None
        model = e5_model or (str(getattr(cfg.text, 'e5_model', 'intfloat/multilingual-e5-small'))
                             if hasattr(cfg, 'text') else 'intfloat/multilingual-e5-small')
        if self.use_e5:
            try:
                self.e5 = E5Embedder(model)
                self._cat_emb = self.e5.embed(self._cat_text, prefix='passage')
            except Exception as e:  # noqa: BLE001 — e5 не должен ронять текстовый путь
                print(f'[text] e5 недоступен ({e}); работаю только на CsvMatcher', flush=True)
                self.use_e5, self.e5 = False, None

    def full_scores(self, fields: dict, weight_e5: float | None = None):
        """Все записи каталога со скорами. -> (rows_sorted, cos_map). Один прогон e5."""
        if not any(clean(v) for v in (fields or {}).values()):
            return [], {}
        if weight_e5 is not None:
            w = float(weight_e5)
        else:
            w = float(getattr(self.cfg.text, 'e5_weight', 0.5)) if hasattr(self.cfg, 'text') else 0.5
        if not self.use_e5:
            w = 0.0
        csv_by = {c['slug']: float(c['confidence']) for c in self.matcher.match(fields, only=None)}
        cos_map = self.e5_cos_map(fields) if self.use_e5 else {}
        rows = []
        for s in self.slugs:
            csv_conf = csv_by.get(s, 0.0)
            e5_cos = cos_map.get(s, 0.0)
            rows.append({'slug': s, 'csv_conf': round(csv_conf, 4),
                         'e5_cos': round(e5_cos, 4),
                         'score': round((1.0 - w) * csv_conf + w * e5_cos, 4)})
        rows.sort(key=lambda r: r['score'], reverse=True)
        return rows, cos_map

    def search(self, fields: dict, k: int = 20, weight_e5: float | None = None) -> list[dict]:
        """Поля запроса -> top-k кандидатов по всему каталогу (best-first)."""
        rows, _ = self.full_scores(fields, weight_e5)
        return rows[:k]

    def e5_cos_map(self, fields: dict) -> dict:
        """slug -> e5-косинус (для fusion-признаков; считается один раз на запрос)."""
        if not (self.use_e5 and self.e5) or not any(clean(v) for v in (fields or {}).values()):
            return {}
        q = self.e5.embed([query_text(fields)], prefix='query')[0]
        sims = self._cat_emb @ q
        return {s: float(v) for s, v in zip(self.slugs, sims)}

    def raw_cos_map(self, raw_text: str) -> dict:
        """slug -> e5-косинус СЫРОЙ транскрипции этикетки ⟷ карточка (признак P0-4).

        ``raw_text`` (полная вычитка этикетки VLM) часто содержит год/сорт/цвет, которые
        не попали в структурные поля, поэтому это отдельный, более шумный, но полезный сигнал.
        """
        if not (self.use_e5 and self.e5) or not str(raw_text or '').strip():
            return {}
        q = self.e5.embed([str(raw_text)], prefix='query')[0]
        return {s: float(v) for s, v in zip(self.slugs, self._cat_emb @ q)}



def _main(argv=None) -> int:
    """CLI: прогнать текстовый путь на eval-наборе (ocr_fields из отчёта) -> top-1."""
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import DATA_REPORTS
    from pipeline_config import load_config

    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description='Полнокаталожный текстовый поиск (P0-2)')
    ap.add_argument('--config', default=str(repo / 'config' / 'pipeline.yaml'))
    ap.add_argument('--ocr-report', default=str(DATA_REPORTS / 'ocr_rerank_report.json'),
                    help='отчёт с сохранёнными ocr_fields (чтобы не платить за VLM)')
    ap.add_argument('--no-e5', action='store_true')
    ap.add_argument('--k', type=int, default=20)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tr = TextRetriever(cfg, use_e5=not args.no_e5)
    data = json.loads(Path(args.ocr_report).read_text(encoding='utf-8'))
    hit, rows = 0, []
    for p in data['predictions']:
        cands = tr.search(p.get('ocr_fields') or {}, k=args.k)
        top1 = cands[0]['slug'] if cands else None
        hit += int(top1 == p['true_slug'])
        rows.append({'image': p['image'], 'true_slug': p['true_slug'],
                     'text_top1': top1, 'hit': int(top1 == p['true_slug']),
                     'top': [c['slug'] for c in cands]})
    n = len(rows) or 1
    print(f'text-only top1 (full catalog): {hit}/{n} = {hit / n:.4f} (e5={tr.use_e5})')
    out = Path(DATA_REPORTS) / 'text_retrieval_report.json'
    out.write_text(json.dumps({'n': n, 'top1': round(hit / n, 4), 'use_e5': tr.use_e5,
                               'predictions': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('отчёт ->', out)
    return 0


if __name__ == '__main__':
    import sys as _sys
    _sys.exit(_main())

