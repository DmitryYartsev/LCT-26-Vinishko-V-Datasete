# ML evaluation — валидация качества поиска

Численно меряет качество retrieval на наборах query, чтобы сравнивать конфиги
(кроп on/off, модель, OCR и т.д.), а не гадать по нескольким фото.

## Файлы

| Файл | Что |
|---|---|
| `build_queryset.py` | собирает набор query → `querysets/*.jsonl` (пока: scrape) |
| `evaluate.py` | queryset + индекс → recall@1/@5, winery@1, margin → `reports/*.json` |
| `analyze_ocr_eval.py` | метрики прогона OCR-rerank по `ocr_rerank_report.json` → `OCR_report_run.md` |
| `replay_ocr.py` | офлайн-калибровка весов/гейта OCR-rerank БЕЗ вызовов VLM (replay по сохранённым полям) |
| `recall_at_k.py` | **P0-0**: recall@K + brand-recall + срез near-dup/singleton; кэш image top-K |
| `audit_refs.py` | **P0-1**: аудит эталонов (фото) и разметки (OCR winery ⟷ карточка) |
| `selftest_p0.py` | самопроверки парсинга ответа VLM (без сети/VLM) |
| `encoder_ab.py` | A/B энкодеров по recall@K (один каталог/препроцесс): SigLIP2 vs OpenRouter-эмбеддинги |
| `querysets/` | наборы `{query_id, image_path, true_slug, source, in_catalog}` |
| `reports/` | результаты прогонов |

## Запуск

```bash
cd "ML evaluation"
uv run python build_queryset.py                                   # -> querysets/scrape.jsonl
# сравнение кроп off vs on (индексы должны быть построены обоими):
CROP_ENABLED=0 uv run python evaluate.py --queryset querysets/scrape.jsonl --tag scrape_base
CROP_ENABLED=1 uv run python evaluate.py --queryset querysets/scrape.jsonl --tag scrape_crop
```

## Прогон OCR-rerank на `data/eval` + метрики

```bash
# 1) живой прогон пайплайна (VLM вызывается 1 раз на фото, ~30 мин на 233 фото)
cd "../ML service"
python3 -u pipeline.py --config ../config/pipeline.yaml \
    --eval-csv ../data/eval.csv --images-dir ../data/eval --out-dir ../data/reports

# 2) метрики по отчёту -> data/reports/OCR_report_run.md
cd "../ML evaluation"
python3 analyze_ocr_eval.py
```

## Офлайн-калибровка OCR-rerank (без повторных вызовов VLM)

`replay_ocr.py` прогоняет РЕАЛЬНЫЙ `ocr_rerank.rerank`, подменяя только сетевой
`OcrExtractor` заглушкой, возвращающей сохранённые в отчёте `ocr_fields`
(short-list — из `retrieval_topk`). Поэтому подбор `csv_match.weights` /
`decision_margin` / ширины пула мгновенный и совпадает с живым прогоном.

```bash
cd "ML evaluation"
python3 replay_ocr.py                       # метрики текущего конфига
python3 replay_ocr.py --pool 6 --margin 0.20
python3 replay_ocr.py --sweep               # сетка pool x margin
```

Пороги выбираются по **2-fold кросс-валидации** (см. результаты ниже), а не по
единственному прогону — иначе легко переобучиться на 233 фото.

## Результат OCR-rerank на `data/eval` (233 фото)

Живой прогон (`pipeline.py --eval-csv ../data/eval.csv --images-dir ../data/eval`)
и офлайн-replay сходятся точка-в-точку:

| | retrieval (до OCR) | final (с OCR-rerank) |
|---|---|---|
| верно | 168/233 = **0.7210** | **178/233 = 0.7639** |

- OCR переключил ответ: **20**, из них починил **10**, сломал **0** → net **+10**.
- Дискриминаторы near-dup «сестёр» (год/цвет/тип) реально работают: примеры fixed —
  шампанское Victor Dravigny brut rosé vs brut blanc, «Черный принц» extra brut
  limited edition vs обычный, Массандра Мускатель чёрный vs портвейн Гурзуф.
- Порог `decision_margin=0.20`, пул `top_k=6`, `missing_credit=0.5` выбраны по
  2-fold кросс-валидации (gain положителен на **обоих** фолдах, broken=0), а не
  по единственному прогону.

## Результаты P0 (233 eval-фото, `data/eval.csv`)

| Метрика | Значение |
|---|---|
| image top-1 (baseline) | 0.7253 |
| OCR-rerank (текущий гейт) | 0.7639 |
| **recall@6 / @30 визуального пула** | 0.8584 / **0.8755** |
| истина НЕ в шортлисте даже top-30 | 29/233 = 12.4% |
| text-only top-1 (полный каталог, e5) | 0.515 |
| **покрытие union(image top-30, text top-10)** | **0.9442** |
| **fusion hill-climb, 2-fold CV** | **0.7725** (folds 0.7863 / 0.7586, broken=0) |

Ключевые выводы: (1) визуальный ретривер сам по себе не даёт 90% (потолок ≈88%);
(2) объединение с текстовым путём поднимает **потолок** до 94%;
(3) наивное расширение пула со старым гейтом **ухудшает** (`broken=8`) — нужен
калиброванный ранкер; (4) fusion уже даёт +0.9 п.п. к текущему, но упирается в
**качество извлечения полей**: год прочитан лишь в 24/55 ошибок, `year_match`
в оракуле по одному признаку чинит 0/51. Отсюда P0-4 (сильнее VLM/кроп).

> Текстовый путь (e5) и fusion-ранкер так и не были включены в прод (`fusion.enabled: false`,
> веса не сданы) и удалены из кода; строки таблицы про них — исторический эксперимент.

**Обновление (отчёты 15–18):** числа выше — записанный baseline до правки кропа запроса.
С политикой `query_crop_*` (включена в `config/pipeline.yaml`, индекс не пересобирается)
на тех же 233 фото: recall@1 0.7253 → **0.7511**, recall@30 0.8755 → **0.9013**,
истина вне top-30 29 → **23**; на 62 фото живой прогон с OCR-rerank:
exact-slug 0.7097 → **0.7903** (`reports/17_recall_233_query_policy.md`,
`reports/18_e2e62_ocr_rerank.md`).

```bash
cd "ML evaluation"
python3 recall_at_k.py       # -> reports/{recall_at_k.json,image_topk.json}
python3 audit_refs.py        # -> reports/audit_refs{,_suspicious}.{json,csv}
python3 selftest_p0.py       # 5 проверок
```

## Метрики

- **recall@1 / recall@5** — доля, где правильный slug на 1-м месте / в top-5.
- **winery@1** — угадана ли винодельня top-1 (показывает near-dup разрыв: бренд vs SKU).
- **mean top1 score / margin** — уверенность и отрыв от 2-го.

## Важно про источники

- **scrape** — метки ШУМНЫЕ (в папке вина бывают чужие кадры) → абсолютные числа занижены,
  но СРАВНЕНИЕ конфигов валидно (шум одинаков для обоих).
- Планируется добавить: **synthetic** (аугментации эталонов), **studio-holdout** (реальные
  запасные фото), **oocatalog** (негативы — вина, спрятанные из индекса, для калибровки порога).

## Первый замер (595 scrape-query, ~99 вин)

| | crop OFF | crop ON |
|---|---|---|
| recall@1 | 0.178 | **0.205** |
| recall@5 | 0.380 | **0.442** |
| winery@1 | 0.662 | **0.721** |

Вывод: кроп — чистый плюс; winery@1≫recall@1 = масштаб near-dup (бренд узнаётся, SKU нет).
