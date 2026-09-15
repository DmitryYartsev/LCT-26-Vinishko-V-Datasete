# harness — валидация качества поиска

Численно меряет качество retrieval на наборах query, чтобы сравнивать конфиги
(кроп on/off, модель, OCR и т.д.), а не гадать по нескольким фото.

## Файлы

| Файл | Что |
|---|---|
| `build_queryset.py` | собирает набор query → `querysets/*.jsonl` (пока: scrape) |
| `evaluate.py` | queryset + индекс → recall@1/@5, winery@1, margin → `reports/*.json` |
| `querysets/` | наборы `{query_id, image_path, true_slug, source, in_catalog}` |
| `reports/` | результаты прогонов |

## Запуск

```bash
cd harness
uv run python build_queryset.py                                   # -> querysets/scrape.jsonl
# сравнение кроп off vs on (индексы должны быть построены обоими):
CROP_ENABLED=0 uv run python evaluate.py --queryset querysets/scrape.jsonl --tag scrape_base
CROP_ENABLED=1 uv run python evaluate.py --queryset querysets/scrape.jsonl --tag scrape_crop
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
