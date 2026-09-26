# ML train — metric learning для энкодера этикеток

Обучение энкодера кропов этикеток методом metric learning (triplet loss): одинаковые
этикетки (один slug) стягиваются в эмбеддинг-пространстве, разные — отталкиваются.
pytorch-lightning + трекинг в ClearML.

## Формат данных

Папка `--data-dir` с четырьмя CSV (имена колонок определяются автоматически,
разделитель — тоже; можно задать явно через `--path-col` / `--slug-col` / `--sep`):

| файл            | что это |
|-----------------|---------|
| `reference.csv` | референсная база: эталонные картинки каталога (по ней считаем accuracy и строим галерею) |
| `train.csv`     | кропы этикеток для трейна (колонки: путь + slug) |
| `val.csv`       | кропы для валидации |
| `test.csv`      | кропы для финального замера accuracy |

## Запуск

```bash
# смоук-тест без реальных данных и весов (игрушечный бэкбон 'tiny')
python make_dummy_data.py
python train.py --backbone tiny --data-dir dummy_data --epochs 6 --clearml offline

# реальное обучение (SigLIP 2 из локальной папки; ClearML на сервер)
python train.py --data-dir /path/to/csv --backbone ../../models/siglip2-base-patch16-256 \
    --epochs 15 --p 8 --k 4 --unfreeze-last-n 2 --clearml on --task-name siglip2-v1
```

## Как это работает

* **Лосс** — triplet (по умолчанию batch-hard, `--mining`, `--distance`, `--triplet-margin`);
  опционально к нему добавляется aux cross-entropy (`--ce-weight`).
* **Семплер** — PK: P классов × K картинок в батче (`--p`, `--k`), поэтому позитивные пары
  в батче гарантированы.
* **Референсная база** — `--include-reference train` подмешивает эталоны в трейн как
  «опорные» позитивы; в валидации всегда считается `losses/val_ref` (кроп против эталона)
  и accuracy по галерее из эталонов.
* **Валидация** — пересчёт эмбеддингов val-сета + галереи, triplet на всём сете,
  `metrics/val_acc@1`, `metrics/val_acc@k` (это же — метрика выбора чекпоинта).
* **Тест** — в конце обучения прогон по `test.csv` с лучшим чекпоинтом:
  `evaluate.run_test` строит галерею из `reference.csv`, считает accuracy
  (`metrics.ExactSlugAccuracy` — пока точное совпадение slug, отдельный заменяемый
  метод в `metrics.py`), сохраняет отчёты и t-SNE-картинку эмбеддингов.
* **ClearML** — скаляры (все лоссы/метрики/lr), конфиги, графики, чекпоинт и отчёты
  теста как артефакты; `--clearml offline` пишет локальную сессию без сервера.

## Артефакты прогона (`artifacts/<run>/`)

`plots/` — кривые лоссов/метрик (PNG + history.csv/json), `reports/` — JSON/CSV теста
и предсказания, `checkpoints/` — best/last, `dataset_stats.json`, `label_space.json`.

## Самопроверка

```bash
python selftest.py          # быстрые юнит-проверки метрики, лоссов, семплера, CSV
```
