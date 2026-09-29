# Метрики и датасеты

## Датасеты

| Набор | Разметка | Фото | Что это |
|---|---|---|---|
| **62 базовых** | `data/real_data_with_slug_eval.csv` (`idx;image;Slug;Site;…`) | `data/real_photo/` | фото от организаторов (реальные полочные снимки) |
| **233** | `data/eval.csv` (`image,true_slug`) | `data/eval/` | более широкий набор фото тех же серий |
| **eval2 (123)** | `data/eval2.csv` (`image,true_slug`) | `data/eval2_images/` | новые присланные фото: первая партия (57 папок → 59 фото) + вторая (`data/eval2_batch2.csv`, 52 папки → 64 фото) |
| **негативы (11)** | `data/negatives.csv` (`image,true_slug` — слаг пустой) | `data/negatives/` | вина **вне каталога**: иностранные картинки из интернета + фото вин, которых нет в каталоге, хотя линейка/производитель есть |

Как собирается `eval2` (правило организаторов): папка названа true-slug, внутри — каталожное фото
`000__<slug>.webp|jpg|jpeg` и полочные снимки. Каталожные фото в датасет не идут (переносятся в
`data/eval2_excluded/`), папки, где был только каталожный снимок, исключаются целиком; полочные
копируются в `data/eval2_images/<slug>__NN__<имя>`. Скрипты, которыми это делалось:
`reports/23_ml_dev2_pipeline_eval/batch2/add_eval2_batch2.py` (вторая партия) и
`reports/23_ml_dev2_pipeline_eval/build_eval2.py` (первая).

## Скрипты

### Пул retrieval: recall@K

`ML evaluation/recall_at_k.py` — прогон «как в проде» (энкодер + YOLO + pgvector), считает
recall@1/3/5/6/10/20/30, brand-recall и срезы near-dup/singleton, пишет
`ML evaluation/reports/<tag>.json` и кэш `image_topk.json` (top-K по обеим ветвям — вход для
fusion и для проверок гейта).

```bash
# на хосте (пути берутся из репозитория)
python recall_at_k.py --eval-csv ../../data/real_data_with_slug_eval.csv \
                      --images-dir ../../data/real_photo --k 30 --tag my62
```

### Полный пайплайн (retrieval + OCR): accuracy, стадии, F1

`ML service/pipeline.py` — тот же код, что у сервиса, но пакетно; пишет
`ocr_rerank_report.json` (`summary` + `predictions[]` с `pre_ocr_slug`, `final_slug`, `stage`,
`reason`, `csv_confidence`, `ocr_fields`, `retrieval_topk`) и `ocr_rerank_predictions.csv`.

```bash
python pipeline.py --config ../config/pipeline.yaml \
  --eval-csv ../data/real_data_with_slug_eval.csv --images-dir ../data/real_photo \
  --out-dir ../reports/e2e62 --limit 0
```

В контейнере CLI не годится (он переписывает пути под хост, `apply_host_paths`), поэтому для
прогонов внутри `ml` есть обёртка `reports/23_ml_dev2_pipeline_eval/container_eval.py` —
вызывает тот же `pipeline.run_eval`, но с контейнерными путями:

```bash
docker compose exec ml python /app/tmp/_container_eval.py \
  --eval-csv /app/ref/eval62.csv --images-dir /app/ref/real_photo --out-dir /app/tmp/out62
```

### Вспомогательные харнессы (`ML evaluation/`)

| Скрипт | Зачем |
|---|---|
| `crop_audit.py` | аудит кропа: как детекторы режут бутылку/этикетку (отчёт 15) |
| `encoder_ab.py`, `encoder_errors.py` | сравнение энкодеров и разбор их ошибок |
| `catalog_ocr.py` | OCR-разметка фото каталога → `data/catalog_ocr_fields.csv` (обогащение матчинга) |
| `field_metrics.py`, `analyze_ocr_eval.py` | метрики полей OCR и разбор прогонов |
| `union_text_eval.py`, `train_fusion.py` | объединённый image+text пул и обучение fusion-ранкера |
| `replay_ocr.py` | **реплей по кэшу OCR-полей** — воспроизводимые метрики без живого VLM |
| `audit_refs.py`, `selftest_p0.py` | аудит эталонов и self-test |

### Скрипты метрик, написанные по ходу работы

| Скрипт | Что считает |
|---|---|
| `reports/26_recall_metrics/metrics_step4.py` | accuracy и recall@K (retrieval и с OCR) на 62 + eval2, в двух семантиках истины (`exact` / `same`) |
| `reports/27_gate_negatives/gate_{sweep,eval2,eval3}.py` | подбор порогов гейта «есть в каталоге» по позитивам и негативам (`data/negatives`) |
| `ML evaluation/wine_identity.py` | правило «то же вино» (год/крепость не влияют, сорт/цвет/сахар/линейка влияют) + самотесты |
| `reports/28_ocr_value/ocr_value.py` | сколько даёт OCR: churn реранка (чинит/ломает), карточки от подтверждения OCR, контрфакты гейта |
| `reports/22_web_pipeline_eval/eval_rank.py` | прогон **retrieval-ветки** (как в старом вебе: бутылочный кроп) — ранжирования top-30 в JSON (запускается в контейнере ветки) |
| `reports/22_web_pipeline_eval/metrics.py` | по этим JSON: accuracy, recall@1/5/10/30, in_catalog, TP/FP/FN, P/R/F1, статистика отрыва |
| `reports/23_ml_dev2_pipeline_eval/metrics_ours.py` | по `ocr_rerank_report.json` + `image_topk*.json`: accuracy retrieval/final, стадии, F1 в трёх вариантах (always / gate / gate+ocr) |
| `reports/23_ml_dev2_pipeline_eval/analyze_run.py` | что именно поправил/сломал OCR, список остаточных ошибок с причиной |
| `reports/23_ml_dev2_pipeline_eval/final_metrics2.py` | сводка по всем наборам сразу (62 / eval2 / batch2 / итого) и сравнение пайплайнов |
| `reports/23_ml_dev2_pipeline_eval/container_eval.py` | обёртка `pipeline.run_eval` для запуска **внутри контейнера** |

Запуск (пример, из корня репозитория, при поднятом стеке):

```bash
ML=$(docker compose ps -q ml)
docker cp reports/23_ml_dev2_pipeline_eval/metrics_ours.py "$ML:/app/tmp/"
docker cp "$ML:/app/tmp/out62/ocr_rerank_report.json" reports/run/report62.json
python3 reports/23_ml_dev2_pipeline_eval/metrics_ours.py \
        reports/run/report62.json reports/run/image_topk_62.json
```

## Определения метрик

* **accuracy** — доля фото, где `final_slug` (после OCR) совпал с истиной; отдельно считаем
  `accuracy_retrieval` (до OCR) — по `pre_ocr_slug`. Считается в двух семантиках истины:
  `exact` (совпал слаг) и `same` («то же вино»: год/крепость не влияют, сорт/цвет/сахар/линейка
  влияют — `ML evaluation/wine_identity.py`); текущие числа — `reports/26_recall_metrics.md`.
* **recall@K (retrieval)** — истинный слаг попал в top-K пула retrieval (обе ветки, кроп запроса).
* **recall@K (с OCR)** — то же, но с учётом ре-ранка: `[final_slug] + пул без него` и берём первые K
  (для K=1 это accuracy).
* **F1 решения «отдать финальную карточку»** — положительное решение = сервис показывает карточку:
  `TP` — показал и верно, `FP` — показал и неверно, `FN` — не показал (экран «нет совпадения»),
  хотя вино в каталоге есть. Варианты правила: `always` (отвечаем всегда), `gate` (только при
  `in_catalog`), `gate+ocr` (`in_catalog` или уверенный OCR-ре-ранк).
* **декисивность** — медиана отрыва `top1−top2` у верных ответов и доля верных с отрывом ≥ порога;
  организаторы просят «ощутимый отрыв», чтобы убрать экран с вариантами.

## Где лежат отчёты

* `reports/NN_*.md` — текстовые отчёты по этапам (15–26), в них таблицы метрик и разбор ошибок;
* `reports/<папка>/` — сырые артефакты прогонов: `ocr_rerank_report.json`, `predictions.csv`,
  `image_topk.json`, ранжирования, скрипты прогонов;
* `ML evaluation/reports/` — отчёты харнессов (`recall_at_k`, `image_topk`, `union_*`).

## Важные оговорки по измерениям

* Живой VLM (OpenRouter) **шумит между прогонами** и падает при 403/429 — прогоны с OCR
  недетерминированы на уровне нескольких фото (отчёт 18: поля совпали лишь у 24/62 между двумя
  одинаковыми прогонами). Для воспроизводимых чисел используйте `replay_ocr.py` (кэш полей).
* Сравнивать прогоны можно только на одном и том же индексе (дамп `deploy/archives/pgvector.dump`)
  и одном и том же наборе; ключ индекса должен совпадать с `retrieval.index_model`.
* Retrieval детерминирован: recall@K и `accuracy_retrieval` воспроизводятся бит-в-бит.

