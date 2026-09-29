# 26. Метрики ветки `ml-web` на 62 + eval2 (185 фото)

## Что считалось

* **наборы**: 62 фото (`data/real_photo`, манифест `data/eval62.csv`) и eval2 — 123 фото
  (`data/eval2_images`, `data/eval2.csv`); вместе **185**;
* **истина берётся из манифестов** (`data/eval62.csv`, `data/eval2.csv`) — единый источник,
  поэтому правки разметки подхватываются пересчётом без повторного прогона;
* **две семантики** (уточнение организаторов, `ML evaluation/wine_identity.py`):
  * `exact` — совпал сам слаг;
  * `same` — «то же вино»: год/крепость (хвостовое число слага) **не важны**, а сорт, цвет,
    сахар/тип и линейка — важны (`…-beloe-suhoe-12` и `…-beloe-polusladkoe-12` — разные вина);
* **recall@K (retrieval)** — истина попала в top-K пула retrieval (обе ветки, кроп запроса);
  **recall@K (с OCR)** — OCR-реранк ставит свой ответ первым, дальше идёт пул без него.

## Accuracy

| набор | семантика | без OCR (retrieval) | с OCR-rerank |
|---|---|---|---|
| 62 | exact | 50/62 = 0.8065 | 54/62 = **0.8710** |
| 62 | same | 51/62 = 0.8226 | 55/62 = **0.8871** |
| eval2 (123) | exact | 83/123 = 0.6748 | 83/123 = **0.6748** |
| eval2 (123) | same | 83/123 = 0.6748 | 83/123 = 0.6748 |
| **185** | exact | 133/185 = 0.7189 | 137/185 = **0.7405** |
| **185** | same | 134/185 = 0.7243 | 138/185 = **0.7459** |

## Recall@K

| набор | семантика | recall | @1 | @5 | @10 | @30 |
|---|---|---|---|---|---|---|
| 62 | exact | retrieval | 0.8065 | 0.8548 | 0.8548 | 0.8871 |
| 62 | exact | с OCR | 0.8710 | 0.9194 | 0.9194 | 0.9355 |
| 62 | same | retrieval | 0.8226 | 0.8548 | 0.8548 | 0.8871 |
| 62 | same | с OCR | 0.8871 | 0.9194 | 0.9194 | 0.9355 |
| eval2 (123) | exact/same | retrieval | 0.6911 | 0.9106 | 0.9268 | 0.9431 |
| eval2 (123) | exact/same | с OCR | 0.6748 | 0.9106 | 0.9268 | 0.9431 |
| **185** | exact | retrieval | 0.7297 | 0.8919 | 0.9027 | 0.9243 |
| **185** | exact | с OCR | 0.7405 | 0.9135 | 0.9243 | 0.9405 |
| **185** | same | retrieval | 0.7351 | 0.8919 | 0.9027 | 0.9243 |
| **185** | same | с OCR | 0.7459 | 0.9135 | 0.9243 | 0.9405 |

Как читать: на 62-наборе OCR-rerank улучшает top-1 (+6.5 п.п. exact, +6.5 п.п. same) и не ломает
recall@5–30. На eval2 выигрыш OCR нулевой: там 49% фото — линейка Абрау-Дюрсо, где VLM не читает ни
сорт, ни цвет, ни сахар с лицевой этикетки (см. `reports/23_*`).

Два прогона 62-набора дают 55/62 и 54/62 по `exact` — это ±1 фото живого VLM между запусками
(сравнение полей — отчёт 18). По правилу `same` оба прогона дают **55/62 = 0.8871**: фото, которое
«потерялось» во втором прогоне, отличается от GT только годом (`reports/26_recall_metrics/report62_run{1,2}.json`).

## Правка разметки (сделана перед этим расчётом)

В `data/eval2` переименована папка одного вина (ошибка разметки): было
`fanagoriya-100-ottenkov-krasnogo-saperavi-krasnoe-suhoe-14`, стало
`fanagoriya-100-ottenkov-krasnogo-kaberne-kaberne-sovinon-krasnoe-suhoe-135`
(«100 оттенков красного. Каберне. Каберне Совиньон»). GT исправлен в `data/eval2.csv` и
`data/eval2_batch2.csv` (по 2 фото). Оба слага есть в каталоге, поэтому правка меняет именно
«какое вино на фото», а не доступность ответа. В свежем прогоне первое из этих двух фото теперь
засчитывается верным (модель отдала новый слаг), второе — нет (модель относит его к «Саперави»:
это ошибка модели, а не разметки). Правило «то же вино» на эту пару не влияет — сорта разные.

## Оговорки

* Числа выше — по **полным** прогонам обоих наборов на ветке `ml-web` (свежие отчёты
  `reports/26_recall_metrics/report62.json`, `report_eval2.json`; предыдущий прогон 123 фото
  (`report_eval2_run1_partial.json`) показывал 82/123 = 0.6667 из-за **10 фото, упавших на 403
  OpenRouter** — без сбоя OCR даёт полноценные 83/123).
* Retrieval детерминирован: `recall@K` и `accuracy` без OCR воспроизводятся бит-в-бит.
* Прогон с OCR недетерминирован на уровне единиц фото (живой VLM): см. `reports/25_*`,
  `reports/18_*`, а также пару `report62_run1.json` / `report62_run2.json` (55/62 и 54/62 по
  `exact`, оба 55/62 по `same`).

## Воспроизведение

```bash
# 1) пул retrieval (recall@K) и кэш top-K для 62 и eval2 — внутри контейнера ml
docker compose exec ml python /app/eval_harness/recall_at_k.py --config /app/config/pipeline.yaml \
  --eval-csv /app/ref/eval62.csv --images-dir /app/ref/real_photo --k 30 --tag my62
# 2) полный пайплайн с OCR
docker compose exec ml python /app/tmp/_container_eval.py --eval-csv /app/ref/eval62.csv \
  --images-dir /app/ref/real_photo --out-dir /app/tmp/out62
# 3) метрики (exact + same, retrieval + с OCR)
python3 reports/26_recall_metrics/metrics_step4.py reports/26_recall_metrics.json
```

Артефакты: `reports/26_recall_metrics.json`, `reports/26_recall_metrics/` (кэши top-K, отчёты
прогонов, скрипты `metrics_step4.py`), правило тождества — `ML evaluation/wine_identity.py`.
