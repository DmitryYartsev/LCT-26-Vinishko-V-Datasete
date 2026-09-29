# Оценка

## Прогон

Набор — папка `queries/` (фото) + `queries.tsv` (`query_id<TAB>image_path`, с заголовком), как в `eval/`.

```bash
docker compose run --rm eval                                   # ./eval
EVAL_DIR=/путь/к/набору docker compose run --rm eval           # свой набор
docker compose run --rm -e EVAL_MAX_TIME=60 eval               # таймаут на фото, с
```

Контейнер ждёт healthy `ml`, гоняет `eval/participant_test.sh` против `http://ml:8080/v1/eval/predict`
и пишет `predictions.jsonl` в папку набора (старый файл → `*.bak`), в конце — сводка по латентности.

```json
{"query_id":"q-000001","image_path":"0001.jpg","image_sha256":"…","predicted_slug":"pino-glyu-2025","latency_ms":15201}
```

С хоста (нужны `bash`, `curl`, `jq`) — как в `eval/README.md`.

## Метрики

- **recall@k** — верный slug в первых k. Accuracy = recall@1.
- **«То же вино»** — slug без хвостового числа (год/крепость) совпадает: `…-bryut-115` = `…-bryut-125`,
  но `…-beloe-suhoe-12` ≠ `…-beloe-polusladkoe-12`.

```python
trail = re.compile(r'-\d+(?:[.,]\d+)?$')
same = lambda a, b: a == b or trail.sub('', a) == trail.sub('', b)
```

## Результаты

300 размеченных фото с полки (62 + 171 + 105, минус спорная разметка), семантика «то же вино»:

| recall | retrieval | + OCR gemini-2.5-flash | + OCR gpt-4o-mini | + OCR gemini-2.5-flash-lite |
|---|---|---|---|---|
| @1 | 0.843 | **0.867** | 0.863 | 0.837 |
| @5 | 0.950 | 0.967 | 0.967 | 0.963 |
| @10 | 0.953 | **0.973** | 0.970 | 0.967 |
| @30 | 0.970 | 0.980 | 0.980 | 0.980 |

Гейт «есть в каталоге» (188 фото из каталога, 8 вин вне каталога): ложных карточек 4/8 → 0/8,
карточку сохраняют 165/188 верных ответов.

Латентность: retrieval ~0.5 с на CPU; с OCR 5–20 с (внешний вызов VLM).
