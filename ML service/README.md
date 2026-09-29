# ML service — распознавание вина по фото

FastAPI :8080. Кроп бутылки (YOLO) → кроп этикетки (YOLO) → выравнивание → SigLIP2 + pgvector →
OCR-rerank → гейт «есть в каталоге». Этапы — [Docs/pipeline.md](../Docs/pipeline.md),
параметры — `config/pipeline.yaml` ([Docs/configuration.md](../Docs/configuration.md)).

| Файл | Что |
|---|---|
| `app.py` | эндпоинты и склейка этапов, гейт `_gate` |
| `crop.py` | кроп бутылки и этикетки, политики кропа индекса/запроса, кроп для OCR |
| `label_align.py` | выравнивание этикетки по 4 углам |
| `encoder.py` | энкодер: картинка → L2-вектор (SigLIP2) |
| `db.py` | pgvector: схема, векторы, поиск (max cosine по slug) |
| `build_index.py` | warm-up: карточки + векторы `filtered/` → pgvector (если их нет) |
| `ocr_rerank.py` | `OcrExtractor` (VLM), `CsvMatcher`, `VisualVerifier`, `rerank()` |
| `prompts_wine_match.txt` | промпт VLM |
| `pipeline_config.py` | загрузка YAML и прокидка в env модулей |
| `preflight.py` | проверка данных и моделей перед стартом |

## Команды

```bash
docker compose logs -f ml
docker compose restart ml                              # после правки кода или config/pipeline.yaml
docker compose exec ml python build_index.py --force   # пересобрать индекс (~40 мин на CPU)
docker compose exec ml python preflight.py             # всё ли скачано
docker compose exec ml python label_align.py           # самопроверка выравнивания
```

## Эндпоинты

`POST /v1/eval/predict`, `POST /v1/search`, `GET /wine/{slug}`, `GET /ref/{slug}`, `GET /health` —
см. [Docs/api.md](../Docs/api.md).
