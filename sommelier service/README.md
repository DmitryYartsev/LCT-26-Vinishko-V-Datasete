# sommelier service — цифровой сомелье

FastAPI :8090. Пользователь рассказывает (текстом или голосом), к чему и для какого повода нужно вино;
LLM обновляет профиль вкуса, сервис подбирает вина и оценивает каждое отсканированное вино относительно профиля.

Stateless: переписка и профиль хранятся в браузере и приходят в каждом запросе. Аудио не хранится.

| Файл | Что |
|---|---|
| `app.py` | эндпоинты ([Docs/api.md](../Docs/api.md#sommelier-8090)) |
| `prefs.py` | схема профиля, словари, детерминированный скоринг: доля выполненных критериев, штраф ×0.3 за «не хочу», `checks` и вердикт `ok / part / bad` |
| `llm.py` | OpenRouter: диалог → JSON `{reply, profile, suggestions}`; STT (ffmpeg → wav 16 кГц) |
| `store.py` | Postgres, полнотекстовый поиск по описаниям для вкусовых заметок |
| `seed.py` | warm-up: `wines_parsed.jsonl` (парсинг vino-svoe.ru) + `filtered/catalog.csv` → `wine_profiles`; маппинг slug — `slug_map.csv` |

Аналоги (`/v1/analogs`) — профиль строится из самого вина, та же винодельня исключается.

```bash
docker compose exec sommelier python seed.py --force   # перезалить wine_profiles
```

| ENV | По умолчанию | Смысл |
|---|---|---|
| `OPENROUTER_API_KEY` | — | без ключа `/v1/chat` и `/v1/stt` отдают 502 |
| `LLM_MODEL` / `STT_MODEL` | `google/gemini-2.5-flash` | модели OpenRouter |
| `RATING_MIN` | 4.0 | народный рейтинг ниже — «подходит, но рейтинг низкий» |
| `HISTORY_TURNS` | 12 | сколько последних реплик уходит в LLM |
| `LOG_FILE` | — | дублировать JSON-логи в файл |
