# deploy

| Файл | Сервис compose | Что |
|---|---|---|
| `fetch_data.py` | `fetch` | качает недостающие архивы (Google Drive) и SigLIP2 (HuggingFace), только stdlib |
| `db_init.sh` | `db-init` | пустая БД + есть `archives/pgvector.dump` → `pg_restore`; иначе выходит |
| `run_eval.sh` | `eval` | гоняет `eval/participant_test.sh` против `http://ml:8080` |
| `Caddyfile` | `https` | HTTPS перед `web` (адрес — `PUBLIC_HOST`) |
| `archives/` | — | сюда `fetch` кладёт дамп индекса (gitignore) |

```bash
docker compose run --rm fetch --status | --check | --force | --only data,dump
docker compose logs db-init
```

Обновить данные: перезалить архив в Google Drive, поправить `gid` и `expect` в `ARCHIVES`
(`fetch_data.py`), затем `docker compose run --rm fetch --force --only <имя>`.

Деплой, HTTPS, индекс, диагностика — [Docs/operations.md](../Docs/operations.md).
