#!/bin/sh
# Восстановление дампа pgvector в БД (выполняется в контейнере db-init, образ с pg_restore).
#
# Логика: если в БД уже есть вина — ничего не делаем (идемпотентно). Если база пустая и
# рядом лежит дамп (его кладёт deploy/fetch_data.sh) — восстанавливаем веса индекса, чтобы
# сервису не пришлось 40 минут считать эмбеддинги. Если дампа нет — выходим успешно: сервис
# соберёт индекс сам (в логе будет предупреждение).
set -u

PGHOST="${PGHOST:-db}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-vino}"
PGDATABASE="${PGDATABASE:-vino}"
export PGHOST PGPORT PGUSER PGDATABASE
DUMP_FILE="${DUMP_FILE:-/deploy/archives/pgvector.dump}"

echo "[db-init] проверяю БД $PGUSER@$PGHOST:$PGPORT/$PGDATABASE"
i=0
while [ "$i" -lt 60 ]; do
  if pg_isready -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" >/dev/null 2>&1; then break; fi
  i=$((i + 1))
  sleep 2
done
if [ "$i" -ge 60 ]; then
  echo '[db-init] БД не ответила за 2 минуты — выхожу'
  exit 1
fi

rows="$(psql -tAc 'select count(*) from wines' 2>/dev/null || true)"
if [ -n "$rows" ] && [ "$rows" -gt 0 ] 2>/dev/null; then
  echo "[db-init] в БД уже $rows вин — восстановление не требуется ✓"
  exit 0
fi

if [ ! -f "$DUMP_FILE" ]; then
  echo "[db-init] дампа нет ($DUMP_FILE) — сервис соберёт индекс сам (это долго, ~40 мин)"
  exit 0
fi

echo "[db-init] база пустая, восстанавливаю дамп $DUMP_FILE ($(du -h "$DUMP_FILE" | cut -f1))"
pg_restore --no-owner --no-privileges --clean --if-exists -d "$PGDATABASE" "$DUMP_FILE" 2>&1 \
  | grep -vE '^pg_restore: (warning|error): (extension|schema)' | tail -5 || true

after="$(psql -tAc 'select count(*) from wines' 2>/dev/null || echo 0)"
vec="$(psql -tAc 'select count(*) from wine_vectors' 2>/dev/null || echo 0)"
echo "[db-init] после восстановления: вин=$after, векторов=$vec"
if [ "${after:-0}" -gt 0 ] 2>/dev/null; then
  echo '[db-init] дамп восстановлен ✓ (сервис не будет пересчитывать индекс)'
  exit 0
fi
echo '[db-init] ОШИБКА: дамп не восстановился — сервис соберёт индекс сам, это долго'
exit 1
