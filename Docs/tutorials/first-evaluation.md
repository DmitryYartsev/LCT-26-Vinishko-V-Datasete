# Оценка на своём наборе фотографий

Вы прогоните сервис по набору фотографий и получите файл с предсказаниями и временем ответа —
по нему видно, как сервис ведёт себя на ваших снимках.

## Что понадобится

* Запущенный стек: `ml` в состоянии `healthy` (`docker compose ps`).
* Папка с фотографиями и манифест к ним.

## Шаг 1. Собрать набор

Набор — папка с фотографиями и TSV-манифест к ним. Такая же структура используется в наборе
`eval/` репозитория:

```
my-set/
├── queries/
│   ├── 0001.jpg
│   ├── 0002.png
│   └── 0003.webp
└── queries.tsv
```

`queries.tsv` — две колонки, разделитель — табуляция, первая строка — заголовок:

```
query_id	image_path
q-000001	0001.jpg
q-000002	0002.jpg
q-000003	0003.jpg
```

Первая колонка — идентификатор запроса, вторая — имя файла относительно папки с фотографиями.
Правильных ответов в манифесте нет: прогон сохраняет только предсказания и время ответа.
Формат TSV выбран потому, что манифест разбирает скрипт `eval/participant_test.sh` — он же считает
контрольную сумму каждой фотографии.

Если у вас есть правильные ответы, это другой формат (CSV `image,true_slug`) — он описан в
[../reference/metrics.md](../reference/metrics.md), там же — как считать по нему метрики.

## Шаг 2. Прогнать

```bash
EVAL_DIR=/абсолютный/путь/my-set docker compose run --rm eval
```

Что происходит: контейнер `eval` дожидается готовности `ml`, запускает скрипт прогона внутри
docker-сети (то есть без прокси и без браузера) и обращается к `http://ml:8080/v1/eval/predict`.
Результат — `predictions.jsonl` в папке набора, по строке на фотографию:

```json
{"query_id":"q-000001","image_path":"0001.jpg","image_sha256":"c975b31e…",
 "predicted_slug":"pino-glyu-2025","latency_ms":15201}
```

## Шаг 3. Посмотреть результат

Скрипт сам печатает итог: сколько фотографий обработано, сколько осталось без предсказания и
статистику времени ответа. Полезно посмотреть на самые медленные и на пустые ответы:

```bash
cd /абсолютный/путь/my-set
jq -r '.predicted_slug' predictions.jsonl | sort | uniq -c | sort -rn | head
jq -r 'select(.predicted_slug == null) | .image_path' predictions.jsonl
jq -s 'map(.latency_ms) | {median: (sort | .[length/2]), max: max}' predictions.jsonl
```

## Шаг 4. Сравнить с истиной

Если у вас есть правильные slug, положите рядом манифест `truth.tsv` (`image_path<TAB>true_slug`)
и посчитайте долю верных ответов:

```bash
python3 - <<'PY'
import json, re
trail = re.compile(r'-\d+(?:[.,]\d+)?$')
same = lambda a, b: a == b or trail.sub('', a) == trail.sub('', b)

truth = {l.split('\t')[0]: l.split('\t')[1].strip()
         for l in open('truth.tsv', encoding='utf-8').read().splitlines()[1:]}
rows = [json.loads(l) for l in open('predictions.jsonl', encoding='utf-8')]
ok_exact = sum(1 for r in rows if r['predicted_slug'] == truth.get(r['image_path']))
ok_same = sum(1 for r in rows if same(r['predicted_slug'] or '', truth.get(r['image_path']) or ''))
print(f'точное совпадение: {ok_exact}/{len(rows)}, «то же вино»: {ok_same}/{len(rows)}')
PY
```

## Чего ожидать

| Наблюдение | Что это значит |
|---|---|
| 5–20 секунд на фотографию | норма: основное время уходит на внешний вызов чтения этикетки |
| около секунды на фотографию | в конфиге выключено чтение этикетки (`ocr.enabled: false`) |
| `predicted_slug: null` | сервис не успел ответить (маленький `EVAL_MAX_TIME`), не смог прочитать изображение или сработала отсечка «вина нет в каталоге» |
| первый ответ дольше остальных | первый скан прогревает модели |

## Куда дальше

| Задача | Страница |
|---|---|
| Понять, как считаются recall, precision, F1 | [../reference/metrics.md](../reference/metrics.md) |
| Настроить пороги «вино есть в каталоге» | [../how-to/tune-catalog-gate.md](../how-to/tune-catalog-gate.md) |
| Разобраться с пустыми ответами | [../how-to/troubleshoot.md](../how-to/troubleshoot.md) |
