# -*- coding: utf-8 -*-
"""Обновлённый пайплайн предсказаний: image retrieval + OCR-rerank.

Все параметры — из YAML через OmegaConf (по умолчанию ``../config/pipeline.yaml``).

Запуск (одиночное фото):
    OPENROUTER_API_KEY=... python pipeline.py --config ../config/pipeline.yaml \
        --image /path/to/photo.webp

Запуск (оценка accuracy по eval-CSV):
    OPENROUTER_API_KEY=... python pipeline.py --config ../config/pipeline.yaml \
        --eval-csv data/real_data_with_slug_eval.csv --images-dir "data/Реальные фото"

Внутри: пре-OCR retrieval (SigLIP2 + YOLO-кроп + pgvector) -> OCR (gpt-4o-mini,
1 вызов) -> самописный мэтчинг по found_in_catalog_corrected.csv -> при
необходимости визуальная проверка фото каталога.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image
from pipeline_config import load_config, apply_retrieval_env

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))


def load_env_file(path: Path) -> None:
    """Простой парсер .env (KEY="value" или KEY=value) — НЕ перезаписывает существующие."""
    if not path.is_file():
        return
    for ln in path.read_text(encoding='utf-8').splitlines():
        ln = ln.strip()
        if not ln or ln.startswith('#') or '=' not in ln:
            continue
        k, v = ln.split('=', 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def apply_host_paths(cfg) -> None:
    """Переопределяет Docker-пути конфига на пути рабочей станции.

    ``pipeline.py`` — standalone-прогон НА ХОСТЕ. Конфиг по умолчанию содержит
    контейнерные пути (/app/...), которые на хосте не существуют. Здесь они
    заменяются на абсолютные пути рабочей станции. Ключ pgvector-индекса
    (``index_model``) оставляем контейнерным — индекс строился в контейнере.
    """
    cfg.paths.models_dir = str(REPO / 'models')
    # Каталог, эталоны, промт и файл OCR-разметки читаем ИЗ РЕПОЗИТОРИЯ (всё внутри
    # <repo>/data и <repo>/ML service; внешних рабочих папок больше нет).
    cfg.paths.reference_csv = str(REPO / 'data' / 'found_in_catalog_corrected.csv')
    cfg.paths.start_photos = str(REPO / 'data' / 'start_photos')
    cfg.paths.prompt_file = str(REPO / 'ML service' / 'prompts_wine_match.txt')
    cfg.paths.catalog_ocr_fields = str(REPO / 'data' / 'catalog_ocr_fields.csv')
    for key in ('reference_csv', 'start_photos', 'prompt_file'):
        p = Path(getattr(cfg.paths, key))
        if not p.exists():
            print(f'[paths] ВНИМАНИЕ: не найден {key}: {p}', flush=True)
    cfg.retrieval.model = str(REPO / 'models/siglip2-base-patch16-256')
    cfg.retrieval.crop_model = str(REPO / 'models/yolo11n.pt')
    cfg.retrieval.label_model = str(REPO / 'models/label_det_best.pt')
    cfg.retrieval.database_url = 'postgresql://vino:vino@127.0.0.1:5432/vino'
    cfg.retrieval.index_model = '/app/models/siglip2-base-patch16-256'
    # output.* в конфиге — контейнерные (/app/...); для хоста переписываем в репозиторий
    cfg.output.eval_csv = str(REPO / 'data' / 'real_data_with_slug_eval.csv')
    cfg.output.images_dir = str(REPO / 'data' / 'real_photo')
    cfg.output.out_dir = str(REPO / 'reports')


def build_rank(cfg):
    """Возвращает rank(img) -> (res, bottle, label) — копия логики app._rank."""
    import db
    from encoder import get_encoder
    from crop import (maybe_crop, maybe_label_crop, use_query_policy,
                      USE_LABEL_BRANCH, LABEL_SUFFIX)

    enc = get_encoder()
    conn = db.connect()
    # ключ индекса в pgvector может отличаться от пути загрузки энкодера
    # (напр. host-путь vs /app/models/... в контейнере)
    model = str(cfg.retrieval.index_model) if 'index_model' in cfg.retrieval \
        else enc.model_name
    pipeline = str(cfg.retrieval.pipeline)
    k = int(cfg.retrieval.top_k)

    def rank(img):
        use_query_policy()                            # кроп ЗАПРОСА: политика query_crop_*
        bottle, _ = maybe_crop(img)
        want_a = pipeline in ('bottle', 'combined')
        want_b = USE_LABEL_BRANCH and pipeline in ('label', 'combined')
        label, found, _ = maybe_label_crop(bottle)
        if want_b and not found:
            label = bottle
        res_a = db.search(conn, enc.embed([bottle])[0], model, k=k) if want_a else None
        res_b = db.search(conn, enc.embed([label])[0], model + LABEL_SUFFIX, k=k) if want_b else None
        if res_a is None:
            res = res_b
        elif res_b is None:
            res = res_a
        elif res_a[0]['score'] >= res_b[0]['score']:
            res = res_a
        else:
            res = res_b
        return res, bottle, label

    return rank, enc, conn


def make_label_crop_fn():
    """crop_fn для OCR-верификатора: применяется к ЭТАЛОННОМУ фото (запрос уже кропнут).

    Поэтому политика всегда индексная (`index_policy()`): иначе активная query-политика
    запроса сдвигала бы кропы референсов и ломала сравнение «запрос vs эталон».
    """
    from crop import maybe_crop, maybe_label_crop, index_policy

    def crop_fn(img):
        with index_policy():                          # эталон — политика индекса
            bottle, _ = maybe_crop(img)
            label, found, _ = maybe_label_crop(bottle)
        return label if found else bottle

    return crop_fn


def build_fusion(cfg, matcher):
    """FusionReranker (P0-3) либо None (``fusion.enabled=false`` — прежний гейт)."""
    if not (getattr(cfg, 'fusion', None) is not None and bool(getattr(cfg.fusion, 'enabled', False))):
        return None
    import rerank_fusion
    import text_retrieval
    from paths import FUSION_WEIGHTS
    tr = text_retrieval.TextRetriever(cfg, use_e5=bool(getattr(cfg.text, 'use_e5', True))) \
        if getattr(cfg, 'text', None) is not None else None
    wpath = str(getattr(cfg.fusion, 'weights_file', FUSION_WEIGHTS))
    if wpath.startswith('/app/'):                    # docker-путь на хосте недоступен
        wpath = str(FUSION_WEIGHTS)
    return rerank_fusion.FusionReranker(cfg, matcher, tr, wpath)


def build_ocr_components(cfg, enc):
    import ocr_rerank as ocr

    extractor = ocr.OcrExtractor(cfg)
    matcher = ocr.CsvMatcher(cfg)
    crop_fn = make_label_crop_fn() if cfg.visual.enabled else None
    verifier = ocr.VisualVerifier(cfg, enc.embed, crop_fn)
    fusion = build_fusion(cfg, matcher)
    return extractor, matcher, verifier, fusion


def predict_one(cfg, img_path, rank, enc, extractor, matcher, verifier, fusion=None):
    img = Image.open(img_path).convert('RGB')
    res, bottle, label = rank(img)
    pre_ocr_slug = res[0]['slug'] if res else None
    pre_ocr_score = res[0]['score'] if res else 0.0
    import ocr_rerank
    ocr_crop = None
    if bool(cfg.ocr.get('use_label_crop', True)):
        from crop import ocr_label_crop
        try:
            ocr_crop, _ = ocr_label_crop(bottle)      # выпрямленный кроп этикетки для VLM
        except Exception as e:  # noqa: BLE001 — не ронять прогон
            print(f'[ocr] ocr_label_crop failed: {e}', flush=True)
    out = ocr_rerank.rerank(cfg, img, label, pre_ocr_slug, pre_ocr_score,
                            extractor, matcher, verifier, ocr_image=ocr_crop,
                            pre_ocr_candidates=[r['slug'] for r in (res or [])],
                            bottle_crop=bottle)
    if fusion is not None and out.get('ocr_fields'):
        # fusion работает поверх union-пула (image top-K + текст) на уже
        # извлечённых полях — повторный вызов VLM не нужен.
        try:
            fr = fusion.rerank(out['ocr_fields'],
                               [{'slug': r['slug'], 'score': r['score']} for r in (res or [])],
                               query_crop=label, verifier=verifier,
                               image_k=int(getattr(cfg.fusion, 'image_k', 30)),
                               text_k=int(getattr(cfg.fusion, 'text_k', 10)))
            if fr.get('final_slug'):
                out['final_slug'] = fr['final_slug']
                out['stage'] = 'fusion'
                out['fusion_pool_size'] = fr.get('pool_size')
                out['fusion_top_candidates'] = fr.get('top_candidates')
        except Exception as e:  # noqa: BLE001 — fusion не должен ронять прогон
            print(f'[fusion] {e}', flush=True)
    out['top5_retrieval'] = [{'slug': r['slug'], 'score': r['score']} for r in (res or [])]
    return out


def read_eval_rows(eval_csv: Path) -> list:
    """Манифест оценки -> [(image_name, true_slug)] с автоопределением формата.

    Поддержаны оба формата, встречающиеся в проекте:
      * ``image,true_slug`` — ``data/eval.csv`` (запятая, колонки по заголовку);
      * ``<idx>;image;true_slug`` — старый манифест (``;``, без заголовка).
    """
    text = Path(eval_csv).read_text(encoding='utf-8-sig')
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    sample = '\n'.join(lines[:5])
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=';,').delimiter
    except csv.Error:
        delimiter = ';' if ';' in sample else ','
    parsed = [r for r in csv.reader(lines, delimiter=delimiter)
              if any(c.strip() for c in r)]
    if not parsed:
        return []

    header = [c.strip().lower() for c in parsed[0]]
    if 'true_slug' in header or 'slug' in header:
        slug_col = header.index('true_slug') if 'true_slug' in header else header.index('slug')
        img_col = header.index('image') if 'image' in header else (1 - slug_col)
        data = parsed[1:]
    else:                                    # без заголовка: idx;image;true_slug
        data = parsed
        img_col, slug_col = (1, 2) if len(data[0]) > 2 else (0, 1)

    rows = []
    for r in data:
        if len(r) <= max(img_col, slug_col):
            continue
        img, slug = r[img_col].strip(), r[slug_col].strip()
        if img and slug:
            rows.append((img, slug))
    return rows


def run_eval(cfg, rank, enc, extractor, matcher, verifier, args, fusion=None):
    eval_csv = Path(args.eval_csv)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_eval_rows(eval_csv)
    if args.limit:
        rows = rows[:args.limit]

    preds, correct_pre, correct_final = [], 0, 0
    n_err = 0
    stopped = False
    t0 = time.time()
    # Промежуточная запись отчёта: длинный прогон (233 фото, ~70 мин) может быть прерван
    # (кредиты/сеть/вручную) — сохраняем результаты каждые N фото, чтобы их можно было
    # переиспользовать (`_scratch/split62.py`, `merge_eval.py`), а не терять.
    dump_every = int(os.environ.get('EVAL_DUMP_EVERY', '10') or 10)

    def _dump(partial: bool) -> None:
        rep = {'n': len(preds),
               'accuracy_retrieval': round(correct_pre / len(preds), 4) if preds else 0.0,
               'accuracy_final': round(correct_final / len(preds), 4) if preds else 0.0,
               'correct_retrieval': correct_pre, 'correct_final': correct_final,
               'seen': len(rows), 'ocr_errors': n_err, 'stopped_early': stopped,
               'partial': partial, 'seconds': round(time.time() - t0, 1)}
        tmp = out_dir / 'ocr_rerank_report.json.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'summary': rep, 'predictions': preds}, f, ensure_ascii=False, indent=2)
        tmp.replace(out_dir / 'ocr_rerank_report.json')   # атомарная замена

    for i, (img_name, true_slug) in enumerate(rows):
        img_path = images_dir / img_name
        if not img_path.exists():
            preds.append({'image': img_name, 'true_slug': true_slug,
                          'pre_ocr_slug': None, 'final_slug': None,
                          'stage': None, 'reason': 'image_missing',
                          'pre_ocr_score': None, 'csv_confidence': None,
                          'visual_similarity': None})
            continue
        r = predict_one(cfg, img_path, rank, enc, extractor, matcher, verifier, fusion)
        pre = r['pre_ocr_slug']
        fin = r['final_slug']
        if r.get('ocr_error'):
            n_err += 1
        correct_pre += int(pre == true_slug)
        correct_final += int(fin == true_slug)
        preds.append({'image': img_name, 'true_slug': true_slug,
                      'pre_ocr_slug': pre, 'final_slug': fin,
                      'stage': r['stage'], 'reason': r.get('reason'),
                      'pre_ocr_score': r['pre_ocr_score'],
                      'csv_confidence': r.get('csv_confidence'),
                      'csv_margin': r.get('csv_margin'),
                      'visual_similarity': r.get('visual_similarity'),
                      'ocr_input': r.get('ocr_input'),
                      'ocr_error': r.get('ocr_error'),
                      'retrieval_topk': [t['slug'] for t in r.get('top5_retrieval', [])],
                      'ocr_fields': r.get('ocr_fields')})
        if (i + 1) % dump_every == 0:
            print(f'  {i + 1}/{len(rows)} ({(time.time()-t0)/(i+1):.1f}s/фото)', flush=True)
            _dump(partial=True)
        # Массовый отказ OCR (кредиты/ключ OpenRouter, сеть) делает metric бессмысленной:
        # останавливаем длинный прогон сразу, а не после часа пустых вызовов.
        if (n_err >= 10 and i + 1 >= 20 and i + 1 < len(rows)
                and n_err / (i + 1) >= 0.5):
            print(f'\n[СТОП] OCR отказывает на {n_err} из {i + 1} фото: '
                  f'{r.get("ocr_error")}\n'
                  f'      Прогон остановлен — метрика была бы недостоверной. '
                  f'Проверьте кредиты/ключ OpenRouter ({cfg.ocr.model}).', flush=True)
            stopped = True
            break

    n = len(preds)
    acc_pre = correct_pre / n if n else 0.0
    acc_final = correct_final / n if n else 0.0
    report = {'n': n, 'accuracy_retrieval': round(acc_pre, 4),
              'accuracy_final': round(acc_final, 4),
              'correct_retrieval': correct_pre, 'correct_final': correct_final,
              'seen': len(rows), 'ocr_errors': n_err, 'stopped_early': stopped,
              'partial': stopped, 'seconds': round(time.time() - t0, 1)}
    if stopped:
        report['degraded'] = True
        print(f'\n[!] прогон НЕПОЛНЫЙ ({n} из {len(rows)} фото) и с отказами OCR '
              f'({n_err}) — числа ниже использовать нельзя')
    print(f'\naccuracy retrieval (до OCR): {correct_pre}/{n} = {acc_pre:.4f}')
    print(f'accuracy final (с OCR-rerank): {correct_final}/{n} = {acc_final:.4f}')

    with open(out_dir / 'ocr_rerank_report.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': report, 'predictions': preds}, f, ensure_ascii=False, indent=2)
    with open(out_dir / 'ocr_rerank_predictions.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image', 'true_slug', 'pre_ocr_slug', 'final_slug', 'stage',
                    'reason', 'pre_ocr_score', 'csv_confidence', 'csv_margin',
                    'visual_similarity', 'ocr_input', 'ocr_error'])
        for p in preds:
            w.writerow([p['image'], p['true_slug'], p['pre_ocr_slug'], p['final_slug'],
                        p['stage'], p['reason'], p['pre_ocr_score'],
                        p.get('csv_confidence'), p.get('csv_margin'),
                        p.get('visual_similarity'), p.get('ocr_input'),
                        p.get('ocr_error')])
    print('отчёт ->', out_dir)
    return report
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(REPO / 'config' / 'pipeline.yaml'))
    ap.add_argument('--image', help='одиночное фото для предсказания')
    ap.add_argument('--eval-csv', help='CSV с фото+slug для оценки accuracy')
    ap.add_argument('--images-dir', help='папка с фото (для --eval-csv)')
    ap.add_argument('--out-dir', help='куда писать отчёты')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args(argv)

    # ключ OpenRouter держим в репозитории: <repo>/.env (в .gitignore, переносится на
    # сервер вручную). Ссылок на файлы ВНЕ репозитория нет.
    load_env_file(REPO / '.env')
    if not os.environ.get('OPENROUTER_API_KEY'):
        print(f'[env] нет OPENROUTER_API_KEY: ожидается {REPO / ".env"} '
              f'(KEY="sk-or-...") либо переменная окружения', flush=True)

    cfg = load_config(args.config)
    apply_host_paths(cfg)          # standalone-прогон на хосте
    apply_retrieval_env(cfg)
    import preflight               # понятное сообщение, если данные/модели не скачаны
    if preflight.check(cfg, where='прогон на хосте (pipeline.py)'):
        return 1

    rank, enc, conn = build_rank(cfg)
    extractor, matcher, verifier, fusion = build_ocr_components(cfg, enc)

    if args.image:
        r = predict_one(cfg, Path(args.image), rank, enc, extractor, matcher, verifier, fusion)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0

    if args.eval_csv:
        images_dir = args.images_dir or str(cfg.output.images_dir)
        run_eval(cfg, rank, enc, extractor, matcher, verifier, argparse.Namespace(
            eval_csv=args.eval_csv, images_dir=images_dir,
            out_dir=args.out_dir, limit=args.limit), fusion)
        return 0

    ap.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
