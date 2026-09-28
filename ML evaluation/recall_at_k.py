# -*- coding: utf-8 -*-
"""P0-0: честный харнесс recall@K для image-retrieval (README `ML evaluation`).

Зачем: accuracy top-1 (0.72) скрывает, ЧТО именно теряется. Здесь измеряем
recall@K, brand-recall (винодельня) и разбивку по «серийным» (near-dup) винам,
а также кэшируем image top-K со score.

Прогон — реальный (encoder + YOLO + pgvector), как в проде:
    python3 recall_at_k.py                 # 233 фото, k=60
    python3 recall_at_k.py --k 30 --limit 20

Артефакты (внутри репо):
  * ``reports/recall_at_k.json`` / ``.md`` — метрики;
  * ``reports/image_topk.json``            — кэш top-K (обе ветки).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'ML service'))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from paths import EVAL_CSV, EVAL_IMAGES, REFERENCE_CSV, EVAL_REPORTS  # noqa: E402
from pipeline_config import load_config, apply_retrieval_env        # noqa: E402
import pipeline as P                                                # noqa: E402


def load_wineries() -> dict:
    """slug -> винодельня (для brand-recall и среза near-dup «серий»)."""
    out = {}
    with open(REFERENCE_CSV, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f, delimiter=';'):
            s = (r.get('Slug') or '').strip()
            if s and s not in out:
                out[s] = (r.get('Винодельня') or '').strip()
    return out


def family_key(slug: str, winery: str):
    """Ключ «серии» (near-dup): винодельня + первые 3 токена слага без винтажа.

    Шире, чем просто винодельня (у одной марки много линеек), но уже, чем
    глобальный slug: «fanagoriya-formula-q-*», «…vedernikov-gubernatorskoe-*».
    """
    import ocr_rerank
    base, _ = ocr_rerank.parse_slug(slug or '')
    return (winery, tuple(base[:3]))


def build_ranker(cfg, k):
    """Энкодер + коннект + функция rank(img) -> dict(res_a, res_b, winner)."""
    import db
    from encoder import get_encoder
    from crop import maybe_crop, maybe_label_crop, use_query_policy, USE_LABEL_BRANCH, LABEL_SUFFIX

    enc = get_encoder()
    conn = db.connect()
    model = str(cfg.retrieval.index_model) if 'index_model' in cfg.retrieval else enc.model_name
    mode = str(cfg.retrieval.pipeline)

    def rank(img):
        use_query_policy()                            # кроп ЗАПРОСА: политика query_crop_*
        bottle, _ = maybe_crop(img)
        want_a = mode in ('bottle', 'combined')
        want_b = USE_LABEL_BRANCH and mode in ('label', 'combined')
        label, found, _ = maybe_label_crop(bottle)
        if want_b and not found:
            label = bottle
        ra = db.search(conn, enc.embed([bottle])[0], model, k=k) if want_a else None
        rb = db.search(conn, enc.embed([label])[0], model + LABEL_SUFFIX, k=k) if want_b else None
        if ra is None:
            winner = 'label'
        elif rb is None:
            winner = 'bottle'
        else:
            winner = 'bottle' if ra[0]['score'] >= rb[0]['score'] else 'label'
        return {'bottle': ra, 'label': rb, 'winner': winner}

    return rank


def rank_of(slugs, true):
    return (slugs.index(true) + 1) if true in slugs else None


def _recall(records, k, key='rank'):
    return round(sum(1 for r in records if r.get(key) and r[key] <= k) / max(len(records), 1), 4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='recall@K harness (P0-0)')
    ap.add_argument('--config', default=str(REPO / 'config' / 'pipeline.yaml'))
    ap.add_argument('--eval-csv', default=str(EVAL_CSV))
    ap.add_argument('--images-dir', default=str(EVAL_IMAGES))
    ap.add_argument('--k', type=int, default=60)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tag', default='recall_at_k')
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    P.apply_host_paths(cfg)
    apply_retrieval_env(cfg)
    K = int(args.k)
    EVAL_REPORTS.mkdir(parents=True, exist_ok=True)

    rank = build_ranker(cfg, K)
    rows = P.read_eval_rows(Path(args.eval_csv))
    if args.limit:
        rows = rows[:args.limit]
    wineries = load_wineries()
    fam_size = {}
    for s, w in wineries.items():
        if w:
            k = family_key(s, w)
            fam_size[k] = fam_size.get(k, 0) + 1

    from PIL import Image
    images_dir = Path(args.images_dir)
    records, cache = [], {}
    t0 = time.time()
    for i, (img_name, true) in enumerate(rows):
        p = images_dir / img_name
        rec = {'image': img_name, 'true_slug': true, 'rank': None,
               'bottle_rank': None, 'label_rank': None}
        if not p.exists():
            rec['error'] = 'missing'
            records.append(rec)
            continue
        try:
            r = rank(Image.open(p).convert('RGB'))
        except Exception as e:  # noqa: BLE001
            rec['error'] = str(e)[:200]
            records.append(rec)
            continue
        b_slugs = [x['slug'] for x in (r['bottle'] or [])]
        l_slugs = [x['slug'] for x in (r['label'] or [])]
        win_slugs = b_slugs if r['winner'] == 'bottle' else l_slugs
        rec['rank'] = rank_of(win_slugs[:K], true)
        rec['bottle_rank'] = rank_of(b_slugs, true)
        rec['label_rank'] = rank_of(l_slugs, true)
        rec['winner'] = r['winner']
        rec['top'] = win_slugs[:K]
        rec['not_in_top30'] = not (rec['rank'] and rec['rank'] <= 30)
        w_true = wineries.get(true, '')
        rec['series'] = bool(w_true and fam_size.get(family_key(true, w_true), 0) > 1)
        rec['brand_in_top1'] = bool(win_slugs and wineries.get(win_slugs[0], '') == w_true and w_true)
        rec['brand_in_topK'] = bool(w_true and any(wineries.get(s) == w_true for s in win_slugs[:K]))
        records.append(rec)
        cache[img_name] = {'true_slug': true, 'winner': r['winner'],
                           'bottle': [{'slug': x['slug'], 'score': x['score']} for x in (r['bottle'] or [])],
                           'label': [{'slug': x['slug'], 'score': x['score']} for x in (r['label'] or [])]}
        if (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(rows)}  {time.time() - t0:.0f}s', flush=True)

    ks = [1, 3, 5, 6, 10, 20, 30, K]
    series = [r for r in records if r.get('series')]
    uniq = [r for r in records if not r.get('series')]
    summary = {
        'n': len(records), 'k': K,
        'recall': {f'recall@{k}': _recall(records, k) for k in ks},
        'recall_bottle': {f'recall@{k}': _recall(records, k, 'bottle_rank') for k in (1, 5, 30)},
        'recall_label': {f'recall@{k}': _recall(records, k, 'label_rank') for k in (1, 5, 30)},
        'brand_recall@1': round(sum(r.get('brand_in_top1', False) for r in records) / max(len(records), 1), 4),
        'brand_recall@K': round(sum(r.get('brand_in_topK', False) for r in records) / max(len(records), 1), 4),
        'slices': {
            'series_near_dup': {'n': len(series), 'recall@1': _recall(series, 1), 'recall@30': _recall(series, 30)},
            'singleton': {'n': len(uniq), 'recall@1': _recall(uniq, 1), 'recall@30': _recall(uniq, 30)},
        },
        'not_in_top30': sum(1 for r in records if r.get('not_in_top30')),
        'seconds': round(time.time() - t0, 1),
    }
    (EVAL_REPORTS / f'{args.tag}.json').write_text(
        json.dumps({'summary': summary, 'predictions': records}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    cache_name = 'image_topk.json' if not args.limit else f'image_topk_{args.tag}.json'
    (EVAL_REPORTS / cache_name).write_text(
        json.dumps({'k': K, 'images': cache}, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('отчёты ->', EVAL_REPORTS)
    return 0


if __name__ == '__main__':
    sys.exit(main())

