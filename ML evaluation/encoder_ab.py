# -*- coding: utf-8 -*-
"""A/B энкодеров на одном и том же каталоге/препроцессе (offline, честное сравнение).

Сравнивает recall@K image-retrieval для разных энкодеров (SigLIP2 локально vs
мультимодальный OpenRouter, напр. ``voyageai/voyage-multimodal-3.5``), используя
ОДИН источник каталога (``filtered/`` как в prod-индексе) и ОДИН препроцесс (кроп
бутылки/этикетки). Эмбеддинги кэшируются в ``reports/encab_*.npz``.

    # SigLIP2 (локально)
    python3 encoder_ab.py --encoder siglip --branch label --eval-csv ../data/eval.csv --images-dir ../data/eval

    # OpenRouter (мультимодальный)
    python3 encoder_ab.py --encoder openrouter --or-model voyageai/voyage-multimodal-3.5 \
        --branch label --eval-csv ../data/eval.csv --images-dir ../data/eval

    # 62-набор (real_photo)
    python3 encoder_ab.py --encoder openrouter --branch label \
        --eval-csv ../data/real_data_with_slug_eval.csv --images-dir ../data/real_photo
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WS = REPO.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'ML service'))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from paths import EVAL_REPORTS, FILTERED, REFERENCE_CSV, DATA          # noqa: E402
from pipeline_config import load_config, apply_retrieval_env           # noqa: E402
import pipeline as P                                                   # noqa: E402

IMG_EXT = {'.webp', '.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.tif', '.tiff'}


def load_env_key():
    """Ключ OpenRouter: <repo>/.env (основной) -> легаси-файл вне репо (fallback)."""
    for p in (REPO / '.env', WS / 'code' / '.env'):
        if not p.is_file():
            continue
        for ln in p.read_text(encoding='utf-8').splitlines():
            if ln.strip().startswith('OPENROUTER_API_KEY') and '=' in ln:
                os.environ.setdefault('OPENROUTER_API_KEY',
                                      ln.split('=', 1)[1].strip().strip('"').strip("'"))
        if os.environ.get('OPENROUTER_API_KEY'):
            return


def make_encoder(kind: str, model: str | None, max_side: int | None):
    os.environ['SEARCH_ENCODER'] = kind
    if model:
        os.environ['SEARCH_MODEL'] = model
    if max_side:
        os.environ['OR_EMBED_MAX_SIDE'] = str(max_side)
    import encoder as E
    importlib.reload(E)
    return E.get_encoder()


def catalog_images(source: str):
    """[(slug, img_path)] — по одному (первому) фото на slug, как в prod-индексе."""
    if source == 'start_photos':
        root = DATA / 'start_photos'
        out = []
        for p in sorted(root.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                out.append((p.stem, p))
        return out
    out, seen = [], set()
    for d in sorted(FILTERED.iterdir()):
        if not d.is_dir() or d.name in seen:
            continue
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        if imgs:
            seen.add(d.name)
            out.append((d.name, imgs[0]))
    return out


def prep(img):
    from crop import maybe_crop, maybe_label_crop
    bottle, _ = maybe_crop(img)
    label, found, _ = maybe_label_crop(bottle)
    return label if found else bottle


def prep_index(img):
    """Кроп эталонов/индекса — всегда под индексной политикой (не query_crop_*)."""
    from crop import index_policy
    with index_policy():
        return prep(img)


def _embed_paths(enc, items, no_crop, chunk=128, label='', index=False):
    from PIL import Image
    embs = []
    t0 = time.time()
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        imgs = []
        for slug, p in part:
            im = Image.open(p).convert('RGB')
            imgs.append(im if no_crop else (prep_index(im) if index else prep(im)))
        embs.append(enc.embed(imgs))
        if (i // chunk + 1) % 2 == 0 or i + chunk >= len(items):
            done = min(i + chunk, len(items))
            print(f'  [{label}] {done}/{len(items)}  {time.time() - t0:.0f}s', flush=True)
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 1), dtype=np.float32)


def embed_catalog(enc, tag, source, no_crop, rebuild=False):
    cache = EVAL_REPORTS / f'encab_cat_{tag}_{source}.npz'
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=True)
        print(f'каталог из кэша: {len(z["slugs"])} slug ({cache.name})')
        return list(z['slugs']), z['emb']
    items = catalog_images(source)
    print(f'каталог: {len(items)} slug, источник={source}')
    emb = _embed_paths(enc, items, no_crop, label='cat', index=True)
    slugs = [s for s, _ in items]
    EVAL_REPORTS.mkdir(parents=True, exist_ok=True)
    np.savez(cache, slugs=np.asarray(slugs), emb=emb)
    return slugs, emb


def embed_queries(enc, qtag, rows, images_dir, no_crop, limit=0, rebuild=False):
    cache = EVAL_REPORTS / f'encab_q_{qtag}.npz'
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=True)
        print(f'запросы из кэша: {len(z["names"])} ({cache.name})')
        return list(z['names']), list(z['trues']), z['emb']
    from PIL import Image
    # ЗАПРОС — query-политика кропа (config: query_crop_*), как в сервисе; каталог
    # эмбеддится раньше и идёт политикой индекса (encoder_ab не вызывает переключение).
    from crop import use_query_policy
    ov = use_query_policy()
    if ov:
        print('query-crop политика:', ', '.join(ov))
    if limit:
        rows = rows[:limit]
    items = [(nm, Path(images_dir) / nm) for nm, _ in rows]
    items = [(nm, p) for nm, p in items if p.exists()]
    truth = {nm: tr for nm, tr in rows}
    print(f'запросы: {len(items)}')
    emb = _embed_paths(enc, items, no_crop, label='q')
    names = [nm for nm, _ in items]
    trues = [truth[nm] for nm in names]
    np.savez(cache, names=np.asarray(names), trues=np.asarray(trues), emb=emb)
    return names, trues, emb


def evaluate(slugs, cat_emb, names, q_emb, trues, ks=(1, 5, 10, 30), topk=60):
    slugs_arr = np.asarray(slugs)
    idx = {s: i for i, s in enumerate(slugs)}
    sims = cat_emb @ q_emb.T                       # [C, Q]
    n = len(names)
    out = {f'recall@{k}': 0 for k in ks}
    mrr = 0.0
    missing = 0
    for j in range(n):
        t = trues[j]
        if t not in idx:
            missing += 1
            continue
        order = np.argsort(-sims[:, j])            # полный порядок каталога
        pos = np.where(slugs_arr[order] == t)[0]
        rank = int(pos[0]) + 1 if pos.size else len(slugs_arr) + 1
        for k in ks:
            out[f'recall@{k}'] += int(rank <= k)
        mrr += 1.0 / rank
    present = max(n - missing, 1)
    for k in ks:
        out[f'recall@{k}'] = round(out[f'recall@{k}'] / present, 4)
    out['mrr'] = round(mrr / present, 4)
    out['n'] = n
    out['missing_true_in_catalog'] = missing
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='A/B энкодеров (recall@K)')
    ap.add_argument('--config', default=str(REPO / 'config' / 'pipeline.yaml'))
    ap.add_argument('--encoder', default='siglip', choices=['siglip', 'dinov2', 'openrouter'])
    ap.add_argument('--or-model', default=None, help='модель OpenRouter embeddings')
    ap.add_argument('--branch', default='label', choices=['label', 'bottle'])
    ap.add_argument('--catalog-source', default='filtered', choices=['filtered', 'start_photos'])
    ap.add_argument('--eval-csv', default=str(DATA / 'eval.csv'))
    ap.add_argument('--images-dir', default=str(DATA / 'eval'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no-crop', action='store_true')
    ap.add_argument('--max-side', type=int, default=512)
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args(argv)

    load_env_key()
    cfg = load_config(args.config)
    P.apply_host_paths(cfg)
    apply_retrieval_env(cfg)

    model = args.or_model if args.encoder == 'openrouter' else None
    tag = args.tag or (f'{args.encoder}_{(model or "siglip").split("/")[-1]}_{args.branch}')

    enc = make_encoder(args.encoder, model, args.max_side if args.encoder == 'openrouter' else None)
    print(f'энкодер: {enc.model_name} (kind={getattr(enc, "kind", "?")})')

    slugs, cat_emb = embed_catalog(enc, tag, args.catalog_source, args.no_crop, args.rebuild)
    rows = P.read_eval_rows(Path(args.eval_csv))
    qtag = f'{tag}_{Path(args.eval_csv).stem}'
    names, trues, q_emb = embed_queries(enc, qtag, rows, args.images_dir, args.no_crop,
                                        args.limit, args.rebuild)
    m = evaluate(slugs, cat_emb, names, q_emb, trues)
    m.update({'encoder': args.encoder, 'model': enc.model_name, 'branch': args.branch,
              'catalog_source': args.catalog_source, 'crop': not args.no_crop})
    print('РЕЗУЛЬТАТ:', json.dumps(m, ensure_ascii=False))
    out = EVAL_REPORTS / f'encab_{tag}.json'
    out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding='utf-8')
    print('отчёт ->', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())

