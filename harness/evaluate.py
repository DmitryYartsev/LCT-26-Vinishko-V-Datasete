# -*- coding: utf-8 -*-
"""Прогонщик метрик: queryset.jsonl + индекс -> recall@1/@5, margin, winery@1.

Конфиг кропа/модели/индекса берётся из ENV (как в сервисе):
  CROP_ENABLED=0/1  — какой индекс (catalog / catalog_crop) и кропать ли запрос
  SIGLIP_MODEL      — энкодер

Запуск:
  CROP_ENABLED=0 uv run python evaluate.py --queryset querysets/scrape.jsonl --tag base
  CROP_ENABLED=1 uv run python evaluate.py --queryset querysets/scrape.jsonl --tag crop
"""
import os, sys, json, argparse, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "service"))
from encoder import get_encoder                       # noqa
from crop import maybe_crop, CROP_ENABLED             # noqa
from search import CatalogIndex                       # noqa

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def load_queryset(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queryset", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    enc = get_encoder()
    idx = CatalogIndex()
    winery_of = {s: idx.card(s).get("winery", "") for s in set(idx.slugs)}

    qs = load_queryset(args.queryset)
    incat = [q for q in qs if q.get("in_catalog", True)]
    print(f"queryset: {len(qs)} (in_catalog: {len(incat)}) | CROP_ENABLED={CROP_ENABLED} | index={idx.emb.shape}")

    hit1 = hit5 = win1 = 0
    top1s, margins = [], []
    per_wine = {}   # slug -> [hit1...]
    t0 = time.time()
    for i, q in enumerate(incat):
        try:
            img = Image.open(q["image_path"]).convert("RGB")
        except Exception:
            continue
        img, _ = maybe_crop(img)
        v = enc.embed([img])[0]
        res = idx.search(v, k=max(args.k, 5))
        slugs = [r["slug"] for r in res]
        true = q["true_slug"]
        h1 = int(slugs[0] == true)
        h5 = int(true in slugs[:5])
        hit1 += h1; hit5 += h5
        win1 += int(winery_of.get(slugs[0], "") == winery_of.get(true, "?") and winery_of.get(true, "") != "")
        top1s.append(res[0]["score"]); margins.append(res[0]["score"] - (res[1]["score"] if len(res) > 1 else 0))
        per_wine.setdefault(true, []).append(h1)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(incat)}...")
    n = len(incat)
    dt = time.time() - t0
    rep = {
        "tag": args.tag, "queryset": args.queryset, "crop": CROP_ENABLED,
        "model": idx.meta.get("model"), "n_queries": n,
        "recall@1": round(hit1 / n, 4), "recall@5": round(hit5 / n, 4),
        "winery@1": round(win1 / n, 4),
        "mean_top1_score": round(float(np.mean(top1s)), 4),
        "mean_margin": round(float(np.mean(margins)), 4),
        "sec_per_query": round(dt / max(n, 1), 3),
    }
    out = REPORTS / f"{args.tag}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== ОТЧЁТ [%s] ===" % args.tag)
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print("Сохранено:", out)


if __name__ == "__main__":
    main()
