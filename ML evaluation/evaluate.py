# -*- coding: utf-8 -*-
"""Прогонщик метрик через HTTP к сервису: queryset.jsonl -> recall@1/@5, winery@1, margin.

Сервис должен быть поднят (docker compose up / uvicorn). Модель/кроп/индекс — конфиг сервиса.

  uv run python evaluate.py --queryset querysets/scrape.jsonl --tag scrape
  EVAL_URL=http://localhost:8080 uv run python evaluate.py --queryset ... --tag ...
"""
import os, sys, json, argparse, time
from pathlib import Path
import numpy as np
import requests

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
URL = os.environ.get("EVAL_URL", "http://localhost:8080").rstrip("/")
REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def load_queryset(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queryset", required=True)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    qs = [q for q in load_queryset(args.queryset) if q.get("in_catalog", True)]
    print(f"queryset: {len(qs)} (in_catalog) | {URL}")
    hit1 = hit5 = win1 = 0
    top1s, margins = [], []
    t0 = time.time()
    for i, q in enumerate(qs):
        try:
            with open(q["image_path"], "rb") as f:
                r = requests.post(f"{URL}/v1/search", files={"image": f}, timeout=30).json()
        except Exception:
            continue
        res = r.get("results", [])
        slugs = [x["slug"] for x in res]
        true = q["true_slug"]
        hit1 += int(slugs[:1] == [true])
        hit5 += int(true in slugs[:5])
        w_true = next((x["card"].get("winery") for x in res if x["slug"] == true), None)
        win1 += int(bool(slugs) and res[0]["card"].get("winery") and res[0]["card"].get("winery") == w_true)
        top1s.append(r["confidence"]["top1_score"]); margins.append(r["confidence"]["margin"])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(qs)}...")
    n = len(qs)
    rep = {"tag": args.tag, "queryset": args.queryset, "n_queries": n,
           "recall@1": round(hit1 / n, 4), "recall@5": round(hit5 / n, 4),
           "winery@1": round(win1 / n, 4),
           "mean_top1_score": round(float(np.mean(top1s)), 4) if top1s else 0,
           "mean_margin": round(float(np.mean(margins)), 4) if margins else 0,
           "sec_per_query": round((time.time() - t0) / max(n, 1), 3)}
    out = REPORTS / f"{args.tag}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== ОТЧЁТ [%s] ===" % args.tag)
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print("Сохранено:", out)


if __name__ == "__main__":
    main()
