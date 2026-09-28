# -*- coding: utf-8 -*-
"""Прогонщик метрик через HTTP к сервису: queryset.jsonl -> recall@1/@5/@10, winery@1, реранк.

Сервис должен быть поднят (docker compose up / uvicorn). Модель/кроп/реранк — конфиг СЕРВИСА
(env сервиса, не этого скрипта); в отчёт он записывается из /health.

  uv run python evaluate.py --queryset querysets/dataset.jsonl --tag labeled_rerank
  uv run python evaluate.py --queryset querysets/dataset.jsonl --tag quick --limit 50
  EVAL_URL=http://127.0.0.1:8080 uv run python evaluate.py ...

Сервис просим отдать top-max(--k, RERANK_TOP_K): в него попадают все кандидаты реранкера,
поэтому порядок ДО реранка восстанавливается сортировкой по визуальному `score` — recall@K
до и после реранка считаются в одном прогоне, отдельный прогон без реранка не нужен.
recall@K_visual при K=RERANK_TOP_K — потолок реранкера (вне top-K он верное вино не найдёт).

Выход: reports/<tag>.json (сводка) + reports/<tag>.csv (построчно: true/pred/visual_top1/scores).
"""
import os, sys, json, argparse, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
URL = os.environ.get("EVAL_URL", "http://127.0.0.1:8080").rstrip("/")
REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def load_queryset(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queryset", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--k", type=int, default=10, help="глубина выдачи для recall@K (деф 10)")
    ap.add_argument("--limit", type=int, default=0, help="случайная подвыборка N query (экономит токены реранкера)")
    ap.add_argument("--seed", type=int, default=0, help="сид подвыборки — одинаковый набор для сравнения конфигов")
    args = ap.parse_args()

    http = requests.Session()                      # keep-alive: одно соединение на весь прогон
    health = http.get(f"{URL}/health", timeout=10).json()
    qs = [q for q in load_queryset(args.queryset) if q.get("in_catalog", True)]
    if args.limit and args.limit < len(qs):
        qs = random.Random(args.seed).sample(qs, args.limit)
    rr_cfg = health.get("rerank") or {}
    rr_on = rr_cfg.get("backend", "none") != "none"
    depth = max(args.k, rr_cfg.get("top_k", 0) if rr_on else 0)
    ks = sorted({1, 5, args.k})
    print(f"queryset: {len(qs)} (in_catalog) | {URL} | model={health.get('model')} crop={health.get('crop')} "
          f"rerank={rr_cfg.get('backend', 'none')} | depth={depth}")

    rows, errors, streak = [], 0, 0
    for i, q in enumerate(qs):
        t = time.perf_counter()
        try:
            with open(q["image_path"], "rb") as f:
                resp = http.post(f"{URL}/v1/search", params={"k": depth}, files={"image": f}, timeout=60)
            resp.raise_for_status()
            r = resp.json()
        except Exception as e:
            errors += 1; streak += 1
            print(f"  ! {q['query_id']}: {str(e)[:120]}")
            if streak >= 5:
                sys.exit("5 ошибок подряд — сервис упал/перезапускается? Прогон остановлен.")
            continue
        streak = 0
        res = r.get("results", [])
        if i == 0 and len(res) < depth:
            print(f"  ВНИМАНИЕ: сервис вернул {len(res)} результатов вместо {depth} — старая версия без `?k=`? "
                  f"Перезапусти сервис, иначе recall@{args.k} посчитан по top-{len(res)}")
        vis = sorted(res, key=lambda x: x["score"], reverse=True)   # порядок до реранка
        rr = r.get("rerank") or {}
        true = q["true_slug"]
        rank = lambda lst: next((i + 1 for i, x in enumerate(lst) if x["slug"] == true), None)
        winery = lambda x: x["card"].get("winery")
        w_true = next((winery(x) for x in res if x["slug"] == true), None)
        rows.append({"query_id": q["query_id"], "true_slug": true,
                     "pred": res[0]["slug"] if res else None, "visual_top1": vis[0]["slug"] if vis else None,
                     "rank": rank(res), "rank_visual": rank(vis),        # позиция верного вина (1-based), None — нет в top
                     "winery1": bool(res) and bool(winery(res[0])) and winery(res[0]) == w_true,
                     "top1_score": r["confidence"]["top1_score"], "margin": r["confidence"]["margin"],
                     "rerank_applied": bool(rr.get("applied")), "rerank_error": rr.get("error"),
                     "rerank_ms": rr.get("ms"), "latency_s": time.perf_counter() - t,
                     "image_path": q["image_path"]})
        if (i + 1) % 10 == 0:
            done = pd.DataFrame(rows)
            print(f"  {i+1}/{len(qs)}  recall@1={(done['rank'] == 1).mean():.3f}")

    if not rows:
        sys.exit(f"ни одного успешного запроса (ошибок {errors}) — сервис поднят? пути в queryset живые?")
    df = pd.DataFrame(rows)
    recall = lambda col, k: round(float((df[col].fillna(10**9) <= k).mean()), 4)
    n = len(df)
    rep = {"tag": args.tag, "queryset": args.queryset, "n_queries": n, "errors": errors,
           "model": health.get("model"), "crop": health.get("crop"), "rerank_config": rr_cfg,
           **{f"recall@{k}": recall("rank", k) for k in ks},
           "winery@1": round(df.winery1.mean(), 4),
           "mean_top1_score": round(df.top1_score.mean(), 4), "mean_margin": round(df.margin.mean(), 4),
           "latency_mean_s": round(df.latency_s.mean(), 3),
           "latency_p95_s": round(float(np.percentile(df.latency_s, 95)), 3)}
    if rr_on:
        applied = df[df.rerank_applied]
        hit1, hit1_vis = applied["rank"] == 1, applied.rank_visual == 1
        rep["rerank"] = {
            "recall_visual": {f"@{k}": recall("rank_visual", k) for k in ks},   # до реранка
            "recall_final": {f"@{k}": recall("rank", k) for k in ks},           # после
            "applied_share": round(float(df.rerank_applied.mean()), 4),
            "errors": int(df.rerank_error.notna().sum()),
            "changed_top1": int((applied.pred != applied.visual_top1).sum()),
            "fixed_top1": int((~hit1_vis & hit1).sum()),     # было неверно -> стало верно
            "broken_top1": int((hit1_vis & ~hit1).sum()),    # было верно -> стало неверно
            "ms_mean": round(float(applied.rerank_ms.mean()), 1) if len(applied) else None}
    out = REPORTS / f"{args.tag}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(REPORTS / f"{args.tag}.csv", index=False, encoding="utf-8")
    print("\n=== ОТЧЁТ [%s] ===" % args.tag)
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print("Сохранено:", out, "+ .csv")


if __name__ == "__main__":
    main()
