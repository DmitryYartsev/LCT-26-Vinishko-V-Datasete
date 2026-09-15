# -*- coding: utf-8 -*-
"""Этап 1: сырой скрейп фото вин из веба «как есть» (без фильтрации).

Движок: Yandex Images через Playwright (реальный Chromium — отдаёт честные РУ-результаты
без decoy, в отличие от HTTP-скрейпа Bing). Чистка/дедуп/верификация — ОТДЕЛЬНО (clean_scraped.py).

Примеры:
  python scrape_images.py --limit-wines 5 --per-wine 8
  python scrape_images.py --all --per-wine 6 --sleep 3
  python scrape_images.py --slugs massandra-muskatel-belyy-belye-sorta-vin,agora-pino-nuar

Прокси (для массового прогона, если начнёт ловить капчу):
  python scrape_images.py --all --proxy http://user:pass@host:port
"""
import re, sys, time, json, argparse, random
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
import httpx
import pandas as pd
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CATALOG_CSV, SCRAPED

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
REFMAP = CATALOG_CSV               # slug, name, winery (выход process_images.py)
OUTDIR = SCRAPED
OUTDIR.mkdir(parents=True, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CONTENT_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
               "image/jfif": ".jpg", "image/bmp": ".bmp"}


def is_captcha(page):
    u = page.url.lower()
    if "showcaptcha" in u or "/checkcaptcha" in u:
        return True
    t = (page.title() or "").lower()
    return t.startswith("ой") or "captcha" in t


def yandex_image_urls(page, query, n):
    """Вернуть до n URL полноразмерных картинок из Яндекс.Картинок."""
    url = f"https://yandex.ru/images/search?text={quote(query)}&isize=large"
    try:
        page.goto(url, timeout=35000, wait_until="domcontentloaded")
    except Exception as e:
        print("   [ya] goto err:", str(e)[:120]); return []
    page.wait_for_timeout(random.randint(1800, 2600))
    if is_captcha(page):
        return "CAPTCHA"
    urls, tries = [], 0
    while len(urls) < n * 2 and tries < 6:
        html = page.content()
        found = [unquote(h) for h in re.findall(r'img_url=([^&"]+)', html)]
        for u in found:
            if u.startswith("http") and u not in urls:
                urls.append(u)
        if len(urls) >= n * 2:
            break
        page.mouse.wheel(0, 5000)
        page.wait_for_timeout(random.randint(700, 1200))
        tries += 1
    return urls


def download(client, url, dst_noext, min_bytes=6000, max_bytes=15_000_000):
    try:
        r = client.get(url, timeout=25, follow_redirects=True,
                       headers={"User-Agent": UA, "Referer": "https://yandex.ru/"})
        if r.status_code != 200:
            return None
        ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
        ext = CONTENT_EXT.get(ct)
        if not ext or not (min_bytes <= len(r.content) <= max_bytes):
            return None
        dst = dst_noext.with_suffix(ext)
        dst.write_bytes(r.content)
        return dst, len(r.content), ct
    except Exception:
        return None


def build_query(row):
    name = str(row.get("Название вина") or row.get("name") or "").strip()
    winery = str(row.get("Винодельня") or row.get("winery") or "").strip()
    return re.sub(r"\s+", " ", f"{winery} {name}".strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-wines", type=int, default=5)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--slugs", default="", help="список slug через запятую")
    ap.add_argument("--per-wine", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=2.5, help="пауза между винами, сек")
    ap.add_argument("--proxy", default="", help="http://user:pass@host:port")
    ap.add_argument("--headful", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(REFMAP, dtype=str, keep_default_na=False)
    df = df.drop_duplicates(subset=["slug"]).reset_index(drop=True)
    if args.slugs:
        want = [s.strip() for s in args.slugs.split(",") if s.strip()]
        wines = df[df["slug"].isin(want)].reset_index(drop=True)
    else:
        wines = df.iloc[args.offset:]
        if not args.all:
            wines = wines.head(args.limit_wines)
    print(f"Вин к обработке: {len(wines)} | per-wine: {args.per_wine} | proxy: {'да' if args.proxy else 'нет'}")

    manifest_path = OUTDIR / "_manifest.csv"
    rows = pd.read_csv(manifest_path, dtype=str, keep_default_na=False).to_dict("records") if manifest_path.exists() else []
    done = {r["slug"] for r in rows}

    launch = {"headless": not args.headful}
    if args.proxy:
        launch["proxy"] = {"server": args.proxy}

    client = httpx.Client()
    with sync_playwright() as p:
        b = p.chromium.launch(**launch)
        ctx = b.new_context(locale="ru-RU", user_agent=UA, viewport={"width": 1366, "height": 900})
        page = ctx.new_page()
        try:
            for i, (_, w) in enumerate(wines.iterrows()):
                slug = w["slug"]
                if slug in done:
                    print(f"[{i}] skip: {slug}"); continue
                query = build_query(w)
                print(f"[{i}] {slug} :: '{query}'")
                urls = yandex_image_urls(page, query, args.per_wine)
                if urls == "CAPTCHA":
                    print("   !!! CAPTCHA от Яндекса — останавливаюсь. Нужны прокси (--proxy) или пауза.")
                    break
                wdir = OUTDIR / slug
                wdir.mkdir(exist_ok=True)
                saved = 0
                for u in urls:
                    if saved >= args.per_wine:
                        break
                    res = download(client, u, wdir / f"ya_{saved:02d}")
                    if not res:
                        continue
                    dst, nbytes, ct = res
                    rows.append({"slug": slug, "engine": "yandex", "query": query,
                                 "source_url": u, "path": str(dst.relative_to(OUTDIR)),
                                 "bytes": nbytes, "content_type": ct})
                    saved += 1
                    time.sleep(random.uniform(0.1, 0.3))
                print(f"    сохранено {saved}")
                pd.DataFrame(rows).to_csv(manifest_path, index=False, encoding="utf-8")
                time.sleep(random.uniform(args.sleep, args.sleep + 1.5))
        finally:
            b.close()
            client.close()

    mf = pd.DataFrame(rows)
    print(f"\nИТОГО в манифесте: {len(mf)} фото по {mf['slug'].nunique() if len(mf) else 0} винам")
    print("Манифест:", manifest_path)


if __name__ == "__main__":
    main()
