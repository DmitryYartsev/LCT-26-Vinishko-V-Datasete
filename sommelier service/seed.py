# -*- coding: utf-8 -*-
"""Warm-up сомелье: карточки сайта (wines_parsed.jsonl) + каталог -> wine_profiles.

Вход (лежат рядом, в git): wines_parsed.jsonl — карточки сайта; slug_map.csv — маппинг
slug каталога -> slug сайта (выжимка из found_in_catalog_corrected.csv парсера).

Правила склейки:
  * одна строка на каждый slug каталога (их сканирует `ml`) — атрибуты берутся из карточки
    сайта по маппингу slug -> matched_slug, если он надёжный (точный slug/site или score >= 0.7);
    иначе — только то, что есть в каталоге (цвет + сладость из названия);
  * плюс вина сайта, которых нет в каталоге (для рекомендаций), in_catalog = false.

  uv run python seed.py            # залить, если таблица пуста
  uv run python seed.py --force    # перезалить
"""
import sys, re, json, argparse
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from paths import WINES_PARSED, WINES_SLUG_MAP, CATALOG_CSV
import store

MAP_MIN_SCORE = 0.7
SPARKLING_RE = re.compile(r"игрист|спуманте|spumante|брют|bryut|brut|пет[- ]?нат|pet[- ]?nat|шампан|"
                          r"жемчуж|col fondo|просекко|prosecco|asti|асти", re.I)


def _sweet_from_text(text: str):
    t = (text or "").lower()
    for key, val in (("экстра брют", "сухое"), ("брют", "сухое"), ("полусух", "полусухое"),
                     ("полусладк", "полусладкое"), ("сладк", "сладкое"), ("сух", "сухое")):
        if key in t:
            return val
    return None


def _from_site(r: dict) -> dict:
    cat = (r.get("category") or "").strip()                 # «Белое экстра брют»
    color, _, rest = cat.partition(" ")
    rest = rest.strip().lower()
    return {
        "site_slug": r["slug"], "title": (r.get("title") or "").strip(),
        "manufacturer": r.get("manufacturer"), "category": cat,
        "color": color.lower() or None,
        "sweetness": _sweet_from_text(rest),
        "sparkling": rest in ("брют", "экстра брют") or bool(SPARKLING_RE.search(r["slug"] + " " + (r.get("title") or ""))),
        "alcohol": r.get("alcohol"),
        "temperature": (r.get("temperature") or "").replace("–", "-") or None,
        "grapes": r.get("grapes") or [], "region": r.get("region"),
        "dishes": r.get("dishes") or [], "rating": r.get("public_rating"),
        "description": (r.get("description") or "").strip(), "url": r.get("url"), "image": r.get("image_main"),
    }


def _from_catalog(c: dict) -> dict:
    """Фолбэк без карточки сайта: цвет из категории каталога, сладость/игристость — из названия."""
    cat = (c.get("category") or "").strip()
    text = f"{c.get('name', '')} {c['slug']}"
    return {
        "site_slug": None, "title": (c.get("name") or "").strip(), "manufacturer": c.get("winery") or None,
        "category": cat, "color": cat.lower() if cat.lower() in ("красное", "белое", "розовое") else None,
        "sweetness": _sweet_from_text(text), "sparkling": cat == "Игристое" or bool(SPARKLING_RE.search(text)),
        "alcohol": None, "temperature": None,
        "grapes": [g.strip() for g in re.split(r"[,;]", c.get("grape") or "") if g.strip()],
        "region": c.get("region") or None, "dishes": [], "rating": None,
        "description": (c.get("description") or "").strip(), "url": None, "image": None,
    }


def build_rows() -> list[dict]:
    site, cat2site = {}, {}
    catalog = pd.read_csv(CATALOG_CSV, dtype=str, keep_default_na=False).to_dict("records")
    if WINES_PARSED.exists() and WINES_SLUG_MAP.exists():
        for line in WINES_PARSED.open(encoding="utf-8"):
            r = json.loads(line)
            if "category" in r:                               # записи с ошибкой парсинга пропускаем
                site[r["slug"]] = r
        m = pd.read_csv(WINES_SLUG_MAP, dtype=str, keep_default_na=False)
        m["match_score"] = pd.to_numeric(m["match_score"], errors="coerce").fillna(0)
        trusted = m[(m["site_slug"] != "") &
                    (m["match_method"].isin(["slug", "site"]) | (m["match_score"] >= MAP_MIN_SCORE))]
        cat2site = dict(zip(trusted["slug"], trusted["site_slug"]))
    else:
        print(f"[seed] нет {WINES_PARSED.name} — только атрибуты каталога (без блюд/крепости)")

    rows, used_site = [], set()
    for c in catalog:
        s = cat2site.get(c["slug"])
        if s in site:
            row = _from_site(site[s])
            row["title"] = (c.get("name") or "").strip() or row["title"]   # название — как в карточке сканера
            used_site.add(s)
        else:
            row = _from_catalog(c)
        rows.append({"slug": c["slug"], "in_catalog": True, **row})
    cat_slugs = {c["slug"] for c in catalog}
    for s, r in site.items():
        if s not in cat_slugs and s not in used_site:
            rows.append({"slug": s, "in_catalog": False, **_from_site(r)})
    return rows


def warmup(force: bool = False):
    conn = store.connect()
    n = store.count(conn)
    if n and not force:
        print(f"[seed] wine_profiles уже заполнена ({n}), пропускаю")
        return
    rows = build_rows()
    store.replace_all(conn, rows)
    enriched = sum(1 for r in rows if r["in_catalog"] and r["site_slug"])
    print(f"[seed] залито {len(rows)} вин: каталог {sum(r['in_catalog'] for r in rows)} "
          f"(с атрибутами сайта {enriched}), только сайт {sum(not r['in_catalog'] for r in rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    warmup(ap.parse_args().force)
