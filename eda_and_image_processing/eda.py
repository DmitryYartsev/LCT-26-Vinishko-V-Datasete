# -*- coding: utf-8 -*-
"""EDA каталога «Своё вино» + сопоставление фото с дампом Strapi."""
import re
import sys
import json
from pathlib import Path
import pandas as pd

ROOT = Path("D:/_hack/lct_26")
CSV = ROOT / "strapi_output0709.csv"
FILELIST = ROOT / "solution/eda/archive_filelist.txt"
OUT = ROOT / "solution/eda"

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)


def norm(s):
    return str(s).strip()


def strapi_base(name):
    """Нормализация имени файла в стиле Strapi: без расширения, спецсимволы->_, lower."""
    stem = re.sub(r"\.(webp|jpg|jpeg|png|JPG|JPEG|PNG|WEBP)$", "", str(name).strip())
    b = stem.lower()
    b = re.sub(r"[^a-z0-9а-яё]+", "_", b)  # всё несловесное -> _
    b = re.sub(r"_+", "_", b).strip("_")
    return b


def main():
    df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].map(norm)

    print("=" * 70)
    print("СТРУКТУРА CSV")
    print("=" * 70)
    print("Колонки:", list(df.columns))
    print("Всего строк:", len(df))

    # Ключ позиции = Slug
    slug_col = "Slug"
    print("\nУникальных Slug:", df[slug_col].nunique())
    print("Пустых Slug:", (df[slug_col] == "").sum())

    # Дубли строк
    dup_full = df.duplicated().sum()
    print("Полных дублей строк:", dup_full)

    # Схлопнем до уникальных позиций по slug
    uniq = df.drop_duplicates(subset=[slug_col]).reset_index(drop=True)
    print("Уникальных позиций (по slug):", len(uniq))

    print("\n" + "=" * 70)
    print("РАСПРЕДЕЛЕНИЯ (по уникальным позициям)")
    print("=" * 70)
    for col in ["Категория", "Цвет", "Регион", "Винодельня"]:
        if col in uniq.columns:
            vc = uniq[col].replace("", "<пусто>").value_counts()
            print(f"\n--- {col}: {uniq[col].replace('', pd.NA).nunique()} уникальных ---")
            print(vc.head(15).to_string())

    # Сорта (мультизначные через запятую)
    if "Сорт винограда" in uniq.columns:
        grapes = {}
        for v in uniq["Сорт винограда"]:
            for g in re.split(r"[;,/]", v):
                g = g.strip()
                if g:
                    grapes[g] = grapes.get(g, 0) + 1
        gs = pd.Series(grapes).sort_values(ascending=False)
        print(f"\n--- Сорт винограда: {len(gs)} уникальных (после split) ---")
        print(gs.head(20).to_string())

    # Пропуски по полям
    print("\n" + "=" * 70)
    print("ПОЛНОТА ПОЛЕЙ (по уникальным позициям)")
    print("=" * 70)
    for col in uniq.columns:
        empty = (uniq[col] == "").sum()
        print(f"{col:20s}: заполнено {len(uniq)-empty:4d} / {len(uniq)}  (пусто {empty})")

    # Длина описания
    if "Описание" in uniq.columns:
        L = uniq["Описание"].str.len()
        print("\nДлина описания: min=%d medians=%d max=%d, пустых=%d" % (
            L.min(), int(L.median()), L.max(), (L == 0).sum()))

    # ---- NEAR-DUPLICATES: одинаковая этикетка, разный год/сезон/категория ----
    print("\n" + "=" * 70)
    print("NEAR-DUPLICATES (главный источник ошибок)")
    print("=" * 70)
    # 1) Одно фото на несколько slug
    photo_col = "Название фото"
    if photo_col in uniq.columns:
        shared_photo = uniq[uniq[photo_col] != ""].groupby(photo_col)[slug_col].nunique()
        multi = shared_photo[shared_photo > 1]
        print("Фото, привязанных к >1 позиции:", len(multi),
              "(покрывают %d позиций)" % uniq[uniq[photo_col].isin(multi.index)].shape[0])

    # 2) Базовое имя без года: убираем год из названия и группируем
    def base_name(n):
        b = n.lower()
        b = re.sub(r"\b20\d{2}\b", "", b)  # год
        b = re.sub(r"[^a-zа-яё0-9]+", " ", b).strip()
        return b
    uniq["_base"] = uniq["Название вина"].map(base_name)
    series_groups = uniq.groupby("_base")[slug_col].nunique().sort_values(ascending=False)
    families = series_groups[series_groups > 1]
    print("Серий (одинаковое имя без года) с >1 позицией:", len(families))
    print("Позиций внутри таких серий:",
          uniq[uniq["_base"].isin(families.index)].shape[0])
    print("\nТоп-10 крупнейших серий:")
    for base, n in families.head(10).items():
        print(f"  [{n}] {base[:60]}")

    # ---- Сопоставление фото с архивом Strapi ----
    print("\n" + "=" * 70)
    print("СОПОСТАВЛЕНИЕ ФОТО С ДАМПОМ STRAPI (uploads/)")
    print("=" * 70)
    files = []
    if FILELIST.exists():
        for line in FILELIST.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip().replace("\\", "/")
            if "/uploads/" in line:
                files.append(line.split("/uploads/")[-1])
    print("Файлов в uploads/:", len(files))

    # индекс: базовое имя strapi -> список файлов; и префиксный (base без хеша)
    def file_stem_variants(fname):
        stem = re.sub(r"\.(webp|jpg|jpeg|png)$", "", fname, flags=re.I)
        # strapi добавляет _<10hex>; уберём хвостовой хеш
        no_hash = re.sub(r"_[0-9a-f]{6,}$", "", stem)
        return stem.lower(), no_hash.lower()

    by_full = {}
    by_nohash = {}
    for f in files:
        full, nh = file_stem_variants(f)
        by_full.setdefault(full, []).append(f)
        by_nohash.setdefault(nh, []).append(f)

    matched, unmatched = 0, []
    match_rows = []
    for _, r in uniq.iterrows():
        photo = r[photo_col]
        if not photo:
            unmatched.append((r[slug_col], "<нет имени фото>"))
            continue
        base = strapi_base(photo)
        cand = by_nohash.get(base) or by_full.get(base)
        if cand:
            matched += 1
            match_rows.append({"slug": r[slug_col], "photo": photo, "file": cand[0]})
        else:
            unmatched.append((r[slug_col], photo))

    print("Сопоставлено (по нормализ. имени): %d / %d" % (matched, len(uniq)))
    print("Не сопоставлено:", len(unmatched))
    print("\nПримеры несопоставленных (slug | Название фото):")
    for s, p in unmatched[:15]:
        print(f"  {s[:40]:40s} | {p[:40]}")

    # сохраним маппинг
    mdf = pd.DataFrame(match_rows)
    mdf.to_csv(OUT / "photo_map.csv", index=False, encoding="utf-8")
    uniq.drop(columns=["_base"]).to_csv(OUT / "catalog_unique.csv", index=False, encoding="utf-8")
    print("\nСохранено: photo_map.csv (%d), catalog_unique.csv (%d)" % (len(mdf), len(uniq)))

    summary = {
        "rows_csv": len(df),
        "unique_slugs": int(df[slug_col].nunique()),
        "unique_positions": len(uniq),
        "photo_matched": matched,
        "photo_unmatched": len(unmatched),
        "uploads_files": len(files),
        "series_families": int(len(families)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSUMMARY:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
