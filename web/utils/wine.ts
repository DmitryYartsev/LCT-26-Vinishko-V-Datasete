// Типы ответов BFF и мелкие хелперы отображения.

export type Wine = {
  slug: string; title?: string; manufacturer?: string | null; category?: string | null;
  color?: string | null; sweetness?: string | null; sparkling?: boolean | null; sugar?: string | null;
  tagline?: string; alcohol?: number | null; temperature?: string | null; grapes?: string[];
  region?: string | null; dishes?: string[]; rating?: number | null; description?: string | null;
  url?: string | null; image?: string | null; in_catalog?: boolean;
};
export type Check = { key: string; label: string; value: string | null; ok: boolean };
export type Verdict = "ok" | "part" | "bad";
export type Match = {
  score: number | null; reasons: { ok: boolean; text: string }[]; checks: Check[];
  verdict: Verdict | null; verdict_text: string | null;
};
export type Item = Match & { wine: Wine };                        // /api/wines, похожие
export type Pick = Wine & { score: number; reasons: { ok: boolean; text: string }[] };   // подборки
export type SimilarItem = Item & { sim: number };

export function wineImg(w: Pick | Wine | undefined | null): string | undefined {
  if (!w) return;
  // 900px по высоте хватает и для крупного фото карточки на ретине
  if (w.in_catalog !== false) return `/api/ref/${encodeURIComponent(w.slug)}?h=900`;
  // CDN сайта умеет ресайз по URL: 1920 -> 600, чтобы не тянуть лишнее на телефон
  return w.image?.replace("/1920/1920/", "/600/600/") || undefined;
}

export const fmtRating = (r?: number | null) => (r == null ? null : r.toFixed(2));

export function plural(n: number, one: string, few: string, many: string) {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}

export function wineName(w?: Wine | null) {
  return (w?.title || w?.slug || "").trim();
}

export const VERDICT_STYLE: Record<Verdict, { label: string; color: string; bg: string }> = {
  ok: { label: "Подходит под вашу цель", color: "var(--ok)", bg: "#F1F5EE" },
  part: { label: "Подходит частично", color: "var(--part)", bg: "#FAF4EA" },
  bad: { label: "Может не подойти под вашу цель", color: "#7A6258", bg: "#F7F2EC" },
};

export type Ranked = Item & { grp: "ok" | "good" | "bad" | "all"; reason: string; good: boolean };

/** Сканы за сессию: сначала подходящие под цель, потом хорошие, но не под цель, потом слабые. */
export function rankScans(items: Item[], hasGoal: boolean): Ranked[] {
  const rows: Ranked[] = items.map((it) => {
    const r = it.wine.rating;
    if (!hasGoal || !it.verdict)
      return { ...it, grp: "all", reason: it.wine.tagline || "", good: false };
    const ratingLow = it.checks.some((c) => c.key === "rating" && !c.ok);
    const fails = it.checks.filter((c) => !c.ok && c.key !== "rating" && c.value)
      .map((c) => (c.key === "dishes" ? "не к вашему блюду" : c.value!.toLowerCase()));
    if (it.verdict === "ok") return { ...it, grp: "ok", reason: "Подходит под вашу цель", good: true };
    if (!ratingLow)
      return { ...it, grp: "good", good: false,
        reason: (r != null ? "Хорошее, но сомелье не советует" : "Сомелье не советует") + " под вашу цель"
          + (fails.length ? ": " + fails.join(", ") : "") };
    return { ...it, grp: "bad", reason: "Сомелье не советует: низкий народный рейтинг", good: false };
  });
  const order = { ok: 0, good: 1, bad: 2, all: 0 };
  return rows.sort((a, b) => order[a.grp] - order[b.grp] || (b.score ?? 0) - (a.score ?? 0)
    || (b.wine.rating ?? 0) - (a.wine.rating ?? 0));
}
