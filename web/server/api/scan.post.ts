// Скан: фото -> ml (поиск по каталогу). Если уверенного совпадения нет — сразу собираем экран
// «не найдено»: похожие по этикетке (top-5 ml) + аналоги по стилю из других виноделен (сомелье).
// Сомелье опционален: без него скан всё равно отвечает.
export default defineEventHandler(async (event) => {
  const parts = await readMultipartFormData(event);
  const file = parts?.find((p) => p.name === "image");
  if (!file) throw createError({ statusCode: 400, statusMessage: "no image" });
  let profile: object = {};
  try { profile = JSON.parse(parts?.find((p) => p.name === "profile")?.data.toString() || "{}"); } catch {}

  const { mlUrl, sommelierUrl } = useRuntimeConfig();
  const fd = new FormData();
  fd.append("image", new Blob([file.data], { type: file.type || "image/jpeg" }), file.filename || "photo.jpg");
  const res: any = await $fetch(`${mlUrl}/v1/search`, { method: "POST", body: fd });
  const results: { slug: string; score: number }[] = (res.results || []).map((r: any) => ({ slug: r.slug, score: r.score }));
  const out = {
    in_catalog: res.in_catalog, elapsed_ms: res.elapsed_ms, confidence: res.confidence,
    top1: results[0] || null, results,
  };
  if (res.in_catalog || !results.length) return out;

  const [similar, analogs] = await Promise.all([
    $fetch<any>(`${sommelierUrl}/v1/match_many`, { method: "POST", body: { slugs: results.map((r) => r.slug), profile } })
      .then((r) => r.items).catch(() => []),
    $fetch<any>(`${sommelierUrl}/v1/analogs`, { method: "POST", body: { slug: results[0].slug, k: 4 } })
      .then((r) => r.picks).catch(() => []),
  ]);
  const sim = Object.fromEntries(results.map((r) => [r.slug, Math.round(r.score * 100)]));
  const shown = new Set(results.map((r) => r.slug));
  return {
    ...out,
    similar: similar.map((it: any) => ({ ...it, sim: sim[it.wine.slug] })),
    analogs: analogs.filter((p: any) => !shown.has(p.slug)),
  };
});
