// Карточка вина: атрибуты (сомелье, фолбэк — карточка ml) + соответствие цели пользователя
// + что показать ниже: «лучше под вашу цель» (если не подходит) или аналоги других виноделен.
export default defineEventHandler(async (event) => {
  const slug = getRouterParam(event, "slug")!;
  const { profile } = (await readBody(event).catch(() => null)) || {};
  const { mlUrl, sommelierUrl } = useRuntimeConfig();
  const som = (path: string, body?: object) =>
    $fetch<any>(`${sommelierUrl}${path}`, body ? { method: "POST", body } : {});

  let wine: any = await som(`/v1/wine/${encodeURIComponent(slug)}`).catch(() => null);
  if (!wine) {
    const c: any = await $fetch(`${mlUrl}/wine/${encodeURIComponent(slug)}`).catch(() => null);
    if (!c) throw createError({ statusCode: 404, statusMessage: "wine not found" });
    wine = { slug, title: c.name, manufacturer: c.winery, region: c.region, color: c.color?.toLowerCase(),
      grapes: c.grape ? [c.grape] : [], description: c.description, category: c.category,
      tagline: c.category, in_catalog: true };
  }

  const match = profile ? await som("/v1/match", { slug, profile }).catch(() => null) : null;
  let alts: any[] = [], analogs: any[] = [];
  if (match?.verdict && match.verdict !== "ok")
    alts = await som("/v1/recommend", { profile, k: 4, exclude: [slug] }).then((r) => r.picks).catch(() => []);
  else
    analogs = await som("/v1/analogs", { slug, k: 4 }).then((r) => r.picks).catch(() => []);

  if (match) delete match.wine;
  return { wine, match, alts, analogs };
});
