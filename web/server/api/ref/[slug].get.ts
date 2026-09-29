// Эталонное фото вина из каталога (отдаёт ml из filtered/).
export default defineEventHandler(async (event) => {
  const slug = getRouterParam(event, "slug");
  const h = Number(getQuery(event).h) || 0;       // превью по высоте
  const res = await fetch(`${useRuntimeConfig().mlUrl}/ref/${slug}${h ? `?h=${h}` : ""}`);
  if (!res.ok) throw createError({ statusCode: 404 });
  setHeader(event, "content-type", res.headers.get("content-type") || "image/jpeg");
  setHeader(event, "cache-control", "public, max-age=86400");
  return Buffer.from(await res.arrayBuffer());
});
