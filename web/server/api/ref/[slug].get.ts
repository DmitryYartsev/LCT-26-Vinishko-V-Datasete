export default defineEventHandler(async (event) => {
  const slug = getRouterParam(event, "slug");
  const ml = useRuntimeConfig().mlUrl;
  const res = await fetch(`${ml}/ref/${slug}`);
  if (!res.ok) throw createError({ statusCode: 404 });
  setHeader(event, "content-type", res.headers.get("content-type") || "image/jpeg");
  return Buffer.from(await res.arrayBuffer());
});
