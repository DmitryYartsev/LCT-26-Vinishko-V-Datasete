// Размер каталога для экрана поиска («сверяем с N этикетками»).
export default defineEventHandler(async () => {
  const h: any = await $fetch(`${useRuntimeConfig().mlUrl}/health`).catch(() => ({}));
  return { wines: h.wines || null };
});
