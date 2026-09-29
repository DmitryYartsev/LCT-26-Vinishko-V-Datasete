// Пакетно: карточки + соответствие цели (сканы за сессию). {slugs, profile} -> {items}
export default defineEventHandler(async (event) => {
  const { slugs, profile } = await readBody(event);
  if (!Array.isArray(slugs) || !slugs.length) return { items: [] };
  return await $fetch(`${useRuntimeConfig().sommelierUrl}/v1/match_many`, {
    method: "POST", body: { slugs, profile: profile || {} },
  });
});
