// Диалог с сомелье: фронт шлёт всю переписку + профиль (stateless), ответ — {reply, profile, picks}.
export default defineEventHandler(async (event) => {
  const body = await readBody(event);
  return await $fetch(`${useRuntimeConfig().sommelierUrl}/v1/chat`, { method: "POST", body });
});
