// Голос -> текст: проксируем запись из браузера (webm/mp4) в сомелье.
export default defineEventHandler(async (event) => {
  const parts = await readMultipartFormData(event);
  const file = parts?.find((p) => p.name === "audio");
  if (!file) throw createError({ statusCode: 400, statusMessage: "no audio" });
  const fd = new FormData();
  fd.append("audio", new Blob([file.data], { type: file.type || "audio/webm" }), file.filename || "voice.webm");
  return await $fetch(`${useRuntimeConfig().sommelierUrl}/v1/stt`, { method: "POST", body: fd });
});
