export default defineEventHandler(async (event) => {
  const parts = await readMultipartFormData(event);
  const file = parts?.find((p) => p.name === "image");
  if (!file) throw createError({ statusCode: 400, statusMessage: "no image" });
  const fd = new FormData();
  fd.append("image", new Blob([file.data], { type: file.type || "image/jpeg" }), file.filename || "photo.jpg");
  const ml = useRuntimeConfig().mlUrl;
  return await $fetch(`${ml}/v1/search`, { method: "POST", body: fd });
});
