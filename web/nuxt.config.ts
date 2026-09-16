export default defineNuxtConfig({
  ssr: true,
  devServer: { host: "0.0.0.0", port: 3000 },
  runtimeConfig: {
    mlUrl: process.env.ML_URL || "http://localhost:8080",
  },
  app: {
    head: {
      title: "Своё Вино — сканер",
      meta: [{ name: "viewport", content: "width=device-width, initial-scale=1, maximum-scale=1" }],
      link: [
        { rel: "preconnect", href: "https://fonts.googleapis.com" },
        { rel: "stylesheet", href: "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap" },
      ],
    },
  },
});
