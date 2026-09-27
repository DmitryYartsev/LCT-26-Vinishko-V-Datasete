export default defineNuxtConfig({
  // SPA: цель сомелье и сканы живут в браузере (localStorage/sessionStorage), SSR тут ничего не даёт.
  ssr: false,
  devServer: { host: "0.0.0.0", port: 3000 },
  css: ["~/assets/css/main.css"],
  runtimeConfig: {
    mlUrl: process.env.ML_URL || "http://localhost:8080",
    sommelierUrl: process.env.SOMMELIER_URL || "http://localhost:8090",
  },
  app: {
    head: {
      htmlAttrs: { lang: "ru" },
      title: "Своё Вино — сканер российских вин",
      meta: [
        { name: "viewport", content: "width=device-width, initial-scale=1, viewport-fit=cover" },
        { name: "theme-color", content: "#FCFAF6" },
      ],
      link: [
        { rel: "icon", type: "image/png", href: "/img/logo-mark.png" },
        { rel: "preconnect", href: "https://fonts.googleapis.com" },
        { rel: "preconnect", href: "https://fonts.gstatic.com", crossorigin: "" },
        { rel: "stylesheet", href: "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500&family=Noto+Sans:wght@400;500;600&display=swap" },
      ],
    },
  },
});
