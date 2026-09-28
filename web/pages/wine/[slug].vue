<script setup lang="ts">
// Карточка вина: после скана, из подборки сомелье или по прямой ссылке.
import type { Wine, Match, Pick, Item } from "~/utils/wine";

const route = useRoute();
const slug = computed(() => String(route.params.slug));
const som = useSommelier();
const scanner = useScanner();

type WineRes = { wine: Wine & { description?: string }; match: Match | null; alts: Pick[]; analogs: Pick[] };
const data = ref<WineRes | null>(null);
const failed = ref(false);
const loading = ref(true);

let seq = 0;   // ответ на устаревший запрос (сменили цель/вино, пока грузилось) не должен затереть свежий
async function load() {
  const my = ++seq;
  loading.value = true; failed.value = false;
  try {
    const res = await $fetch<WineRes>(`/api/wine/${encodeURIComponent(slug.value)}`, {
      method: "POST", body: { profile: som.goalProfile.value },
    });
    if (my === seq) data.value = res;
  } catch {
    if (my === seq) failed.value = true;
  } finally {
    if (my === seq) loading.value = false;
  }
}
// цель поменялась в чате, пока открыта карточка, — пересчитываем соответствие
watch([slug, () => JSON.stringify(som.goalProfile.value)], load, { immediate: true });

const wine = computed(() => data.value?.wine);
const match = computed(() => (data.value?.match?.verdict ? data.value.match : null));
const vs = computed(() => (match.value ? VERDICT_STYLE[match.value.verdict!] : null));
const okCount = computed(() => match.value?.checks.filter((c) => c.ok).length ?? 0);

// это вино пришло из только что сделанного скана?
const fromScan = computed(() => scanner.last.value?.top1?.slug === slug.value && scanner.last.value?.in_catalog);
const debug = useRuntimeConfig().public.debugInfo;
const seconds = computed(() => scanner.last.value?.seconds.toFixed(1).replace(".", ","));
const conf = computed(() => scanner.last.value?.confidence);

const attrs = computed(() => {
  const w = wine.value;
  if (!w) return [];
  return [
    ["Цвет", w.color && w.color[0].toUpperCase() + w.color.slice(1)],
    ["Сахар", w.sugar],
    ["Крепость", w.alcohol != null ? `${String(w.alcohol).replace(".", ",")}%` : null],
    ["Сорт", w.grapes?.join(", ")],
    ["Регион", w.region],
    ["Подача", w.temperature ? `${w.temperature} °C` : null],
  ].filter(([, v]) => v) as [string, string][];
});

// место в сканах за сессию («Это вино — 2 из 5»)
const rank = ref<number | null>(null);
let rankSeq = 0;
watch([data, () => scanner.history.value.length], async () => {
  const h = scanner.history.value, my = ++rankSeq;
  rank.value = null;
  if (h.length < 2 || !h.includes(slug.value)) return;
  const res = await $fetch<{ items: Item[] }>("/api/wines", { method: "POST", body: { slugs: h, profile: som.goalProfile.value } })
    .catch(() => null);
  const r = res ? rankScans(res.items, som.hasGoal.value).findIndex((x) => x.wine.slug === slug.value) : -1;
  if (my === rankSeq) rank.value = r >= 0 ? r + 1 : null;
});
const sessionHint = computed(() => (som.hasGoal.value ? `Сравним под цель «${som.goalTitle.value}»` : "Сравним по народному рейтингу"));

useHead(() => ({ title: wine.value ? `${wineName(wine.value)} — Своё Вино` : "Своё Вино" }));
</script>

<template>
  <div class="page">
    <NuxtLink to="/" class="back">← Новое сканирование</NuxtLink>

    <div v-if="loading && !data" class="skeleton"><div class="sk-img" /><div class="sk-txt"><i /><i /><i /></div></div>

    <div v-else-if="failed || !wine" class="empty">
      <h1 class="h1">Вино не найдено</h1>
      <p class="muted">Возможно, ссылка устарела. Попробуйте отсканировать этикетку ещё раз.</p>
      <NuxtLink to="/" class="btn btn-primary">К сканеру</NuxtLink>
    </div>

    <template v-else>
      <div class="grid">
        <div class="visual">
          <RatingPill :rating="wine.rating" class="vr" />
          <div class="bottle"><WineImage :wine="wine" /></div>
          <figure v-if="fromScan && scanner.last.value?.photo" class="yours">
            <img :src="scanner.last.value.photo" alt="Ваше фото" />
            <figcaption>Ваше фото</figcaption>
          </figure>
        </div>

        <div class="info">
          <div class="title">
            <span class="muted sm">{{ [wine.manufacturer, wine.region].filter(Boolean).join(" · ") }}</span>
            <h1 class="h1">{{ wineName(wine) }}</h1>
            <span v-if="fromScan" class="found">
              <UiIcon name="check" :size="14" /> Найдено в каталоге<template v-if="debug"> за {{ seconds }} с
                · score {{ conf?.top1_score.toFixed(3) }} · отрыв {{ conf?.margin.toFixed(3) }}</template>
            </span>
          </div>

          <div v-if="wine.rating != null" class="stat">
            <span class="muted">Народный рейтинг</span>
            <span class="serif num">{{ fmtRating(wine.rating) }}</span>
          </div>

          <div v-if="match && vs" class="verdict">
            <div class="v-head" :style="{ background: vs.bg }">
              <div class="v-row">
                <span class="v-label" :style="{ color: vs.color }"><i :style="{ background: vs.color }" />{{ vs.label }}</span>
                <span class="muted xs">{{ okCount }} из {{ match.checks.length }}</span>
              </div>
              <span class="v-text">{{ match.verdict_text }}</span>
            </div>
            <div v-for="c in match.checks" :key="c.key + c.label" class="check">
              <span class="mark" :class="{ ok: c.ok }"><UiIcon :name="c.ok ? 'check' : 'minus'" :size="c.ok ? 14 : 12" /></span>
              <span class="c-label">{{ c.label }}</span>
              <span class="c-val">{{ c.value }}</span>
            </div>
          </div>
          <button v-else-if="!som.hasGoal.value" class="invite" @click="som.open.value = true">
            <UiIcon name="chat" :size="22" />
            <span>Подходит ли это вино к вашему ужину? Расскажите сомелье — проверим по вашей цели</span>
            <span class="arr">→</span>
          </button>

          <div v-if="attrs.length" class="attrs">
            <div v-for="[k, v] in attrs" :key="k" class="attr"><span class="muted">{{ k }}</span><span>{{ v }}</span></div>
          </div>

          <div v-if="wine.dishes?.length || wine.description" class="pair">
            <template v-if="wine.dishes?.length">
              <span class="muted sm">К чему подать</span>
              <div class="dishes"><span v-for="d in wine.dishes" :key="d">{{ d }}</span></div>
            </template>
            <p v-if="wine.description" class="desc">{{ wine.description }}</p>
          </div>

          <div class="actions">
            <a v-if="wine.url" :href="wine.url" target="_blank" rel="noopener" class="btn btn-primary">Карточка на «Своё Вино» ↗</a>
            <NuxtLink to="/" class="btn btn-secondary">Сканировать ещё</NuxtLink>
          </div>

          <NuxtLink v-if="rank" to="/session" class="link-card">
            <span class="lc">
              <b>Это вино — {{ rank }} из {{ scanner.history.value.length }} в ваших сканах</b>
              <span class="muted">{{ sessionHint }}</span>
            </span>
            <span class="cmp">Сравнить →</span>
          </NuxtLink>
        </div>
      </div>

      <section v-if="data?.alts.length" class="more">
        <h2 class="h2">Лучше подойдут под вашу цель</h2>
        <div class="tiles">
          <WineTile v-for="p in data.alts" :key="p.slug" :wine="p" :badge="{ label: 'Подходит', color: 'var(--ok)', bg: 'var(--ok-bg)' }" />
        </div>
      </section>
      <section v-else-if="data?.analogs.length" class="more">
        <h2 class="h2">Похожие вина других виноделен</h2>
        <div class="tiles">
          <WineTile v-for="p in data.analogs" :key="p.slug" :wine="p" />
        </div>
      </section>
    </template>
  </div>
</template>

<style scoped>
.page { display: flex; flex-direction: column; gap: 24px; padding-top: 8px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap: 20px; align-items: start; }
.sm { font-size: 14px; }
.xs { font-size: 12px; }

.visual { position: relative; background: rgba(251, 245, 234, .8); border-radius: 28px; padding: 20px; display: flex;
  flex-direction: column; align-items: center; }
.vr { align-self: flex-start; }
.bottle { height: 280px; width: 100%; display: flex; justify-content: center; margin: 8px 0 12px; }
.yours { position: absolute; right: 16px; bottom: 16px; margin: 0; display: flex; flex-direction: column; align-items: center; gap: 4px; }
.yours img { width: 64px; height: 84px; object-fit: cover; border-radius: 12px; border: 3px solid #fff; box-shadow: 0 4px 14px rgba(43, 37, 34, .12); }
.yours figcaption { font-size: 11px; color: var(--muted); }

.info { display: flex; flex-direction: column; gap: 20px; min-width: 0; }
.title { display: flex; flex-direction: column; gap: 6px; }
.found { font-size: 13px; color: var(--ok); display: flex; align-items: center; gap: 6px; }
.stat { background: #fff; border: 1px solid var(--line); border-radius: 20px; padding: 14px 16px; display: flex; flex-direction: column; gap: 2px; }
.stat .muted { font-size: 13px; }
.num { font-size: 34px; line-height: 1.1; }

.verdict { border-radius: 20px; border: 1px solid var(--line); overflow: hidden; background: #fff; }
.v-head { padding: 12px 16px; display: flex; flex-direction: column; gap: 4px; }
.v-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.v-label { font-weight: 600; font-size: 15px; display: flex; align-items: center; gap: 8px; }
.v-label i { width: 8px; height: 8px; border-radius: 50%; }
.v-text { color: var(--ink-2); font-size: 14px; text-wrap: pretty; }
.check { display: flex; align-items: center; gap: 12px; padding: 8px 16px; border-top: 1px solid #F1E9E1; }
.mark { width: 24px; height: 24px; flex: none; border-radius: 50%; background: #F1EAE3; color: var(--muted); display: flex;
  align-items: center; justify-content: center; }
.mark.ok { background: var(--ok-bg); color: var(--ok); }
.c-label { flex: 1; min-width: 0; font-size: 14px; }
.c-val { font-size: 14px; color: var(--ink-2); text-align: right; }

.invite { cursor: pointer; display: flex; align-items: center; gap: 14px; padding: 14px 18px; border-radius: 20px; border: none;
  background: var(--rose); color: var(--wine-deep); text-align: left; font: 400 14px/1.45 var(--sans); }
.invite span:not(.arr) { flex: 1; }
.invite .arr { font-size: 18px; }

.attrs { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1px; background: var(--line);
  border: 1px solid var(--line); border-radius: 20px; overflow: hidden; }
.attr { background: rgba(255, 253, 249, .95); padding: 12px 16px; display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.attr .muted { font-size: 12px; }
.attr span:last-child { font-size: 15px; font-weight: 500; overflow-wrap: anywhere; }

.pair { display: flex; flex-direction: column; gap: 10px; }
.dishes { display: flex; gap: 8px; flex-wrap: wrap; }
.dishes span { background: var(--cream); border-radius: 999px; padding: 6px 14px; font-size: 14px; }
.desc { margin: 4px 0 0; color: var(--ink-2); text-wrap: pretty; white-space: pre-line; }

.actions { display: flex; gap: 12px; flex-wrap: wrap; }
.actions .btn-primary { flex: 1 1 240px; }
.actions .btn-secondary { flex: 1 1 160px; }
.lc { display: flex; flex-direction: column; flex: 1; min-width: 0; }
.lc b { font-weight: 600; }
.lc .muted { font-size: 13px; }
.cmp { color: var(--wine); font-size: 14px; font-weight: 600; white-space: nowrap; }

.more { display: flex; flex-direction: column; gap: 16px; margin-top: 24px; }

.empty { display: flex; flex-direction: column; gap: 12px; align-items: flex-start; padding: 24px 0; }
.empty p { margin: 0; }
.skeleton { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap: 20px; }
.sk-img { height: 340px; border-radius: 28px; background: var(--cream); }
.sk-txt { display: flex; flex-direction: column; gap: 14px; }
.sk-txt i { height: 22px; border-radius: 8px; background: var(--cream); animation: wsDots 1.4s infinite; }
.sk-txt i:first-child { width: 40%; }
.sk-txt i:nth-child(2) { height: 44px; }
.sk-txt i:last-child { width: 70%; }

@media (min-width: 720px) {
  .grid, .skeleton { gap: 40px; }
  .visual { position: sticky; top: 96px; }
  .bottle { height: 420px; }
  .attrs { grid-template-columns: repeat(3, 1fr); }
}
</style>
