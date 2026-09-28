<script setup lang="ts">
// Сканы за сессию: все бутылки, которые сфотографировали, — от лучшего выбора под цель.
import type { Item } from "~/utils/wine";
const som = useSommelier();
const scanner = useScanner();
const items = ref<Item[] | null>(null);

let seq = 0;
watch([() => scanner.history.value.join(), () => JSON.stringify(som.goalProfile.value)], async () => {
  const slugs = scanner.history.value, my = ++seq;
  if (!slugs.length) { items.value = []; return; }
  const res = await $fetch<{ items: Item[] }>("/api/wines", { method: "POST", body: { slugs, profile: som.goalProfile.value } })
    .catch(() => ({ items: [] as Item[] }));
  if (my === seq) items.value = res.items;
}, { immediate: true });

const ranked = computed(() => rankScans(items.value || [], som.hasGoal.value));
const lastSlug = computed(() => scanner.history.value[scanner.history.value.length - 1]);
const lead = computed(() => (som.hasGoal.value
  ? `Все бутылки, которые вы сфотографировали, — от лучшего выбора под цель «${som.goalTitle.value}».`
  : "Все бутылки, которые вы сфотографировали, по народному рейтингу. Задайте цель у сомелье — ранжируем под неё."));
</script>

<template>
  <div class="page">
    <NuxtLink to="/" class="back">← К сканеру</NuxtLink>
    <div class="title">
      <h1 class="h1">Ваши сканы за сессию</h1>
      <p>{{ lead }}</p>
    </div>

    <div v-if="items === null" class="list"><div v-for="k in 3" :key="k" class="sk" /></div>
    <div v-else-if="!ranked.length" class="empty">
      <p class="muted">Вы ещё ничего не сканировали. Сфотографируйте пару бутылок на полке — сравним их здесь.</p>
      <NuxtLink to="/" class="btn btn-primary"><UiIcon name="camera" /> Сканировать</NuxtLink>
    </div>
    <div v-else class="list">
      <WineRow v-for="(r, i) in ranked" :key="r.wine.slug" :wine="r.wine" :rank="i + 1" :reason="r.reason" :good="r.good"
               :active="r.wine.slug === lastSlug" />
    </div>
    <button v-if="!som.hasGoal.value && ranked.length > 1" class="invite" @click="som.open.value = true">
      <UiIcon name="chat" :size="22" />
      <span>Расскажите сомелье, к чему вино, — отсортируем сканы под вашу цель</span>
      <span>→</span>
    </button>
  </div>
</template>

<style scoped>
.page { display: flex; flex-direction: column; gap: 24px; padding-top: 8px; max-width: 760px; margin: 0 auto; }
.title { display: flex; flex-direction: column; gap: 6px; }
.title p { margin: 0; color: var(--ink-2); text-wrap: pretty; }
.list { display: flex; flex-direction: column; gap: 10px; }
.sk { height: 98px; border-radius: 20px; background: var(--cream); animation: wsDots 1.4s infinite; }
.empty { display: flex; flex-direction: column; gap: 12px; align-items: flex-start; }
.empty p { margin: 0; }
.invite { cursor: pointer; display: flex; align-items: center; gap: 14px; padding: 14px 18px; border-radius: 20px; border: none;
  background: var(--rose); color: var(--wine-deep); text-align: left; font: 400 14px/1.45 var(--sans); }
.invite span:nth-child(2) { flex: 1; }
</style>
