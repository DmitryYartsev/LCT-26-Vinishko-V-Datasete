<script setup lang="ts">
// Карточка вина в сетке: «лучше под цель», «похожие», «аналоги».
import type { Wine } from "~/utils/wine";
defineProps<{ wine: Wine; badge?: { label: string; color: string; bg: string } | null; note?: string }>();
</script>

<template>
  <NuxtLink :to="`/wine/${encodeURIComponent(wine.slug)}`" class="tile">
    <div class="top">
      <RatingPill :rating="wine.rating" small />
      <span v-if="badge" class="badge" :style="{ color: badge.color, background: badge.bg }">{{ badge.label }}</span>
    </div>
    <div class="img"><WineImage :wine="wine" /></div>
    <div class="txt">
      <span class="name">{{ wineName(wine) }}</span>
      <span class="sub">{{ wine.manufacturer }}</span>
      <span v-if="note || wine.tagline" class="note">{{ [wine.tagline, note].filter(Boolean).join(" · ") }}</span>
    </div>
  </NuxtLink>
</template>

<style scoped>
.tile { cursor: pointer; background: rgba(251, 245, 234, .8); border-radius: 24px; padding: 14px; display: flex;
  flex-direction: column; gap: 10px; color: inherit; transition: background .15s; min-width: 0; }
.tile:hover { background: #F7EEDF; color: inherit; }
.top { display: flex; justify-content: space-between; align-items: center; gap: 6px; flex-wrap: wrap; min-height: 26px; }
.badge { font-size: 12px; border-radius: 999px; padding: 3px 10px; margin-left: auto; }
.img { height: 150px; display: flex; justify-content: center; }
.txt { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.name { font-weight: 500; text-transform: uppercase; font-size: 14px; line-height: 1.3; overflow-wrap: anywhere; }
.sub { font-size: 13px; color: var(--muted); }
.note { font-size: 12px; color: var(--muted); }
@media (min-width: 720px) { .img { height: 190px; } }
</style>
