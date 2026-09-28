<script setup lang="ts">
// Строка вина: сканы за сессию и рекомендации сомелье в чате.
import type { Wine } from "~/utils/wine";
defineProps<{ wine: Wine; rank?: number; reason?: string; good?: boolean; active?: boolean }>();
defineEmits<{ open: [] }>();
</script>

<template>
  <NuxtLink :to="`/wine/${encodeURIComponent(wine.slug)}`" class="row" :class="{ active, ranked: rank }" @click="$emit('open')">
    <span v-if="rank" class="rank serif">{{ rank }}</span>
    <div class="thumb"><WineImage :wine="wine" /></div>
    <div class="txt">
      <span class="name">{{ wineName(wine) }}</span>
      <span class="sub">{{ [wine.manufacturer, wine.tagline].filter(Boolean).join(" · ") }}</span>
      <span v-if="reason" class="reason" :class="{ good }">{{ reason }}</span>
    </div>
    <RatingPill :rating="wine.rating" plain small />
  </NuxtLink>
</template>

<style scoped>
.row { cursor: pointer; display: flex; align-items: center; gap: 12px; background: rgba(255, 255, 255, .82);
  border: 1px solid var(--line); border-radius: 18px; padding: 10px 14px 10px 10px; color: inherit; transition: border-color .15s; }
.row:hover, .row.active { border-color: var(--wine); color: inherit; }
.row.ranked { border-radius: 20px; gap: 14px; padding-right: 16px; }
.rank { width: 28px; text-align: center; font-size: 22px; color: var(--muted); flex: none; }
.thumb { width: 44px; height: 72px; background: var(--cream); border-radius: 12px; display: flex; justify-content: center;
  align-items: center; flex: none; padding: 4px; overflow: hidden; }
.ranked .thumb { width: 48px; height: 76px; }
.txt { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 2px; line-height: 1.3; }
.name { font-weight: 500; font-size: 14px; }
.ranked .name { font-size: 15px; }
.sub { font-size: 12px; color: var(--muted); }
.reason { font-size: 13px; color: var(--muted); }
.reason.good { color: var(--ok); }
</style>
