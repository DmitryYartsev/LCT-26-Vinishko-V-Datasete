<script setup lang="ts">
// Фото бутылки: эталон каталога (или фото с сайта для вин вне каталога).
// Нет фото / не загрузилось — силуэт бутылки в цвет вина, как в макете.
import type { Wine } from "~/utils/wine";
const props = defineProps<{ wine?: Wine | null }>();
const failed = ref(false);
const src = computed(() => wineImg(props.wine));
watch(src, () => (failed.value = false));
const BOTTLE: Record<string, string> = { красное: "#3A1519", белое: "#C9AE5C", розовое: "#C98A8A", оранжевое: "#C7853F" };
const bottle = computed(() => BOTTLE[props.wine?.color || ""] || "#4A1B22");
</script>

<template>
  <img v-if="src && !failed" :src="src" :alt="wine?.title || ''" loading="lazy" class="wi" @error="failed = true" />
  <div v-else class="sil" aria-hidden="true">
    <div class="neck" :style="{ background: bottle }" />
    <div class="body" :style="{ background: bottle }"><div class="label" /></div>
  </div>
</template>

<style scoped>
.wi { width: 100%; height: 100%; object-fit: contain; mix-blend-mode: multiply; }
.sil { height: 92%; aspect-ratio: 1 / 3.3; display: flex; flex-direction: column; align-items: center; margin: auto; }
.neck { width: 30%; height: 34%; border-radius: 6px 6px 0 0; }
.body { position: relative; width: 100%; height: 66%; border-radius: 40% 40% 10px 10px / 18% 18% 10px 10px; }
.label { position: absolute; left: 8%; right: 8%; top: 32%; height: 40%; background: #EFE6D6; border-radius: 3px; }
</style>
