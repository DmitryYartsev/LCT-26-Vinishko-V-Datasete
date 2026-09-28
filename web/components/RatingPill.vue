<script setup lang="ts">
// Народный рейтинг с бокалом, как на vino-svoe.ru. Нет рейтинга — ничего не рисуем.
const props = defineProps<{ rating?: number | null; plain?: boolean; small?: boolean }>();
const text = computed(() => fmtRating(props.rating));
</script>

<template>
  <span v-if="text" class="rp" :class="{ plain, small }" :title="`Народный рейтинг ${text} из 5`">
    <svg :width="small ? 12 : 16" :height="small ? 14 : 18" viewBox="0 0 16 18" aria-hidden="true">
      <path d="M3 1h10c0 5-1.5 8-5 8S3 6 3 1z" fill="#B3262E" />
      <path d="M8 9v6M4.5 17h7" stroke="#8A817C" stroke-width="1.4" stroke-linecap="round" />
    </svg>
    {{ text }}
  </span>
</template>

<style scoped>
.rp { display: inline-flex; align-items: center; gap: 6px; background: var(--cream-2); border-radius: 999px;
  padding: 4px 12px 4px 8px; font-size: 14px; font-weight: 500; white-space: nowrap; flex: none; }
.rp.small { font-size: 13px; gap: 5px; padding: 3px 10px 3px 7px; }
.rp.plain { background: none; padding: 0; gap: 4px; }
</style>
