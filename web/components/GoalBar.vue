<script setup lang="ts">
// Плашка «Цель: …» — то, что сомелье запомнил на сессию; по ней оценивается каждый скан.
const { goalTitle, open, clearGoal } = useSommelier();
</script>

<template>
  <div class="goal">
    <UiIcon name="target" :size="18" />
    <span class="t">Цель: <b>{{ goalTitle }}</b></span>
    <div class="act">
      <button class="edit" @click="open = true">Изменить</button>
      <button class="x" aria-label="Сбросить цель" @click="clearGoal">×</button>
    </div>
  </div>
</template>

<style scoped>
.goal { display: flex; align-items: center; gap: 10px; margin: 4px 0 8px; padding: 8px 8px 8px 16px; border-radius: 18px;
  background: rgba(246, 238, 232, .9); color: var(--ink-2); font-size: 14px; line-height: 1.35; animation: wsFade .25s; }
.t { flex: 1; min-width: 0; }
.t b { font-weight: 600; color: var(--ink); }
.act { display: flex; align-items: center; gap: 2px; flex: none; }
.edit { border: none; background: #fff; color: var(--wine); border-radius: 999px; padding: 6px 14px; font: 500 13px var(--sans); cursor: pointer; }
.x { border: none; background: transparent; color: var(--wine); width: 32px; height: 32px; border-radius: 50%; cursor: pointer;
  font-size: 18px; line-height: 1; }
</style>
