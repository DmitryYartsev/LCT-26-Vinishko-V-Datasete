<script setup lang="ts">
const som = useSommelier();
const scanner = useScanner();

const route = useRoute();
const showFab = computed(() => !som.open.value && !scanner.busy.value);
watch(() => route.fullPath, () => { som.open.value = false; });
</script>

<template>
  <div class="app">
    <img src="/img/bg-left.png" alt="" class="decor l" />
    <img src="/img/bg-right.png" alt="" class="decor r" />

    <header class="head">
      <div class="head-in"><NuxtLink to="/" class="brand">
        <img src="/img/logo-mark.png" alt="" width="40" height="40" />
        <span class="names">
          <span class="serif name">Своё Вино</span>
          <span class="by">От Россельхозбанка</span>
        </span>
      </NuxtLink></div>
    </header>

    <main class="main">
      <GoalBar v-if="som.hasGoal.value" />
      <slot />
    </main>

    <div v-if="showFab" class="fab-wrap">
      <button class="fab" @click="som.open.value = true"><UiIcon name="chat" /> Спросить сомелье</button>
    </div>

    <SommelierDrawer />
  </div>
</template>

<style scoped>
.app { position: relative; min-height: 100vh; min-height: 100dvh; overflow-x: hidden; }
.decor { position: absolute; top: 60px; height: 560px; opacity: .32; pointer-events: none; z-index: 0;
  -webkit-mask-composite: source-in; mask-composite: intersect; }
.decor.l { left: 0;
  -webkit-mask-image: linear-gradient(to bottom, #000 70%, transparent), linear-gradient(to right, #000 55%, transparent);
  mask-image: linear-gradient(to bottom, #000 70%, transparent), linear-gradient(to right, #000 55%, transparent); }
.decor.r { right: 0;
  -webkit-mask-image: linear-gradient(to bottom, #000 70%, transparent), linear-gradient(to left, #000 55%, transparent);
  mask-image: linear-gradient(to bottom, #000 70%, transparent), linear-gradient(to left, #000 55%, transparent); }

.head { position: sticky; top: 0; z-index: 5; padding: 10px 12px 14px; background: rgba(252, 250, 246, .82);
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); }
.head-in { max-width: 1200px; margin: 0 auto; }
.brand { display: flex; align-items: center; gap: 10px; color: inherit; width: fit-content; }
.brand img { width: 40px; height: 40px; flex: none; }
.names { display: flex; flex-direction: column; line-height: 1.1; }
.name { font-size: 22px; font-weight: 500; color: var(--ink); }
.by { font-size: 12px; color: var(--muted); }

.main { position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 0 var(--pad-x) 112px; }

.fab-wrap { position: fixed; left: 0; right: 0; bottom: 20px; bottom: calc(20px + env(safe-area-inset-bottom)); z-index: 10;
  display: flex; padding: 0 var(--pad-x); pointer-events: none; max-width: 1264px; margin: 0 auto; }
.fab { pointer-events: auto; width: 100%; height: 56px; padding: 0 26px; border: none; border-radius: 999px; background: var(--fab);
  color: #fff; font: 600 16px var(--sans); cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 10px;
  box-shadow: 0 10px 30px rgba(123, 47, 43, .3); transition: background .15s; }
.fab:hover { background: var(--fab-hover); }

@media (min-width: 720px) {
  .decor { height: 1000px; }
  .head { padding: 16px 32px 20px; }
  .fab-wrap { justify-content: flex-end; }
  .fab { width: auto; }
}
</style>
