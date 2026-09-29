<script setup lang="ts">
// Уверенного совпадения нет: похожие по этикетке (top-5 поиска) + аналоги по стилю из других виноделен.
const som = useSommelier();
const scanner = useScanner();
const last = scanner.last;
const camera = ref<HTMLInputElement>();
const debug = useRuntimeConfig().public.debugInfo;

onMounted(() => { if (!last.value || last.value.in_catalog) navigateTo("/", { replace: true }); });

function onPick(e: Event) {
  const el = e.target as HTMLInputElement;
  scanner.scan(el.files?.[0], som.goalProfile.value);
  el.value = "";
}
function badge(v: string | null) {
  if (!v || !som.hasGoal.value) return null;
  const s = VERDICT_STYLE[v as keyof typeof VERDICT_STYLE];
  return { label: v === "ok" ? "Подходит цели" : v === "part" ? "Частично" : "Не под цель", color: s.color, bg: s.bg };
}
</script>

<template>
  <div v-if="last" class="page">
    <NuxtLink to="/" class="back">← Новое сканирование</NuxtLink>

    <div class="head">
      <img v-if="last.photo" :src="last.photo" alt="Ваше фото" class="ph" />
      <div class="txt">
        <h1 class="h1">Точного совпадения нет</h1>
        <span v-if="debug" class="dbg">за {{ last.seconds.toFixed(1) }} с · score {{ last.confidence.top1_score.toFixed(3) }}
          · отрыв {{ last.confidence.margin.toFixed(3) }}</span>
        <p>Возможно, этого вина пока нет в каталоге. Вот самые похожие — по этикетке, стилю и винограду.
          Или переснимите фото ровнее и без бликов.</p>
      </div>
      <button class="btn btn-primary" @click="camera?.click()"><UiIcon name="camera" /> Переснять</button>
      <input ref="camera" type="file" accept="image/*" capture="environment" hidden @change="onPick" />
    </div>

    <section v-if="last.similar?.length" class="sec">
      <h2 class="h2">Похожие по этикетке</h2>
      <div class="tiles">
        <WineTile v-for="s in last.similar" :key="s.wine.slug" :wine="s.wine" :badge="badge(s.verdict)"
                  :note="debug ? `сходство ${s.sim}%` : undefined" />
      </div>
    </section>

    <section v-if="last.analogs?.length" class="sec">
      <h2 class="h2">Аналоги из других виноделен</h2>
      <p class="muted lead">Тот же стиль и сорт — от производителей, которые есть в каталоге «Своё Вино».</p>
      <div class="tiles">
        <WineTile v-for="p in last.analogs" :key="p.slug" :wine="p" />
      </div>
    </section>
  </div>
</template>

<style scoped>
.page { display: flex; flex-direction: column; gap: 24px; padding-top: 8px; }
.head { display: flex; gap: 20px; align-items: center; flex-wrap: wrap; background: rgba(251, 245, 234, .8); border-radius: 28px; padding: 20px; }
.ph { width: 84px; height: 110px; object-fit: cover; border-radius: 16px; border: 3px solid #fff; flex: none; }
.txt { flex: 1; min-width: 220px; display: flex; flex-direction: column; gap: 6px; }
.dbg { font-size: 12px; color: var(--muted); }
.txt p { margin: 0; color: var(--ink-2); text-wrap: pretty; }
.head .btn { height: 52px; border-radius: 16px; font-size: 15px; padding: 0 24px; }
.sec { display: flex; flex-direction: column; gap: 16px; }
.sec + .sec { margin-top: 16px; }
.lead { margin: -8px 0 0; font-size: 14px; }
@media (max-width: 719px) { .head .btn { width: 100%; } }
@media (min-width: 720px) { .head { padding: 28px 32px; } }
</style>
