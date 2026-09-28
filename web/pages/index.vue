<script setup lang="ts">
// Главная: загрузка фото этикетки + экран поиска (пока идёт скан).
const som = useSommelier();
const scanner = useScanner();
const { busy, photo, error, history, catalogSize } = scanner;

const camera = ref<HTMLInputElement>();
const gallery = ref<HTMLInputElement>();
const drag = ref(false);

function run(f?: Blob | null) { scanner.scan(f, som.goalProfile.value); }
function onPick(e: Event) {
  const el = e.target as HTMLInputElement;
  run(el.files?.[0]);
  el.value = "";
}
function onDrop(e: DragEvent) { drag.value = false; run(e.dataTransfer?.files?.[0]); }
async function trySample() {
  const blob = await $fetch<Blob>("/img/sample-label.jpg", { responseType: "blob" }).catch(() => null);
  run(blob);
}

const hc = computed(() => history.value.length);
const catalogText = computed(() => (catalogSize.value
  ? `Сверяем с ${catalogSize.value.toLocaleString("ru-RU")} ${plural(catalogSize.value, "этикеткой", "этикетками", "этикетками")} «Своё Вино»`
  : "Сверяем с каталогом «Своё Вино»"));
</script>

<template>
  <section v-if="busy" class="scanning">
    <div class="frame">
      <img v-if="photo" :src="photo" alt="Ваше фото" />
      <div class="line" />
    </div>
    <div class="st">
      <div class="serif big">Ищем в каталоге…</div>
      <div class="steps">
        <span>Находим бутылку и этикетку на фото</span>
        <span>{{ catalogText }}</span>
        <span v-if="som.hasGoal.value">Проверяем цель «{{ som.goalTitle.value }}»</span>
      </div>
    </div>
  </section>

  <section v-else class="home">
    <div class="hero">
      <h1 class="serif">Сканер российских вин</h1>
      <p>Сфотографируйте этикетку — покажем карточку вина с народным рейтингом и подскажем, подходит ли оно вам</p>
    </div>

    <div class="dz" :class="{ over: drag }" @dragover.prevent="drag = true" @dragenter.prevent="drag = true"
         @dragleave.prevent="drag = false" @drop.prevent="onDrop">
      <div class="ico"><UiIcon name="camera" :size="32" /></div>
      <div class="dz-t">
        <div class="dz-title"><span class="m">Сфотографируйте этикетку</span><span class="d">Перетащите фото этикетки сюда</span></div>
        <div class="muted">Этикетка целиком в кадре, можно под углом · JPG, PNG, HEIC</div>
      </div>
      <div class="btns">
        <button class="btn btn-primary" @click="camera?.click()"><UiIcon name="camera" /> Сфотографировать</button>
        <button class="btn btn-secondary" @click="gallery?.click()"><UiIcon name="upload" /> Из галереи</button>
      </div>
      <button class="sample" @click="trySample">Попробовать на примере</button>
      <p v-if="error" class="err">{{ error }}</p>
      <input ref="camera" type="file" accept="image/*" capture="environment" hidden @change="onPick" />
      <input ref="gallery" type="file" accept="image/*" hidden @change="onPick" />
    </div>

    <NuxtLink v-if="hc" to="/session" class="link-card">
      <span class="lc">
        <b>Ваши сканы за сессию · {{ hc }}</b>
        <span class="muted">{{ som.hasGoal.value ? "Отсортируем под вашу цель" : "Отсортированы по народному рейтингу" }}</span>
      </span>
      <span class="arr">→</span>
    </NuxtLink>

    <div class="ask">
      <span class="muted">Не знаете, что выбрать? Спросите сомелье</span>
      <div class="chips">
        <button v-for="q in QUICK_PROMPTS" :key="q" class="chip" @click="som.ask(q)">{{ q }}</button>
      </div>
    </div>
  </section>
</template>

<style scoped>
.home { display: flex; flex-direction: column; gap: 28px; padding-top: 12px; }
.hero { text-align: center; display: flex; flex-direction: column; gap: 12px; align-items: center; }
.hero h1 { margin: 0; font-size: 36px; line-height: 1.1; text-wrap: balance; }
.hero p { margin: 0; max-width: 560px; font-size: 16px; color: var(--ink-2); text-wrap: pretty; }

.dz { border: 2px dashed var(--line-2); background: rgba(255, 253, 249, .7); border-radius: 32px; padding: 28px 18px;
  display: flex; flex-direction: column; align-items: center; gap: 18px; text-align: center; transition: background .15s, border-color .15s; }
.dz.over { border-color: var(--wine); background: rgba(243, 230, 227, .9); }
.ico { width: 72px; height: 72px; border-radius: 50%; background: var(--rose); color: var(--wine); display: flex;
  align-items: center; justify-content: center; }
.dz-t { display: flex; flex-direction: column; gap: 4px; }
.dz-title { font-size: 20px; font-weight: 600; }
.dz-title .d { display: none; }
.dz-t .muted { font-size: 14px; }
.btns { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; width: 100%; }
.btns .btn { flex: 1 1 100%; }
.sample { background: none; border: none; font-size: 14px; color: var(--wine); cursor: pointer; text-decoration: underline;
  text-underline-offset: 3px; padding: 0; }
.err { margin: 0; color: var(--wine); font-size: 14px; }

.lc { display: flex; flex-direction: column; flex: 1; min-width: 0; }
.lc b { font-weight: 600; }
.lc .muted { font-size: 13px; }
.arr { color: var(--wine); font-size: 20px; }

.ask { display: flex; flex-direction: column; gap: 12px; align-items: center; text-align: center; }
.ask .muted { font-size: 14px; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }

.scanning { display: flex; flex-direction: column; align-items: center; gap: 28px; padding-top: 40px; text-align: center; }
.frame { position: relative; width: 260px; height: 340px; border-radius: 28px; overflow: hidden; background: var(--cream); }
.frame img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.line { position: absolute; left: 0; right: 0; height: 3px; background: var(--wine); box-shadow: 0 0 18px 6px rgba(142, 58, 69, .45);
  animation: wsScan 1.8s ease-in-out infinite; }
.st { display: flex; flex-direction: column; gap: 8px; align-items: center; }
.big { font-size: 28px; }
.steps { display: flex; flex-direction: column; gap: 4px; color: var(--muted); font-size: 14px; }

@media (min-width: 720px) {
  .home { padding-top: 40px; }
  .hero h1 { font-size: 60px; }
  .hero p { font-size: 18px; }
  .dz { padding: 48px 32px; }
  .dz-title .m { display: none; }
  .dz-title .d { display: inline; }
  .btns .btn { flex: 0 0 auto; padding: 0 32px; }
}
</style>
