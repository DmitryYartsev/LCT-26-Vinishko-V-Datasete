<script setup lang="ts">
type Card = { slug: string; name?: string; winery?: string; region?: string; grape?: string;
  color?: string; category?: string; description?: string; rating?: string };
type Res = { slug: string; score: number; card: Card };
type Scan = { in_catalog: boolean; confidence: { top1_score: number; margin: number };
  top1: Res | null; results: Res[] };

const loading = ref(false);
const preview = ref<string | null>(null);
const scan = ref<Scan | null>(null);
const error = ref<string | null>(null);
const dragging = ref(false);
const fileInput = ref<HTMLInputElement>();
const galInput = ref<HTMLInputElement>();

async function onFile(f?: File | null) {
  if (!f || !f.type.startsWith("image/")) return;
  preview.value = URL.createObjectURL(f);
  scan.value = null; error.value = null; loading.value = true;
  try {
    const fd = new FormData();
    fd.append("image", f);
    scan.value = await $fetch<Scan>("/api/scan", { method: "POST", body: fd });
  } catch {
    error.value = "Не удалось распознать. Попробуйте ещё раз.";
  } finally {
    loading.value = false;
  }
}

function onPick(e: Event) { onFile((e.target as HTMLInputElement).files?.[0]); }
function onDrop(e: DragEvent) { dragging.value = false; onFile(e.dataTransfer?.files?.[0]); }

const card = computed(() => scan.value?.top1?.card);
const conf = computed(() => Math.round((scan.value?.confidence.top1_score ?? 0) * 100));
function reset() { scan.value = null; preview.value = null; error.value = null; }
</script>

<template>
  <div class="wrap">
    <header class="bar">
      <div class="logo"><span class="dot" /> Своё&nbsp;Вино</div>
      <div class="sub">От Россельхозбанка</div>
    </header>

    <section v-if="!scan && !loading" class="hero">
      <h1>Сканер<br />российских вин</h1>
      <p>Наведите камеру на этикетку или перетащите фото — покажем карточку вина из каталога «Своё Вино».</p>

      <div class="dropzone" :class="{ over: dragging }"
           @dragover.prevent="dragging = true" @dragenter.prevent="dragging = true"
           @dragleave.prevent="dragging = false" @drop.prevent="onDrop">
        <img v-if="preview" :src="preview" class="preview" />
        <template v-else>
          <div class="dz-ico">🍷</div>
          <div class="dz-hint">Перетащите фото сюда</div>
        </template>
        <div class="dz-actions">
          <button class="btn primary" @click="fileInput?.click()">Сфотографировать этикетку</button>
          <button class="btn ghost" @click="galInput?.click()">Выбрать из галереи</button>
        </div>
      </div>
      <p v-if="error" class="err">{{ error }}</p>
    </section>

    <section v-if="loading" class="loading">
      <div class="spinner" />
      <p>Ищем вино…</p>
    </section>

    <section v-if="scan && !loading" class="result">
      <div class="banner" :class="{ warn: !scan.in_catalog }">
        {{ scan.in_catalog ? "Найдено в каталоге" : "Точного совпадения нет — ближайшее" }}
        <span class="conf">уверенность {{ conf }}%</span>
      </div>

      <article class="card" v-if="card">
        <div class="compare">
          <figure>
            <img :src="preview!" />
            <figcaption>Ваше фото</figcaption>
          </figure>
          <figure>
            <img :src="`/api/ref/${card.slug}`" @error="(e:any)=>e.target.closest('figure').style.display='none'" />
            <figcaption>Эталон каталога</figcaption>
          </figure>
        </div>
        <div class="body">
          <div class="tags">
            <span v-if="card.category" class="tag">{{ card.category }}</span>
            <span v-if="card.color" class="tag light">{{ card.color }}</span>
            <span v-if="card.rating" class="tag light">Роскачество {{ card.rating }}</span>
          </div>
          <h2>{{ card.name || card.slug }}</h2>
          <div class="winery" v-if="card.winery">{{ card.winery }}</div>
          <dl class="meta">
            <div v-if="card.region"><dt>Регион</dt><dd>{{ card.region }}</dd></div>
            <div v-if="card.grape"><dt>Сорт</dt><dd>{{ card.grape }}</dd></div>
          </dl>
          <p class="desc" v-if="card.description">{{ card.description }}</p>
        </div>
      </article>

      <button class="btn primary" @click="reset">Сканировать ещё</button>
    </section>

    <input ref="fileInput" type="file" accept="image/*" capture="environment" hidden @change="onPick" />
    <input ref="galInput" type="file" accept="image/*" hidden @change="onPick" />
  </div>
</template>

<style>
:root {
  --bg: #f4ecdd; --surface: #ece1cf; --card: #fbf6ee; --wine: #7c2e3b; --wine-d: #5f2028;
  --ink: #2c2724; --muted: #8a7d6d; --line: #e3d6c1;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font-family: Inter, system-ui, sans-serif; -webkit-font-smoothing: antialiased; }
.wrap { max-width: 460px; margin: 0 auto; min-height: 100vh; padding: 16px; }
.bar { display: flex; align-items: baseline; gap: 10px; padding: 8px 4px 20px; }
.logo { font-family: "Playfair Display", Georgia, serif; font-weight: 700; font-size: 22px;
  display: flex; align-items: center; gap: 8px; }
.dot { width: 20px; height: 20px; border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, #a9455a, var(--wine)); display: inline-block; }
.sub { color: var(--muted); font-size: 12px; }
h1 { font-family: "Playfair Display", Georgia, serif; font-weight: 700; font-size: 40px;
  line-height: 1.05; margin: 12px 0 10px; }
.hero p { color: var(--muted); font-size: 15px; line-height: 1.5; margin: 0 0 22px; }
.dropzone { display: flex; flex-direction: column; gap: 12px; align-items: stretch;
  background: var(--surface); border: 2px dashed transparent; border-radius: 22px; padding: 20px;
  transition: border-color .15s, background .15s; }
.dropzone.over { border-color: var(--wine); background: #efe1cc; }
.dz-ico { font-size: 40px; text-align: center; }
.dz-hint { text-align: center; color: var(--muted); font-size: 14px; margin-bottom: 4px; }
.preview { width: 100%; border-radius: 16px; object-fit: cover; max-height: 300px; }
.dz-actions { display: flex; flex-direction: column; gap: 12px; }
.btn { border: none; border-radius: 999px; padding: 16px 20px; font-size: 15px; font-weight: 600;
  cursor: pointer; font-family: inherit; }
.btn.primary { background: var(--wine); color: #fff; }
.btn.primary:active { background: var(--wine-d); }
.btn.ghost { background: transparent; color: var(--wine); border: 1px solid var(--wine); }
.loading { display: flex; flex-direction: column; align-items: center; gap: 16px; padding: 80px 0;
  color: var(--muted); }
.spinner { width: 42px; height: 42px; border: 4px solid var(--line); border-top-color: var(--wine);
  border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.banner { background: #e8efe4; color: #35603f; border-radius: 14px; padding: 12px 16px;
  font-size: 14px; font-weight: 600; display: flex; justify-content: space-between; align-items: center; }
.banner.warn { background: #f7ead6; color: #8a5a1e; }
.conf { font-weight: 500; opacity: 0.8; font-size: 12px; }
.card { background: var(--card); border-radius: 22px; overflow: hidden; margin: 14px 0;
  box-shadow: 0 8px 30px rgba(60, 40, 20, 0.08); }
.compare { display: grid; grid-template-columns: 1fr 1fr; gap: 2px; background: var(--line); }
.compare figure { margin: 0; background: #fff; display: flex; flex-direction: column; }
.compare img { width: 100%; height: 220px; object-fit: contain; background: #fff; padding: 14px 8px 4px; }
.compare figcaption { text-align: center; font-size: 12px; color: var(--muted); padding: 4px 0 12px; }
.body { padding: 18px 20px 22px; }
.tags { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
.tag { background: var(--wine); color: #fff; border-radius: 999px; padding: 5px 12px; font-size: 12px; font-weight: 600; }
.tag.light { background: #efe3d0; color: var(--wine); }
h2 { font-family: "Playfair Display", Georgia, serif; font-size: 26px; margin: 4px 0 4px; line-height: 1.15; }
.winery { color: var(--muted); font-size: 14px; margin-bottom: 14px; }
.meta { display: flex; gap: 24px; margin: 0 0 14px; padding: 12px 0; border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line); }
.meta dt { color: var(--muted); font-size: 12px; }
.meta dd { margin: 2px 0 0; font-size: 15px; font-weight: 600; }
.desc { font-size: 14px; line-height: 1.6; color: #4a423b; margin: 0; }
.err { color: var(--wine); font-size: 14px; margin-top: 14px; }
.result > .btn { width: 100%; margin-top: 6px; }
</style>
