<script setup lang="ts">
// Цифровой сомелье: выезжает справа (десктоп) / на весь экран (телефон).
// Запоминает цель на сессию — по ней потом оценивается каждый скан.
const { messages, busy, error, open, send, transcribe } = useSommelier();

const draft = ref("");
const list = ref<HTMLElement>();
const input = ref<HTMLInputElement>();
const recording = ref(false);
const transcribing = ref(false);
let recorder: MediaRecorder | null = null;
let chunks: Blob[] = [];

async function submit(text = draft.value) {
  draft.value = "";
  const back = await send(text);
  if (back) draft.value = back;
}

const scrollDown = () => nextTick(() => list.value && (list.value.scrollTop = list.value.scrollHeight));
watch(() => [messages.value.length, busy.value, transcribing.value], scrollDown);
watch(open, (v) => {
  document.documentElement.style.overflow = v ? "hidden" : "";
  if (v) { scrollDown(); if (window.matchMedia("(min-width: 720px)").matches) nextTick(() => input.value?.focus()); }
  else if (recording.value) recorder?.stop();
});
function onKey(e: KeyboardEvent) { if (e.key === "Escape" && open.value) open.value = false; }
onMounted(() => window.addEventListener("keydown", onKey));
onBeforeUnmount(() => window.removeEventListener("keydown", onKey));

// подсказки-ответы — только под последней репликой сомелье
const lastBot = computed(() => {
  const m = messages.value;
  return m.length && m[m.length - 1].role === "assistant" ? m.length - 1 : -1;
});

async function toggleMic() {
  if (recording.value) { recorder?.stop(); return; }
  error.value = null;
  // вне secure context браузер вообще не отдаёт mediaDevices — просить разрешение бесполезно
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    error.value = "Голосовой ввод работает только по HTTPS — откройте сайт через https://.";
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      recording.value = false;
      transcribing.value = true;
      try {
        const text = await transcribe(new Blob(chunks, { type: recorder?.mimeType || "audio/webm" }));
        if (text.trim()) await submit(text);
        else error.value = "Не расслышал — попробуйте ещё раз.";
      } catch {
        error.value = "Не удалось распознать голос.";
      } finally {
        transcribing.value = false;
      }
    };
    recorder.start();
    recording.value = true;
  } catch {
    error.value = "Нет доступа к микрофону — разрешите его в настройках браузера.";
  }
}
</script>

<template>
  <Transition name="dr">
    <div v-if="open" class="shade" @click.self="open = false">
      <aside class="panel" role="dialog" aria-label="Цифровой сомелье">
        <div class="hd">
          <div class="ava">
            <svg width="20" height="22" viewBox="0 0 16 18" aria-hidden="true">
              <path d="M3 1h10c0 5-1.5 8-5 8S3 6 3 1z" fill="#fff" />
              <path d="M8 9v6M4.5 17h7" stroke="#fff" stroke-width="1.4" stroke-linecap="round" />
            </svg>
          </div>
          <div class="ttl">
            <span class="serif">Цифровой сомелье</span>
            <span class="muted">Подберу вино и запомню цель на сессию</span>
          </div>
          <button class="close" aria-label="Закрыть" @click="open = false">×</button>
        </div>

        <div ref="list" class="msgs">
          <div class="bot">Здравствуйте! Расскажите, к какому блюду или поводу нужно вино, какой вкус любите. Можно голосом.</div>
          <template v-for="(m, i) in messages" :key="i">
            <div v-if="m.role === 'user'" class="user">{{ m.content }}</div>
            <div v-else class="bot-wrap">
              <div class="bot">{{ m.content }}</div>
              <div v-if="m.goal" class="saved"><UiIcon name="check" :size="14" /> Цель сохранена: {{ m.goal }}</div>
              <WineRow v-for="p in m.picks || []" :key="p.slug" :wine="p" />
              <div v-if="i === lastBot && m.suggestions?.length" class="chips">
                <button v-for="s in m.suggestions" :key="s" class="chip sm" :disabled="busy" @click="submit(s)">{{ s }}</button>
              </div>
            </div>
          </template>
          <div v-if="busy || transcribing" class="bot typing" :aria-label="transcribing ? 'Слушаю запись' : 'Сомелье печатает'">
            <span v-for="k in 3" :key="k" :style="{ animationDelay: (k - 1) * 0.15 + 's' }" />
          </div>
        </div>

        <div v-if="!messages.length" class="quick">
          <button v-for="q in QUICK_PROMPTS" :key="q" class="chip sm" :disabled="busy" @click="submit(q)">{{ q }}</button>
        </div>
        <p v-if="error" class="err">{{ error }}</p>

        <form class="bar" @submit.prevent="submit()">
          <button type="button" class="mic" :class="{ rec: recording }" :disabled="busy || transcribing"
                  :aria-label="recording ? 'Остановить запись' : 'Голосовой ввод'" @click="toggleMic">
            <UiIcon name="mic" />
          </button>
          <div v-if="recording" class="listening"><span class="dot" />Слушаю… нажмите ещё раз, чтобы закончить</div>
          <input v-else ref="input" v-model="draft" :disabled="busy" placeholder="Например: терпкое красное к стейку"
                 enterkeyhint="send" />
          <button type="submit" class="send" aria-label="Отправить" :disabled="busy || recording || !draft.trim()">
            <UiIcon name="send" />
          </button>
        </form>
      </aside>
    </div>
  </Transition>
</template>

<style scoped>
.shade { position: fixed; inset: 0; z-index: 20; display: flex; justify-content: flex-end; background: rgba(43, 37, 34, .32); }
.panel { width: 100%; max-width: 440px; height: 100%; background: var(--bg); display: flex; flex-direction: column;
  box-shadow: -12px 0 40px rgba(43, 37, 34, .18); }
.dr-enter-active, .dr-leave-active { transition: background .2s; }
.dr-enter-active .panel, .dr-leave-active .panel { transition: transform .25s ease; }
.dr-enter-from, .dr-leave-to { background: transparent; }
.dr-enter-from .panel, .dr-leave-to .panel { transform: translateX(100%); }

.hd { display: flex; align-items: center; gap: 12px; padding: 16px 16px 14px 20px; border-bottom: 1px solid var(--line);
  padding-top: calc(16px + env(safe-area-inset-top)); }
.ava { width: 44px; height: 44px; border-radius: 50%; background: var(--fab); display: flex; align-items: center; justify-content: center; flex: none; }
.ttl { flex: 1; display: flex; flex-direction: column; line-height: 1.25; }
.ttl .serif { font-size: 20px; }
.ttl .muted { font-size: 12px; }
.close { width: 44px; height: 44px; border: none; border-radius: 50%; background: var(--sand); color: var(--ink); font-size: 22px;
  cursor: pointer; line-height: 1; flex: none; }

.msgs { flex: 1; overflow-y: auto; padding: 18px 16px; display: flex; flex-direction: column; gap: 14px; overscroll-behavior: contain; }
.bot, .user { max-width: 88%; padding: 12px 16px; text-wrap: pretty; white-space: pre-wrap; animation: wsFade .2s; }
.bot { align-self: flex-start; background: var(--cream); border-radius: 20px 20px 20px 6px; }
.user { align-self: flex-end; max-width: 85%; background: var(--wine); color: #fff; border-radius: 20px 20px 6px 20px; }
.bot-wrap { align-self: flex-start; max-width: 92%; width: 92%; display: flex; flex-direction: column; gap: 10px; }
.bot-wrap .bot { max-width: 100%; align-self: flex-start; }
.saved { align-self: flex-start; display: flex; align-items: center; gap: 8px; background: var(--rose); color: var(--wine-deep);
  border-radius: 999px; padding: 6px 14px; font-size: 13px; }
.chips, .quick { display: flex; gap: 8px; flex-wrap: wrap; }
.quick { flex-wrap: nowrap; overflow-x: auto; padding: 0 16px 12px; scrollbar-width: none; }
.chip.sm { flex: none; padding: 7px 14px; font-size: 13px; white-space: nowrap; }
.chip:disabled { opacity: .5; cursor: default; }
.typing { display: flex; gap: 5px; padding: 14px 16px; }
.typing span { width: 7px; height: 7px; border-radius: 50%; background: var(--wine); animation: wsDots 1.2s infinite; }
.err { margin: 0; padding: 0 16px 10px; color: var(--wine); font-size: 14px; }

.bar { display: flex; gap: 8px; align-items: center; padding: 12px 12px 16px; border-top: 1px solid var(--line); background: #fff;
  padding-bottom: calc(16px + env(safe-area-inset-bottom)); }
.bar input { flex: 1; min-width: 0; height: 48px; border: 1px solid var(--line-3); border-radius: 999px; padding: 0 18px;
  font: 400 16px var(--sans); color: var(--ink); background: var(--bg); outline: none; }
.bar input:focus { border-color: var(--wine); }
.mic, .send { width: 48px; height: 48px; flex: none; border: none; border-radius: 50%; cursor: pointer; display: flex;
  align-items: center; justify-content: center; }
.mic { background: var(--sand); color: var(--wine); }
.mic.rec { background: var(--rec); color: #fff; }
.send { background: var(--wine); color: #fff; }
.mic:disabled, .send:disabled { opacity: .5; cursor: default; }
.listening { flex: 1; height: 48px; display: flex; align-items: center; gap: 10px; padding: 0 16px; border-radius: 999px;
  background: var(--rose); color: var(--wine); font-size: 14px; line-height: 1.2; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: var(--rec); flex: none; animation: wsPulse 1s infinite; }
</style>
