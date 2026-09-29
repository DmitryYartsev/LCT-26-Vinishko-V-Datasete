// Цифровой сомелье: переписка + профиль («цель») живут только на клиенте (localStorage),
// сервер stateless — всё уходит с каждым запросом.
import type { Pick } from "~/utils/wine";

export type Profile = {
  colors?: string[]; sweetness?: string[]; sparkling?: boolean | null; dishes?: string[];
  regions?: string[]; grapes?: string[]; notes?: string[]; exclude_colors?: string[];
  exclude_sweetness?: string[]; exclude_grapes?: string[]; alcohol_max?: number | null;
  occasion?: string | null; summary?: string | null;
};
export type Msg = {
  role: "user" | "assistant"; content: string; picks?: Pick[]; suggestions?: string[];
  goal?: string | null;   // цель, сохранённая на этом ходе (для плашки «Цель сохранена»)
};

const KEY = "svoe-vino-sommelier-v2";
const LIST_LABELS: [keyof Profile, (v: string) => string][] = [
  ["colors", (v) => v], ["sweetness", (v) => v], ["dishes", (v) => `к: ${v.toLowerCase()}`],
  ["regions", (v) => v], ["grapes", (v) => v], ["notes", (v) => v],
  ["exclude_colors", (v) => `не ${v}`], ["exclude_sweetness", (v) => `не ${v}`], ["exclude_grapes", (v) => `без: ${v}`],
];
export const QUICK_PROMPTS = ["Терпкое к стейку", "Белое к рыбе", "Сладкое к десерту", "Игристое на праздник"];

function chipLabels(p: Profile): string[] {
  const out: string[] = [];
  if (p.sparkling === true) out.push("игристое");
  if (p.sparkling === false) out.push("тихое");
  for (const [key, fmt] of LIST_LABELS)
    for (const v of (p[key] as string[] | undefined) || []) out.push(fmt(v));
  if (p.alcohol_max) out.push(`до ${p.alcohol_max}%`);
  if (p.occasion) out.push(p.occasion);
  return out;
}

export function useSommelier() {
  const messages = useState<Msg[]>("som-msgs", () => []);
  const profile = useState<Profile>("som-profile", () => ({}));
  const busy = useState("som-busy", () => false);
  const error = useState<string | null>("som-error", () => null);
  const open = useState("som-open", () => false);
  const ready = useState("som-ready", () => false);

  // один раз на клиенте (из layout)
  function init() {
    if (ready.value) return;
    ready.value = true;
    try {
      const s = JSON.parse(localStorage.getItem(KEY) || "null");
      if (s) { messages.value = s.messages || []; profile.value = s.profile || {}; }
    } catch {}
    watch([messages, profile], () => {
      try { localStorage.setItem(KEY, JSON.stringify({ messages: messages.value, profile: profile.value })); } catch {}
    }, { deep: true });
  }

  const hasGoal = computed(() => chipLabels(profile.value).length > 0);
  const goalTitle = computed(() => {
    const p = profile.value;
    if (p.summary) return p.summary;
    const c = chipLabels(p).join(", ");
    return c ? c[0].toUpperCase() + c.slice(1) : "";
  });
  // для запросов: пустой профиль не шлём вовсе — сервер тогда не считает соответствие
  const goalProfile = computed(() => (hasGoal.value ? profile.value : null));

  async function send(text: string) {
    text = text.trim();
    if (!text || busy.value) return;
    error.value = null;
    messages.value.push({ role: "user", content: text });
    busy.value = true;
    const before = goalTitle.value;
    try {
      const res = await $fetch<{ reply: string; profile: Profile; picks: Pick[]; suggestions?: string[] }>("/api/chat", {
        method: "POST",
        body: { messages: messages.value.map(({ role, content }) => ({ role, content })), profile: profile.value },
      });
      profile.value = res.profile || {};
      const changed = goalTitle.value && goalTitle.value !== before;
      messages.value.push({ role: "assistant", content: res.reply, picks: res.picks, suggestions: res.suggestions,
        goal: changed ? goalTitle.value : null });
    } catch {
      error.value = "Сомелье сейчас недоступен. Попробуйте ещё раз.";
      messages.value.pop();
      return text;   // вернуть текст в поле ввода
    } finally {
      busy.value = false;
    }
  }

  async function transcribe(blob: Blob): Promise<string> {
    const fd = new FormData();
    const ext = blob.type.includes("mp4") ? "m4a" : blob.type.includes("ogg") ? "ogg" : "webm";
    fd.append("audio", blob, `voice.${ext}`);
    const res = await $fetch<{ text: string }>("/api/stt", { method: "POST", body: fd });
    return res.text || "";
  }

  function ask(text: string) { open.value = true; return send(text); }
  function clearGoal() { messages.value = []; profile.value = {}; error.value = null; }

  return { messages, profile, busy, error, open, hasGoal, goalTitle, goalProfile, init, send, ask, transcribe, clearGoal };
}
