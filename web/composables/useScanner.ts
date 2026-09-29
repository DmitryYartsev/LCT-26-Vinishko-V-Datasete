// Сканирование: фото -> /api/scan -> карточка (/wine/:slug) или «не найдено» (/no-match).
// Сканы за сессию — slug'и в sessionStorage (живут, пока открыта вкладка: один поход в магазин).
import type { SimilarItem, Pick } from "~/utils/wine";

export type ScanResult = {
  in_catalog: boolean; elapsed_ms?: number;
  confidence: { top1_score: number; margin: number };
  top1: { slug: string; score: number } | null;
  similar?: SimilarItem[]; analogs?: Pick[];
};
export type LastScan = ScanResult & { photo: string; seconds: number };

const KEY = "svoe-vino-scans-v1";
const MAX_HISTORY = 30;
// Фото с телефона (4000px, 3–8 МБ) через медленную сеть/VPN не долетает: соединение рвётся
// посреди загрузки. Модели столько не нужно (SigLIP — 256px, OCR-кроп этикетки ≤1400px),
// поэтому ужимаем до 2560px по длинной стороне — выходит ~0.5–1.5 МБ.
const UPLOAD_MAX_SIDE = 2560;
const UPLOAD_QUALITY = 0.88;

async function shrink(file: Blob): Promise<Blob> {
  try {
    const bmp = await createImageBitmap(file); // EXIF-поворот применяется сам
    const k = Math.min(1, UPLOAD_MAX_SIDE / Math.max(bmp.width, bmp.height));
    if (k === 1 && file.size < 1.5e6) { bmp.close(); return file; }
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bmp.width * k);
    canvas.height = Math.round(bmp.height * k);
    canvas.getContext("2d")!.drawImage(bmp, 0, 0, canvas.width, canvas.height);
    bmp.close();
    const out = await new Promise<Blob | null>((ok) => canvas.toBlob(ok, "image/jpeg", UPLOAD_QUALITY));
    return out && out.size < file.size ? out : file;
  } catch {
    return file; // не смогли декодировать (напр. HEIC) — шлём как есть
  }
}

export function useScanner() {
  const history = useState<string[]>("scan-history", () => []);
  const last = useState<LastScan | null>("scan-last", () => null);
  const busy = useState("scan-busy", () => false);
  const photo = useState<string | null>("scan-photo", () => null);
  const error = useState<string | null>("scan-error", () => null);
  const catalogSize = useState<number | null>("scan-catalog", () => null);
  const ready = useState("scan-ready", () => false);

  function init() {
    if (ready.value) return;
    ready.value = true;
    try { history.value = JSON.parse(sessionStorage.getItem(KEY) || "[]"); } catch {}
    watch(history, () => { try { sessionStorage.setItem(KEY, JSON.stringify(history.value)); } catch {} }, { deep: true });
    $fetch<{ wines: number }>("/api/stats").then((s) => (catalogSize.value = s.wines || null)).catch(() => {});
  }

  function remember(slug: string) {
    history.value = [...history.value.filter((s) => s !== slug), slug].slice(-MAX_HISTORY);
  }

  async function scan(file: Blob | null | undefined, profile: object | null) {
    if (!file || !file.type.startsWith("image/") || busy.value) return;
    error.value = null;
    if (photo.value?.startsWith("blob:")) URL.revokeObjectURL(photo.value);
    photo.value = URL.createObjectURL(file);
    busy.value = true;
    if (useRouter().currentRoute.value.path !== "/") await navigateTo("/");
    const t0 = performance.now();
    try {
      const fd = new FormData();
      fd.append("image", await shrink(file), "photo.jpg");
      if (profile) fd.append("profile", JSON.stringify(profile));
      // один повтор — только при обрыве сети (ответа нет вовсе), ошибки сервера не повторяем
      const post = () => $fetch<ScanResult>("/api/scan", { method: "POST", body: fd });
      const res = await post().catch((e) => (e?.response ? Promise.reject(e) : post()));
      last.value = { ...res, photo: photo.value!, seconds: (performance.now() - t0) / 1000 };
      if (res.in_catalog && res.top1) {
        remember(res.top1.slug);
        await navigateTo(`/wine/${encodeURIComponent(res.top1.slug)}`);
      } else {
        await navigateTo("/no-match");
      }
    } catch {
      error.value = "Не удалось распознать фото. Попробуйте ещё раз.";
    } finally {
      busy.value = false;
    }
  }

  return { history, last, busy, photo, error, catalogSize, init, scan, remember };
}
