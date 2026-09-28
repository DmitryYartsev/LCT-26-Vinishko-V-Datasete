<script setup lang="ts">
// Линейные иконки из макета (stroke = currentColor).
defineProps<{ name: "camera" | "upload" | "chat" | "target" | "check" | "mic" | "send" | "close" | "minus"; size?: number }>();
</script>

<template>
  <svg :width="size || 20" :height="size || 20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"
       :stroke-width="name === 'check' || name === 'minus' ? 2.6 : 1.8" style="flex: none">
    <template v-if="name === 'camera'"><path d="M4 8h3l2-3h6l2 3h3v11H4z" /><circle cx="12" cy="13" r="3.5" /></template>
    <path v-else-if="name === 'upload'" d="M12 16V4M7 9l5-5 5 5M4 16v4h16v-4" />
    <template v-else-if="name === 'chat'">
      <path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z" /><path d="M8.5 10.5h7M8.5 13.5h4.5" />
    </template>
    <template v-else-if="name === 'target'">
      <circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="5" /><circle cx="12" cy="12" r="1.5" fill="currentColor" />
    </template>
    <path v-else-if="name === 'check'" d="M5 12l5 5 9-10" />
    <path v-else-if="name === 'minus'" d="M6 12h12" />
    <template v-else-if="name === 'mic'"><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></template>
    <path v-else-if="name === 'send'" d="M5 12h14M13 6l6 6-6 6" />
    <path v-else-if="name === 'close'" d="M6 6l12 12M18 6L6 18" />
  </svg>
</template>
