<template>
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path v-for="(path, i) in paths[name] ?? paths.plane" :key="i" :d="path" />
  </svg>
</template>

<script setup lang="ts">
defineProps<{ name: string }>()
const paths: Record<string, string[]> = {
  plane: ['m22 2-7 20-4-9-9-4L22 2Z', 'M22 2 11 13'],
  plus: ['M12 5v14M5 12h14'],
  chat: ['M21 11.5a8.5 8.5 0 0 1-8.5 8.5H4l-2 2v-10a9.5 9.5 0 0 1 19 0Z', 'M7 9h9M7 13h6'],
  search: ['M16.5 16.5 21 21', 'M18 10.5a7.5 7.5 0 1 1-15 0 7.5 7.5 0 0 1 15 0Z'],
  ticket: ['M3 6h18v4a2 2 0 0 0 0 4v4H3v-4a2 2 0 0 0 0-4V6Z', 'M15 6v2m0 3v2m0 3v2'],
  exchange: ['M4 7h16m-4-4 4 4-4 4', 'M20 17H4m4-4-4 4 4 4'],
  book: ['M12 5v16', 'M12 5C8 2 4 3 2 4v15c4-2 7-1 10 2 3-3 6-4 10-2V4c-2-1-6-2-10 1Z'],
  arrow: ['M5 12h14m-5-5 5 5-5 5'],
  arrowUp: ['M12 19V5m-6 6 6-6 6 6'],
  arrowDown: ['M12 5v14m-6-6 6 6 6-6'],
  chevron: ['m9 5 7 7-7 7'],
  check: ['m5 12 4 4L19 6'],
  shield: ['M12 3 3 7v5c0 5 9 10 9 10s9-5 9-10V7l-9-4Z', 'm8 12 3 3 5-6'],
  clock: ['M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z', 'M12 7v5l3 2'],
  globe: ['M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z', 'M3 12h18M12 3c5 5 5 13 0 18-5-5-5-13 0-18Z'],
  menu: ['M4 6h16M4 12h16M4 18h16'],
  close: ['m6 6 12 12M6 18 18 6'],
  layers: ['m12 3 10 5-10 5L2 8l10-5Z', 'm2 12 10 5 10-5M2 16l10 5 10-5'],
  copy: ['M9 9h12v12H9z', 'M15 9V3H3v12h6'],
  alert: ['M12 3 1 21h22L12 3Z', 'M12 9v5m0 3v.1'],
  refresh: ['M20 7v5h-5', 'M4 17v-5h5', 'M6 6a8 8 0 0 1 13 2l1 4M4 12l1 4a8 8 0 0 0 13 2'],
  user: ['M16 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z', 'M4 21v-2a8 8 0 0 1 16 0v2'],
}
</script>
