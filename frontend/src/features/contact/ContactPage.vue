<template>
  <main class="contact-page">
    <h1>{{ $t('menu.contact') }}</h1>
    <div class="contact-list">
      <div class="contact-row">
        <span class="contact-label">{{ $t('menu.qqGroup') }}</span>
        <span class="contact-value">1108338721</span>
      </div>
      <div class="contact-row">
        <span class="contact-label">Bilibili</span>
        <a href="https://space.bilibili.com/123773889" target="_blank" rel="noopener noreferrer">
          https://space.bilibili.com/123773889 <ExternalLink :size="15" aria-hidden="true" />
        </a>
      </div>
      <div class="contact-row">
        <span class="contact-label">Discord</span>
        <a href="https://discord.gg/Ru4GPUDMZ" target="_blank" rel="noopener noreferrer">
          https://discord.gg/Ru4GPUDMZ <ExternalLink :size="15" aria-hidden="true" />
        </a>
      </div>
    </div>
  </main>
</template>

<script setup>
import { ExternalLink } from '@lucide/vue';
</script>

<style scoped>
.contact-page { max-width:960px;height:100%;margin:0 auto;padding:32px;overflow:auto;color:var(--text-main); }
h1 { font-size:24px;margin:0 0 28px; }
.contact-list { max-width:680px;border-top:1px solid var(--border-main); }
.contact-row { display:grid;grid-template-columns:120px minmax(0,1fr);gap:16px;align-items:center;min-height:68px;padding:16px 0;border-bottom:1px solid var(--border-main); }
.contact-label { color:var(--text-secondary);font-weight:700; }
.contact-value { font-weight:700;user-select:text; }
a { display:inline-flex;align-items:center;gap:6px;width:fit-content;max-width:100%;color:var(--accent);font-weight:700;text-decoration:none;overflow-wrap:anywhere; }
a:hover { text-decoration:underline; }
a:focus-visible { outline:2px solid var(--accent);outline-offset:3px; }
a svg { flex-shrink:0; }
@media (max-width:520px) { .contact-page { padding:24px 18px; }.contact-row { grid-template-columns:1fr;gap:6px; } }
</style>
