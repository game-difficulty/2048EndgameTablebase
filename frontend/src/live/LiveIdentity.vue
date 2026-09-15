<template>
  <span :class="['live-identity', `supporter-${level}`]">
    <AccountAvatar :user="{ display_name: actor.name, profile: { avatar_url: actor.avatar_url } }" :supporter="level > 0" />
    <strong>{{ actor.name }}</strong>
    <span v-if="level > 0" class="identity-emblem" :title="label" :aria-label="label" role="img">
      <Crown v-if="level === 2" :size="15" :stroke-width="1.8" /><Gem v-else :size="12" :stroke-width="1.8" />
    </span>
  </span>
</template>
<script setup>
import { computed } from 'vue';
import { Crown, Gem } from '@lucide/vue';
import AccountAvatar from '../features/auth/AccountAvatar.vue';
import { liveSupporterLevel } from './supporterIdentity.js';
const props = defineProps({ actor: { type: Object, required: true }, lang: String });
const level = computed(() => liveSupporterLevel(props.actor));
const label = computed(() => props.lang === 'zh' ? (level.value === 2 ? '荣耀赞助者' : '赞助者') : (level.value === 2 ? 'Gold Supporter' : 'Supporter'));
</script>
<style scoped>
.live-identity { display:inline-flex;align-items:center;gap:7px;min-width:0;max-width:100%;color:var(--text-main); }
strong { overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;font-size:inherit; }
.identity-emblem { display:inline-flex;flex:0 0 auto; }
.supporter-1 { color:#167865; }.supporter-2 { color:#8b5b12; }
:global(:root[data-theme='dark'] .live-page .live-identity.supporter-1) { color:#66d5b7; }
:global(:root[data-theme='dark'] .live-page .live-identity.supporter-2) { color:#edc86b; }
.supporter-2 :deep(.account-avatar-shell) { color:inherit;border-color:#b99240;box-shadow:0 0 0 2px #d6b46120; }
.supporter-2 :deep(.account-supporter-mark) { background:#d6b461; }
</style>
