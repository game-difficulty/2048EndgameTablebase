<script setup>
import { computed } from 'vue';
import PlayerAvatar from './PlayerAvatar.vue';
import { t } from './i18n.js';

const props = defineProps({ seats: { type: Array, default: () => [] }, removedMembers: { type: Array, default: () => [] }, currentUserId: Number, busy: Boolean });
const userId = defineModel('userId', { default: '' });
const emit = defineEmits(['remove', 'restore']);
const teams = computed(() => ['yellow', 'white'].map(side => ({ side, label: side === 'yellow' ? '黄方' : '白方', seats: props.seats.filter(seat => seat.side === side).slice().sort((a, b) => a.position - b.position) })));
const validId = computed(() => /^[1-9]\d*$/.test(String(userId.value)) && Number.isSafeInteger(Number(userId.value)) && Number(userId.value) !== props.currentUserId);
</script>

<template>
  <div class="member-management">
    <p class="member-policy">{{ t('管理员可管理任意房间，房主仅限自建房间。赛中移出选手将暂停比赛，历史成绩保留。') }}</p>
    <div class="member-teams">
      <section v-for="team in teams" :key="team.side" :class="['member-team', team.side]">
        <h3>{{ t(team.label) }}<span>{{ team.seats.length }}</span></h3>
        <ul>
          <li v-for="seat in team.seats" :key="seat.user_id" class="member-row">
            <PlayerAvatar :person="seat" />
            <div class="member-identity"><strong>{{ seat.display_name }}</strong><small>ID {{ seat.user_id }}<span v-if="seat.user_id === currentUserId"> · {{ t('你自己') }}</span></small></div>
            <button class="member-remove" type="button" :disabled="busy || seat.user_id === currentUserId" :aria-label="`${t('移出')} ${seat.display_name} (ID ${seat.user_id})`" @click="emit('remove', seat.user_id)">{{ t('移出') }}</button>
          </li>
        </ul>
        <p v-if="!team.seats.length" class="member-empty">{{ t('暂无已落座选手') }}</p>
      </section>
    </div>
    <form class="member-manual" @submit.prevent="validId && !busy && emit('remove', userId)">
      <label><span>{{ t('其他人员的用户 ID') }}</span><input v-model="userId" type="number" inputmode="numeric" min="1" step="1" required :disabled="busy" :placeholder="t('输入用户 ID')" /></label>
      <button class="member-remove" type="submit" :disabled="busy || !validId">{{ t('移出此人') }}</button>
    </form>
    <section v-if="removedMembers.length" class="member-removed">
      <h3>{{ t('已移出人员') }}<span>{{ removedMembers.length }}</span></h3>
      <ul><li v-for="member in removedMembers" :key="member.user_id" class="member-row"><div class="member-identity"><strong>{{ member.display_name || `ID ${member.user_id}` }}</strong><small v-if="member.display_name">ID {{ member.user_id }}</small></div><button type="button" :disabled="busy" @click="emit('restore', member.user_id)">{{ t('允许重新进入') }}</button></li></ul>
    </section>
  </div>
</template>

<style scoped>
.member-management{padding:0 14px 14px;font-size:14px;line-height:1.5;color:var(--competition-text,#51473d)}
.member-policy{margin:14px 0 18px;padding:12px 14px;border-left:3px solid #b3905b;border-radius:4px;background:light-dark(#f4efe7,#263247);color:var(--competition-muted,#75644f)}
.member-teams{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.member-team{min-width:0;border:1px solid var(--competition-border,#e4ddd2);border-radius:10px;overflow:hidden;background:var(--competition-card,#fffdf9)}
h3{display:flex;align-items:center;gap:8px;margin:0;font-size:14px;font-weight:700}
.member-team h3{padding:11px 14px;border-bottom:1px solid var(--competition-border,#e4ddd2);background:light-dark(#f3f0eb,#2b374b)}
.member-team.yellow h3{background:light-dark(#f7efd8,#3b3527);color:light-dark(#87671f,#e8cb7c)}
h3>span{display:inline-grid;place-items:center;min-width:22px;height:22px;padding:0 4px;box-sizing:border-box;border-radius:12px;background:light-dark(#ffffffa6,#ffffff12);font-size:12px;color:var(--competition-muted,#736754)}
ul{margin:0;padding:0;list-style:none}
.member-row{display:flex;align-items:center;gap:12px;min-width:0;padding:12px 14px}
.member-row+.member-row{border-top:1px solid var(--competition-border,#eee8df)}
.member-row :deep(.player-avatar){--player-avatar-size:36px}
.member-identity{flex:1;min-width:0;display:grid;gap:3px}
.member-identity strong{font-size:14px;font-weight:600;overflow-wrap:anywhere}
.member-identity small{font-size:12px;color:var(--competition-muted,#8b8072);font-variant-numeric:tabular-nums}
button{flex:none;min-height:40px;padding:8px 12px;border:1px solid var(--competition-border,#cfc4b5);border-radius:6px;background:var(--competition-card,#fffdf9);color:var(--competition-text,#68583e);font:inherit;cursor:pointer;white-space:normal}
button.member-remove{color:light-dark(#a54840,#f2aaa0);border-color:light-dark(#dfc4be,#78504e);background:light-dark(#fff8f6,#382b31)}
button:hover:not(:disabled){background:light-dark(#f4ede3,#334155)}
button.member-remove:hover:not(:disabled){background:light-dark(#fbeae5,#4c3036)}
button:disabled{opacity:.45;cursor:not-allowed}
button:focus-visible,input:focus-visible{outline:2px solid #a18145;outline-offset:3px}
.member-manual{display:flex;align-items:flex-end;gap:12px;margin-top:18px;padding-top:18px;border-top:1px solid var(--competition-border,#e4ddd2)}
.member-manual label{flex:1;min-width:0;display:grid;gap:7px;font-size:13px;color:var(--competition-muted,#75644f)}
input{box-sizing:border-box;width:100%;min-width:0;min-height:42px;padding:8px 12px;border:1px solid var(--competition-border,#d6cdc0);border-radius:6px;background:var(--competition-card,#fffdf9);color:var(--competition-text,#51473d);font:inherit}
.member-empty{margin:0;padding:18px 14px;color:var(--competition-muted,#8b8072)}
.member-removed{margin-top:20px;padding-top:16px;border-top:1px solid var(--competition-border,#e4ddd2)}
.member-removed h3{margin-bottom:8px}
.member-removed .member-row{padding-inline:0}
@media(max-width:700px){.member-teams{grid-template-columns:1fr;gap:12px}.member-row{gap:10px;padding:12px}.member-policy{font-size:13px}.member-manual{gap:10px}}
@media(max-width:360px){.member-manual{flex-wrap:wrap}.member-manual label,.member-manual button{width:100%;flex-basis:100%}}
</style>
