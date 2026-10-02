<template>
  <aside class="roster-hud" :class="[side, { collapsed }]" :aria-label="teamLabel">
    <div :id="panelId" class="hud-players" :inert="collapsed" :aria-hidden="collapsed">
      <small class="hud-team">{{ teamLabel }}</small>
      <a v-for="player in orderedRoster" :key="player.position" class="hud-player"
         :class="{ active: isActive(player), finished: isActive(player) && finished }"
         :href="profileUrl(player)" target="_blank" rel="noopener noreferrer"
         :aria-label="`${player.display_name} · ${t('打开对局站个人主页（新标签页）', 'Open player profile (new tab)')}`">
        <span class="hud-avatar" aria-hidden="true">
          <img v-if="player.avatar_url && !failedImages[player.avatar_url]" :src="player.avatar_url" alt=""
               @error="failedImages[player.avatar_url] = true" />
          <span v-else>{{ String(player.display_name || '?').slice(0, 2) }}</span>
        </span>
        <span class="hud-copy"><strong :title="player.display_name">{{ player.display_name }}</strong>
          <small>{{ player.is_captain ? t('队长 · ', 'CPT · ') : '' }}{{ isActive(player) ? (finished ? t('已完成', 'DONE') : t('出战中', 'LIVE')) : t('候场', 'STANDBY') }}</small>
          <small v-if="assignments?.[player.position]" class="hud-project" :title="assignments[player.position]">{{ assignments[player.position] }}</small>
        </span>
      </a>
    </div>
    <button class="hud-toggle" type="button" :aria-controls="panelId" :aria-expanded="!collapsed"
            :aria-label="`${collapsed ? t('展开', 'Show') : t('收起', 'Hide')} ${teamLabel}`"
            :title="`${collapsed ? t('展开', 'Show') : t('收起', 'Hide')} ${teamLabel}`" @click="collapsed = !collapsed">
      <span aria-hidden="true">{{ (side === 'yellow') !== collapsed ? '‹' : '›' }}</span>
    </button>
  </aside>
</template>

<script setup>
import { computed, ref, useId } from 'vue';

const props = defineProps({ side: String, team: Object, activePlayer: Object, assignments: Object, finished: Boolean, lang: String });
const collapsed = defineModel('collapsed', { type: Boolean, default: false });
const failedImages = ref({});
const panelId = useId();
const t = (zh, en) => props.lang === 'zh' ? zh : en;
const teamLabel = computed(() => props.side === 'yellow' ? t('黄方队员', 'Yellow players') : t('白方队员', 'White players'));
const orderedRoster = computed(() => [...(props.team?.roster || [])].sort((a, b) => a.position - b.position));
const isActive = player => Number(player.position) === Number(props.activePlayer?.position);
// Use the same public name-based profile route as HumanPlayContent.
const profileUrl = player => `https://play.2048tables.online/user/${encodeURIComponent(player.display_name || '')}`;
</script>

<style scoped>
.hud-players{max-height:480px;overflow-y:auto;scrollbar-width:thin}
.roster-hud{--accent:var(--match-accent,#e1bd59);position:absolute;z-index:5;top:50%;left:-22px;width:154px;display:flex;align-items:center;transform:translateY(-50%);transition:transform 240ms cubic-bezier(.2,.8,.2,1);pointer-events:none}
.roster-hud.white{--accent:var(--match-copy,#cbd5e1);left:auto;right:-22px;flex-direction:row-reverse}
.roster-hud.collapsed{transform:translate(-130px,-50%)}
.roster-hud.white.collapsed{transform:translate(130px,-50%)}
.hud-players{width:130px;flex:none;display:grid;gap:12px;pointer-events:auto;transition:opacity 160ms ease}
.collapsed .hud-players{opacity:0;pointer-events:none}
.hud-team{padding-inline:10px;font-size:10px;color:var(--match-muted,#aebbd0);letter-spacing:.08em}
.white .hud-team{text-align:right}
.hud-player{display:flex;gap:7px;align-items:center;min-height:66px;padding:6px 6px 6px 8px;border-left:2px solid transparent;color:var(--match-text,#f8fafc);text-decoration:none;background:linear-gradient(90deg,var(--match-hud-fade,rgba(15,23,42,.8)),rgba(15,23,42,0));box-sizing:border-box}
.white .hud-player{flex-direction:row-reverse;padding:6px 8px 6px 6px;border-left:0;border-right:2px solid transparent;background:linear-gradient(270deg,var(--match-hud-fade,rgba(15,23,42,.8)),rgba(15,23,42,0));text-align:right}
.hud-player.active{border-color:var(--accent);background:linear-gradient(90deg,rgba(141,112,42,.25),transparent)}
.white .hud-player.active{background:linear-gradient(270deg,rgba(148,163,184,.22),transparent)}
.hud-avatar{display:grid;place-items:center;flex:0 0 42px;width:42px;height:42px;border:2px solid var(--match-dim,#64748b);border-radius:50%;overflow:hidden;background:var(--match-line,#334155);font-size:12px;font-weight:700;box-sizing:border-box}
.hud-avatar img{width:100%;height:100%;object-fit:cover}
.active .hud-avatar{border-color:var(--accent)}
.hud-copy{display:grid;min-width:0;gap:5px}.hud-copy strong{font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.hud-copy small{font-size:9px;line-height:1.4;color:var(--match-muted,#aebbd0)}.active .hud-copy small{color:var(--accent)}.finished .hud-copy small{color:var(--match-success,#78c59b)}
.hud-player:hover strong{text-decoration:underline}.hud-player:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.hud-copy .hud-project{font-size:10px;color:var(--match-copy,#cbd5e1);overflow-wrap:anywhere}
.hud-toggle{pointer-events:auto;flex:0 0 24px;width:24px;min-height:48px;padding:0;border:1px solid var(--match-border,#475569);border-radius:0 7px 7px 0;background:var(--match-hud-toggle,rgba(15,23,42,.82));color:var(--accent);font-size:24px;cursor:pointer}
.white .hud-toggle{border-radius:7px 0 0 7px}.hud-toggle:hover,.hud-toggle:focus-visible{background:var(--match-line,#334155);outline:2px solid var(--accent);outline-offset:-2px}
@media(prefers-reduced-motion:reduce){.roster-hud,.hud-players{transition:none}}
</style>
