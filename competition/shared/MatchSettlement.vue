<script setup>
import { computed, ref } from 'vue';
import { settlementMetric } from './settlementMetric.mjs';
const props = defineProps({ games:{type:Array,default:()=>[]}, teams:{type:Object,default:()=>({})}, score:{type:Object,default:()=>({})}, points:Object, winner:String, reason:String, lang:{type:String,default:'zh'}, dark:Boolean, cancelled:Boolean });
const sides=['yellow','white'];
const failed=ref(new Set());
const t=(zh,en)=>props.lang==='zh'?zh:en;
const name=side=>props.teams[side]?.name || (side==='yellow'?t('黄方','Yellow'):t('白方','White'));
const headline=computed(()=>props.cancelled?t('比赛已取消','MATCH CANCELLED'):props.winner==='draw'?t('全场平局','MATCH DRAW'):sides.includes(props.winner)?`${name(props.winner)}${t('获胜',' wins')}`:t('全场结算','FINAL RESULT'));
const rows=computed(()=>props.games.map(game=>({...game, metrics:Object.fromEntries(sides.map(side=>[side,settlementMetric(game.result,side,game.project,props.lang)]))})));
const resultText=result=>!result?t('未进行','NOT PLAYED'):result.winner_side==='draw'?t('平局','DRAW'):`${name(result.winner_side)}${t('胜',' wins')}`;
const profile=player=>`https://play.2048tables.online/user/${encodeURIComponent(player.display_name)}`;
const avatar=player=>player?.avatar_url?new URL(player.avatar_url,'https://2048tables.online/').href:'';
</script>

<template>
  <section :class="['match-settlement', {dark}]" :aria-label="t('全场结算','Match result')">
    <header class="settlement-heading">
      <div class="settlement-team yellow"><small>{{ t('黄方','YELLOW') }}</small><h2>{{ name('yellow') }}</h2></div>
      <div class="settlement-outcome"><small>MATCH FINISHED</small><h1>{{ headline }}</h1><div class="settlement-total" :aria-label="`${score.yellow || 0} : ${score.white || 0}`"><b>{{ score.yellow || 0 }}</b><span>:</span><b>{{ score.white || 0 }}</b></div></div>
      <div class="settlement-team white"><small>{{ t('白方','WHITE') }}</small><h2>{{ name('white') }}</h2></div>
    </header>
    <p v-if="reason==='late_forfeit'" class="settlement-notice">{{ t('迟到判负 · 三个项目均未实际进行','Late forfeit · no games were played') }}</p>
    <p v-if="points" class="settlement-notice">{{ t('本轮积分','ROUND POINTS') }} · {{ points.yellow }} : {{ points.white }} <small>（{{ t('胜2 · 平1 · 负0','W2 · D1 · L0') }}）</small></p>
    <div class="settlement-games">
      <article v-for="game in rows" :key="game.game_key" class="settlement-row">
        <template v-for="side in sides" :key="side">
          <div :class="['settlement-player',side]">
            <a v-if="game.players?.[side]" :href="profile(game.players[side])" target="_blank" rel="noopener noreferrer" :aria-label="`${game.players[side].display_name} · ${t('个人主页（新标签页）','Profile (new tab)')}`">
              <span class="settlement-avatar"><img v-if="avatar(game.players[side]) && !failed.has(avatar(game.players[side]))" :src="avatar(game.players[side])" alt="" @error="failed.add(avatar(game.players[side]))"/><span v-else>{{ game.players[side].display_name?.slice(0,2) || '?' }}</span></span>
              <strong>{{ game.players[side].display_name }}</strong>
            </a>
            <span v-else class="unassigned">{{ t('未安排出战','No player assigned') }}</span>
          </div>
          <div :class="['settlement-metric',side,{won:game.result?.winner_side===side}]">
            <small>{{ game.metrics[side].label }}</small><strong :class="{dnf:game.metrics[side].value==='DNF'}">{{ game.metrics[side].value }}</strong><span>{{ game.metrics[side].note || '\u00a0' }}</span>
          </div>
        </template>
        <div class="settlement-project">
          <small>{{ t('项目','GAME') }} {{ game.game_key }}</small>
          <img v-if="game.icon" :src="game.icon" alt="" />
          <strong>{{ game.name || t('未选定项目','Project not selected') }}</strong>
          <span :class="['settlement-verdict',game.result?.winner_side]"><b aria-hidden="true">{{ game.result?.winner_side==='yellow'?'◀':game.result?.winner_side==='white'?'▶':'—' }}</b> {{ resultText(game.result) }}</span><small v-if="game.result?.record_eligible_side">{{ t('获胜队成绩可参评纪录','Winning team result eligible for records') }}</small>
        </div>
      </article>
    </div>
    <footer v-if="$slots.default" class="settlement-footer"><slot /></footer>
  </section>
</template>

<style scoped>
.match-settlement{--ink:#40372d;--muted:#82796c;--line:#e5ddd0;--surface:#fffdf8;--accent:#a18145;--tint:#f4f0e8;color:var(--ink);background:var(--surface);width:100%;max-width:1180px;margin:0 auto;padding:24px 28px;box-sizing:border-box;border:1px solid var(--line);border-radius:8px;container-type:inline-size}.match-settlement.dark{--ink:#f1f5f9;--muted:#9baabd;--line:#334155;--surface:#111c30;--accent:#dfc16d;--tint:#19273c;border:0;border-radius:0}
.settlement-heading{display:grid;grid-template-columns:1fr 1.1fr 1fr;gap:20px;align-items:center;text-align:center;padding:0 0 20px}.settlement-team h2{font-size:clamp(22px,3vw,36px);margin:8px 0;overflow-wrap:anywhere;line-height:1.2}.settlement-team small,.settlement-outcome>small{font-size:11px;letter-spacing:.16em;color:var(--muted)}.settlement-team.yellow h2{color:var(--accent)}.settlement-outcome h1{font-size:clamp(22px,2.4vw,30px);line-height:1.25;margin:6px 0 10px;overflow-wrap:anywhere}.settlement-total{display:flex;justify-content:center;align-items:center;gap:22px;font-size:46px;line-height:1;font-variant-numeric:tabular-nums}.settlement-total b:first-child{color:var(--accent)}.settlement-total>span{color:var(--muted);font-size:28px}
.settlement-row{display:grid;grid-template-columns:minmax(90px,1fr) minmax(110px,1.35fr) minmax(170px,1.5fr) minmax(110px,1.35fr) minmax(90px,1fr);align-items:center;gap:12px;padding:16px 0;border-top:1px solid var(--line);min-width:0}.settlement-player.yellow{grid-column:1;grid-row:1}.settlement-metric.yellow{grid-column:2;grid-row:1}.settlement-project{grid-column:3;grid-row:1}.settlement-metric.white{grid-column:4;grid-row:1}.settlement-player.white{grid-column:5;grid-row:1}.settlement-player,.settlement-metric,.settlement-project{min-width:0;text-align:center}.settlement-player a{display:flex;flex-direction:column;align-items:center;gap:10px;color:inherit;text-decoration:none}.settlement-player a:hover strong{text-decoration:underline}.settlement-player strong{font-size:14px;overflow-wrap:anywhere;line-height:1.4}.settlement-avatar{display:grid;place-items:center;flex:none;width:66px;height:66px;border-radius:50%;overflow:hidden;background:var(--tint);border:2px solid var(--line);font-size:20px}.yellow .settlement-avatar{border-color:var(--accent)}.settlement-avatar img{width:100%;height:100%;object-fit:cover}.settlement-metric{display:grid;gap:6px}.settlement-metric small{font-size:12px;color:var(--muted)}.settlement-metric>strong{font-size:clamp(24px,3.3vw,42px);font-variant-numeric:tabular-nums;letter-spacing:-.04em;line-height:1.2;white-space:nowrap}.settlement-metric.won>strong{color:var(--accent)}.settlement-metric>span,.unassigned{font-size:12px;color:var(--muted)}.settlement-metric .dnf{color:var(--muted);letter-spacing:.04em}.settlement-project{display:grid;justify-items:center;gap:5px;padding:8px;border-radius:6px;background:var(--tint)}.settlement-project>small{font-size:10px;letter-spacing:.12em;color:var(--muted)}.settlement-project img{width:76px;height:76px;object-fit:contain}.settlement-project>strong{font-size:13px;line-height:1.35;overflow-wrap:anywhere}.settlement-verdict{font-size:12px;line-height:1.4;color:var(--muted)}.settlement-verdict.yellow{color:var(--accent)}.settlement-verdict.white{color:var(--ink)}.settlement-notice{text-align:center;color:var(--muted)}.settlement-footer{text-align:center;border-top:1px solid var(--line);padding-top:18px;margin-top:2px}
@container(max-width:720px){.settlement-row{grid-template-columns:minmax(48px,.7fr) minmax(72px,1fr) minmax(102px,1.15fr) minmax(72px,1fr) minmax(48px,.7fr);gap:5px;padding:12px 0}.settlement-avatar{width:44px;height:44px;font-size:16px}.settlement-player strong{font-size:11px}.settlement-metric>strong{font-size:22px}.settlement-project img{width:60px;height:60px}.settlement-heading{gap:8px}.settlement-team h2{font-size:23px}.settlement-outcome h1{font-size:21px}.settlement-total{font-size:36px}}
@container(max-width:470px){.settlement-row{grid-template-columns:1fr 112px 1fr;gap:5px 8px;align-items:start;padding:14px 0}.settlement-player.yellow{grid-column:1;grid-row:1}.settlement-player.white{grid-column:3;grid-row:1}.settlement-metric.yellow{grid-column:1;grid-row:2}.settlement-metric.white{grid-column:3;grid-row:2}.settlement-project{grid-column:2;grid-row:1/3;align-self:center;padding:8px 4px}.settlement-player a{gap:4px}.settlement-player strong{font-size:11px}.settlement-metric{gap:3px}.settlement-metric>strong{font-size:20px}.settlement-metric small,.settlement-metric>span{font-size:10px}.settlement-team h2{font-size:17px}.settlement-outcome h1{font-size:18px}.settlement-total{font-size:30px;gap:12px}.settlement-project>strong{font-size:11px}.settlement-project img{width:62px;height:62px}}
@media(max-width:600px){.match-settlement{padding:16px 10px}}
</style>
