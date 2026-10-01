<template>
  <div class="lobby-shell">
    <header><a class="brand" href="https://2048tables.online/"><Radio :size="25" />2048 <b>LIVE</b></a><div><span>{{ t('直播大厅','LIVE LOBBY') }}</span><button @click="lang = lang === 'zh' ? 'en' : 'zh'" aria-label="Language">{{ lang==='zh'?'EN':'中' }}</button><button @click="load">{{ t('刷新','Refresh') }}</button></div></header>
    <main>
      <div class="intro"><div><small>{{ t('直播大厅','LIVE LOBBY') }}</small><h1>{{ t('正在直播','Live now') }}</h1><p>{{ t('观看赛事、玩家和 AI 正在进行的 2048 对局。','Watch live tournaments, players and AI games.') }}</p></div><span class="room-count"><i></i>{{ rooms.length }} {{ t('个直播间','rooms') }}</span></div>
      <p v-if="error" class="error">{{ t('直播大厅暂时无法更新。','The lobby could not be refreshed.') }}</p>
      <section v-if="rooms.length" class="room-grid">
        <a v-for="room in rooms" :key="room.id" :href="room.path" class="room-card">
          <div class="preview">
            <div v-if="room.content_kind==='competition-match'" class="competition-cover">
              <div v-for="side in ['yellow','white']" :key="side" class="cover-team" :class="side">
                <small>{{ side==='yellow'?t('黄方','Yellow'):t('白方','White') }}</small>
                <div class="cover-avatars"><PlayerAvatar v-for="person in room.preview?.teams?.[side]?.roster || []" :key="person.position" :person="person" :title="person.display_name" /></div>
                <strong :title="rosterNames(room, side)">{{ rosterNames(room, side) }}</strong>
                <b>{{ room.preview?.[side+'_score'] || 0 }}</b>
              </div>
              <div class="cover-center"><img v-if="projectIconUrl(room.preview?.current_project?.project_ref,'dark')" :src="projectIconUrl(room.preview.current_project.project_ref,'dark')" alt="" /><b v-else>VS</b></div>
              <div class="cover-footer"><span>{{ phase(room.preview?.phase) }}</span><strong>{{ competitionProjectLabel(room.preview?.current_project,lang) || (room.preview?.prediction_open?t('赛事下注开放','Predictions open'):t('项目待定','Projects pending')) }}</strong></div>
            </div>
            <LiveBoardThumbnail v-else :board="room.board" :variant="room.variant" :appearance="room.appearance" />
            <em><i></i>{{ localized(room.badge) || (room.content_kind === 'human-play' ? t('玩家直播','PLAYER') : t('AI 直播','AI')) }}</em>
          </div>
          <div class="card-copy"><div class="identity"><img v-if="room.streamer?.avatar_url" :src="room.streamer.avatar_url" alt="" /><span v-else>{{ room.content_kind === 'competition-match' ? 'VS' : room.content_kind === 'human-play' ? 'P' : 'AI' }}</span><div><strong>{{ title(room) }}</strong><small>{{ localized(room.subtitle) || `${room.variant} · ${Number(room.score || 0).toLocaleString()}` }}</small></div></div><small class="watch"><Users :size="15" />{{ room.viewers }}</small></div>
        </a>
      </section>
      <div v-else-if="!error" class="empty">{{ t('当前没有正在直播的对局。','No streams are live right now.') }}</div>
    </main>
  </div>
</template>
<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue';
import { liveLanguage, saveLiveLanguage } from './language.js';
import { Radio, Users } from '@lucide/vue';
import LiveBoardThumbnail from './LiveBoardThumbnail.vue';
import PlayerAvatar from '../../../competition/frontend/src/PlayerAvatar.vue';
import { projectIconUrl } from '../../../competition/shared/projectIcons.js';
import { competitionPhaseLabel, competitionProjectLabel } from '../../../competition/shared/projectLabels.mjs';
const lang = ref(liveLanguage());
watch(lang, saveLiveLanguage);
const rooms = ref([]), error = ref(false); let timer, controller, stopped = false;
const t = (zh,en) => lang.value === 'zh' ? zh : en;
const title = room => room.title?.[lang.value] || room.title?.en || room.streamer?.display_name || room.id;
const localized = value => value?.[lang.value] || value?.en || value?.zh || '';
const phase = value => competitionPhaseLabel(value,lang.value);
const rosterNames=(room,side)=>(room.preview?.teams?.[side]?.roster||[]).map(person=>person.display_name).join(' / ') || t('等待队员','Waiting for players');
async function load() {
  if (stopped || document.hidden || controller) return;
  const request = new AbortController(); controller = request;
  const timeout = setTimeout(() => request.abort(), 8000);
  try {
    const response = await fetch('/api/live/lobby', { cache: 'no-store', signal: request.signal });
    if (!response.ok) throw Error();
    const data = await response.json();
    if (!stopped && controller === request) { rooms.value = data.rooms || []; error.value = false; }
  } catch {
    if (!stopped && controller === request && !document.hidden) error.value = true;
  } finally {
    clearTimeout(timeout);
    if (controller === request) controller = null;
  }
}
function visibility() {
  if (document.hidden) { controller?.abort(); controller = null; }
  else void load();
}
onMounted(()=>{document.documentElement.dataset.theme='dark';load();timer=setInterval(load,10000);document.addEventListener('visibilitychange',visibility);});
onUnmounted(()=>{stopped=true;clearInterval(timer);controller?.abort();controller=null;document.removeEventListener('visibilitychange',visibility);});
</script>
<style scoped>
.lobby-shell{min-height:100vh;background:radial-gradient(circle at 14% -8%,#294064 0,transparent 34%),radial-gradient(circle at 88% 18%,#172a45 0,transparent 27%),var(--bg-main);color:var(--text-main)}
header{height:72px;padding:0 clamp(22px,4vw,68px);display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #ffffff12;background:#0b1323cc;box-shadow:0 1px 0 #0004;backdrop-filter:blur(16px)}header>div,.brand{display:flex;align-items:center;gap:18px}header>div>span{color:var(--text-secondary);font-size:12px;font-weight:800;letter-spacing:.12em}.brand{font-size:23px;font-weight:800;color:#fff;text-decoration:none}.brand b{color:#fb7185}button{border:1px solid #ffffff1c;border-radius:8px;background:#ffffff0a;color:var(--text-main);padding:8px 14px;cursor:pointer;transition:border-color .16s,background .16s}button:hover{border-color:#38bdf866;background:#38bdf80d}
main{max-width:1280px;margin:auto;padding:54px clamp(22px,4vw,68px) 90px}.intro{display:flex;align-items:end;justify-content:space-between;gap:28px;margin-bottom:30px}.intro>div>small{display:block;margin-bottom:8px;color:#67c9f4;font-size:11px;font-weight:800;letter-spacing:.18em}.intro h1{font-size:clamp(34px,4vw,48px);line-height:1.08;margin:0 0 10px;letter-spacing:-.025em}.intro p{margin:0;color:#9fb0c8;font-size:15px}.room-count{display:inline-flex;align-items:center;gap:8px;flex:0 0 auto;padding:8px 12px;border:1px solid #ffffff14;border-radius:999px;background:#ffffff08;color:#b6c3d6;font-size:12px}.room-count i{width:7px;height:7px;border-radius:50%;background:#34d399;box-shadow:0 0 10px #34d399aa}
.room-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:24px}.room-card{overflow:hidden;border:1px solid #ffffff16;border-radius:15px;background:linear-gradient(145deg,#19263b,#152136);color:inherit;text-decoration:none;box-shadow:0 15px 38px #02081726;transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease}.room-card:hover{transform:translateY(-4px);border-color:#38bdf870;box-shadow:0 20px 45px #02081755,0 0 0 1px #38bdf815}
.preview{position:relative;display:grid;place-items:center;aspect-ratio:16/9;background:radial-gradient(circle at 50% 44%,#1c2d47 0,#0c1525 70%);overflow:hidden}.preview:after{content:"";position:absolute;inset:auto 0 0;height:42%;background:linear-gradient(transparent,#07101d55);pointer-events:none}.preview em{z-index:2;position:absolute;left:13px;top:13px;display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border:1px solid #ffffff14;border-radius:999px;background:#020817c9;color:#e8eef7;font-size:10px;font-weight:800;font-style:normal;letter-spacing:.08em;backdrop-filter:blur(8px)}.preview em i{width:6px;height:6px;border-radius:50%;background:#fb7185;box-shadow:0 0 8px #fb7185}
.competition-preview{display:grid;place-items:center;gap:9px;color:#f8fafc}.competition-preview>small{color:#d8bd69;font-weight:800;letter-spacing:.14em}.competition-preview>strong{display:flex;align-items:center;gap:18px;font-size:48px}.competition-preview>strong i{color:#64748b;font-size:26px;font-style:normal}.competition-preview>span{color:#94a3b8;font-size:12px;letter-spacing:.08em}
.competition-cover{position:absolute;inset:0;display:grid;grid-template-columns:1fr 1fr;gap:52px;padding:42px 16px 42px;color:#f1f5f9;background:linear-gradient(100deg,#49401c44,transparent 48%,#cbd5e112)}.cover-team{min-width:0;text-align:center;display:flex;flex-direction:column;align-items:center;gap:4px}.cover-team>small{font-size:11px;color:#b8c5d8}.cover-team.yellow>small,.cover-team.yellow>b{color:#eacb68}.cover-team>strong{max-width:100%;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.cover-team>b{font-size:26px;line-height:1}.cover-avatars{display:flex;justify-content:center;height:29px;gap:3px}.cover-avatars :deep(.player-avatar){width:28px;height:28px;flex:0 0 28px;font-size:10px;object-fit:cover;border-radius:50%;background:#64748b55;border:1px solid #ffffff33}.cover-center{position:absolute;inset:42px calc(50% - 27px) 40px;display:grid;place-items:center}.cover-center img{width:58px;height:58px;object-fit:contain}.cover-center b{color:#91a2ba;font-size:17px}.cover-footer{position:absolute;bottom:10px;left:14px;right:14px;display:flex;gap:8px;justify-content:space-between;font-size:10px;color:#b7c6d8;z-index:1}.cover-footer strong{overflow:hidden;white-space:nowrap;text-overflow:ellipsis;text-align:right}
.card-copy{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:17px 18px}.identity{display:flex;align-items:center;gap:12px;min-width:0}.identity>img,.identity>span{width:42px;height:42px;flex:0 0 42px;border:1px solid #ffffff20;border-radius:12px;object-fit:cover;background:linear-gradient(145deg,#38bdf8,#2563eb);color:#fff;display:grid;place-items:center;font-size:13px;font-weight:800;box-shadow:0 7px 18px #02081745}.identity div{display:grid;gap:4px;min-width:0}.identity strong{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:15px}.identity small,.watch{color:#91a2ba}.identity small{font-size:12px}.watch{display:flex;gap:5px;align-items:center;flex:0 0 auto;font-size:12px}.empty,.error{padding:60px;text-align:center;border:1px dashed var(--border-main);border-radius:15px;background:#ffffff05;color:var(--text-secondary)}
@media(max-width:600px){header{padding-inline:18px}header>div>span{display:none}main{padding:38px 18px 70px}.intro{align-items:flex-start;flex-direction:column;gap:18px}.room-grid{grid-template-columns:1fr;gap:18px}}
</style>
