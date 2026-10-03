<script setup>
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue';
import { api, commandId } from './api.js';
import { language, t } from './i18n.js';
import { userFacingError } from './errorMessages.js';
import { PUBLIC_ROOM_MODES, publicRoomMode, isPublicRoom, isOpenRoom, modeText, creationValid, personalPublicRooms } from './publicRoomModes.js';
import { PUBLIC_ROOM_SETUPS } from './publicRoomViews.js';
import PublicRoomCard from './PublicRoomCard.vue';
const props=defineProps({session:Object,rooms:{type:Array,default:()=>[]},mainSiteUrl:String,initialMode:String,initialError:String,busy:Boolean});
const emit=defineEmits(['navigate','close']);
const en=computed(()=>language.value==='en');
const localRooms=ref([]), error=ref(props.initialError||''), known=ref(false);
const allRooms=computed(()=>localRooms.value.filter(isPublicRoom).sort((a,b)=>String(b.updated_at).localeCompare(String(a.updated_at))));
const active=computed(()=>allRooms.value.filter(isOpenRoom)), history=computed(()=>allRooms.value.filter(r=>!isOpenRoom(r)));
const creating=ref(!!props.initialMode), modeKind=ref(props.initialMode||null), selected=computed(()=>publicRoomMode(modeKind.value));
const drafts=ref(Object.fromEntries(PUBLIC_ROOM_MODES.map(m=>[m.kind,{settings:m.defaults(),minutes:m.defaultMinutes,name:''}])));
const draft=computed(()=>drafts.value[modeKind.value]);
const valid=computed(()=>selected.value&&creationValid(selected.value,draft.value.settings,draft.value.minutes,draft.value.name));
const code=ref(''), submitting=ref(false), refreshing=ref(false), filter=ref('all'), creator=ref(null);
let command=commandId(), alive=true, roomsEpoch=0;
onBeforeUnmount(()=>{alive=false;});
async function setRooms(rows){
  const epoch=++roomsEpoch;
  const own=props.session?.can_create_competition
    ? await personalPublicRooms(rows,async code=>(await api.room(code,{timeoutMs:5000})).competition) : rows;
  if(alive&&epoch===roomsEpoch){localRooms.value=own;known.value=true;}
}
watch(()=>props.rooms,rows=>{
  if(props.initialError&&!known.value){error.value=props.initialError;return;}
  setRooms(rows).catch(e=>{if(alive){known.value=false;error.value=t(userFacingError(e));}});
},{immediate:true});
watch(()=>props.initialError,value=>{if(value)error.value=value;});
watch(drafts,()=>command=commandId(),{deep:true});
watch(modeKind,()=>command=commandId());
const records=computed(()=>history.value.filter(r=>filter.value==='all'||r.room_kind===filter.value));
async function openCreate(){creating.value=true;await nextTick();creator.value?.scrollIntoView({behavior:'smooth',block:'start'});}
async function refresh(){
  if(!props.session||refreshing.value)return;
  refreshing.value=true;
  try{const result=await api.list({timeoutMs:5000});await setRooms(result.competitions);if(alive)error.value='';}
  catch(e){if(alive){error.value=t(userFacingError(e));known.value=false;}}finally{refreshing.value=false;}
}
async function create(){
  if(submitting.value||refreshing.value||!props.session||!valid.value)return;
  submitting.value=true;error.value='';
  // Recheck cross-mode occupancy immediately before creating, including other tabs.
  await refresh();
  if(!alive)return;
  if(!known.value||active.value.length){submitting.value=false;return;}
  try{
    const result=await api[selected.value.createMethod]({...selected.value.payload(draft.value.settings),
      name:draft.value.name.trim()||modeText(selected.value.title,language.value),clock_seconds:Math.round(Number(draft.value.minutes)*60),command_id:command});
    if(alive)emit('navigate',`/rooms/${result.competition.room_code}`);
  }catch(e){
    if(e.code==='DUEL_ACTIVE_ROOM')await refresh();
    if(alive)error.value=t(userFacingError(e));
  }finally{submitting.value=false;}
}
</script>
<template>
  <main class="page-width public-lobby">
    <header class="public-intro"><div><p class="eyebrow">PLAY TOGETHER · 1 VS 1</p><h1>{{en?'Free play':'自由对战'}}</h1><p>{{en?'Pick a format. Invite a friend. Play your way.':'选择一种玩法，邀请一位对手，开始你们的对战。'}}</p></div><button class="primary-button" :disabled="!session || submitting || active.length>0 || !known" @click="openCreate">＋ {{en?'Create room':'创建房间'}}</button></header>
    <form class="panel public-join" @submit.prevent="code.trim() && emit('navigate',`/rooms/${encodeURIComponent(code.trim().toUpperCase())}`)"><label for="public-room-code">{{en?'Have a room code?':'已有房间码？'}}<small>{{en?'One entrance for every format.':'所有玩法都从这里加入。'}}</small></label><div class="inline-form"><input id="public-room-code" v-model="code" maxlength="12" required :placeholder="en?'Enter room code':'输入房间码'" autocapitalize="characters" autocomplete="off"/><button class="secondary-button">{{en?'Join room':'进入房间'}}</button></div></form>
    <p v-if="error" class="alert" role="alert">{{t(error)}} <button type="button" class="secondary-button" :disabled="refreshing||submitting" @click="refresh">{{en?'Refresh rooms':'刷新房间'}}</button></p>
    <p v-if="!session" class="panel public-signin">{{en?'Sign in to create a room or join a friend.':'登录后即可创建房间或加入对战。'}} <a :href="mainSiteUrl">{{en?'Sign in':'前往登录'}}</a></p>
    <section v-if="session" class="public-active"><div class="public-section-heading"><h2>{{en?'Your active room':'进行中的房间'}}</h2><button type="button" class="secondary-button" :disabled="refreshing||submitting||busy" @click="refresh">{{refreshing?(en?'Refreshing…':'刷新中…'):(en?'Refresh':'刷新')}}</button></div>
      <p v-if="active.length" class="public-occupancy">{{en?'Continue your current match before creating another room. This limit is shared across all formats.':'请先返回当前房间，完成对战或关闭未开赛房间，再创建新房间。所有玩法共用这一限制。'}}</p>
      <div class="public-room-list"><PublicRoomCard v-for="r in active" :key="r.id" :room="r" :busy="busy||submitting" @navigate="path=>emit('navigate',path)" @close="r=>emit('close',r)" /></div>
      <p v-if="!active.length&&known" class="muted">{{en?'No active room. Choose a format below to start.':'当前没有进行中的房间，选择下方玩法即可开始。'}}</p>
    </section>
    <section ref="creator" class="public-create" :aria-label="en?'Create a room':'创建房间'">
      <div class="public-section-heading"><h2>{{creating?(en?'1 · Choose a format':'1 · 选择玩法'):(en?'Available formats':'可选玩法')}}</h2><button v-if="creating" type="button" class="secondary-button" :disabled="submitting" @click="creating=false;modeKind=null">{{en?'Cancel':'取消创建'}}</button></div>
      <div class="public-mode-grid"><button v-for="m in PUBLIC_ROOM_MODES" :key="m.kind" type="button" class="public-mode" :aria-pressed="creating&&modeKind===m.kind" :disabled="submitting" @click="modeKind=m.kind;creating=true">
        <small>{{m.tag}}</small><h3>{{modeText(m.title,language)}}</h3><p>{{modeText(m.description,language)}}</p><span>{{modeText(m.detail,language)}}</span>
      </button></div>
      <form v-if="creating&&selected" class="panel public-builder" @submit.prevent="create">
        <h2>{{en?'2 · Configure your room':'2 · 设置房间'}} <small>{{modeText(selected.title,language)}}</small></h2>
        <p v-if="active.length" class="public-occupancy">{{en?'Your settings are kept here. Return to the active room above before creating.':'这里会保留你的配置，请先处理上方进行中的房间。'}}</p>
        <fieldset :disabled="submitting||!session||active.length>0||!known">
          <div class="public-common-fields"><label>{{en?'Room name':'房间名称'}}<input v-model="draft.name" minlength="2" maxlength="100" :placeholder="modeText(selected.title,language)"/></label><label>{{modeText(selected.clockLabel,language)}}<input v-model="draft.minutes" type="number" :min="selected.minMinutes" max="1440" :step="selected.step" required/></label></div>
          <component :is="PUBLIC_ROOM_SETUPS[modeKind]" :key="modeKind" v-model="draft.settings" />
          <div class="public-rules"><strong>{{en?'Format rules':'玩法说明'}}</strong><p>{{modeText(selected.rules,language)}}</p></div>
          <footer class="public-submit"><div><strong>{{selected.summary(draft.settings,en)}} · {{draft.minutes}} {{en?'min':'分钟'}}</strong><small>{{en?'Both players must ready up. No referees or automatic ready-up.':'双方准备后开赛，无裁判，不会自动准备。'}}</small></div><button class="primary-button" :disabled="!valid||refreshing">{{submitting?(en?'Creating…':'创建中…'):(en?'Create room':'创建房间')}}</button></footer>
        </fieldset>
        <p class="muted public-limits">{{en?'One active room per user. Unstarted rooms expire after 30 minutes; project duels also expire after a 30-minute wait between games. Creation: once per minute, up to 10 per hour.':'每人最多一个活跃房间。开赛前等待满 30 分钟关闭；项目对决的局间等待也有 30 分钟上限。每分钟最多创建一次，每小时最多 10 次。'}}</p>
      </form>
    </section>
    <section v-if="session" class="public-history"><div class="public-section-heading"><h2>{{en?'Room history':'历史房间'}}</h2><label class="public-filter"><span>{{en?'Format':'玩法'}}</span><select v-model="filter"><option value="all">{{en?'All formats':'全部玩法'}}</option><option v-for="m in PUBLIC_ROOM_MODES" :key="m.kind" :value="m.kind">{{modeText(m.title,language)}}</option></select></label></div>
      <div class="public-room-list"><PublicRoomCard v-for="r in records" :key="r.id" :room="r" @navigate="path=>emit('navigate',path)" /></div><p v-if="!records.length" class="muted">{{en?'No past rooms for this filter.':'暂无符合条件的历史房间。'}}</p>
    </section>
  </main>
</template>
<style scoped>
.public-lobby{padding-block:32px 56px}.public-intro,.public-section-heading{display:flex;align-items:center;justify-content:space-between;gap:16px}.public-intro{margin-bottom:24px}.public-intro h1{font-size:clamp(28px,4vw,40px);line-height:1.2;margin:8px 0}.public-intro p{line-height:1.6;color:var(--competition-muted)}.public-intro>.primary-button{flex-shrink:0;min-height:46px}.public-join{padding:18px 22px;display:flex;gap:20px;align-items:center;justify-content:space-between}.public-join label{font-weight:700}.public-join small{display:block;font-size:12px;font-weight:400;color:var(--competition-muted);margin-top:5px}.public-join .inline-form{width:min(100%,370px)}.public-join input{min-width:0;flex:1}.public-active,.public-create,.public-history{margin-top:28px}.public-section-heading{margin-bottom:14px}.public-section-heading h2,.public-builder h2{font-size:20px;margin:0}.public-room-list{display:grid;gap:10px}.public-mode-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.public-mode{display:block;padding:22px;text-align:left;color:inherit;background:var(--competition-card);border:1px solid var(--competition-border);border-radius:14px;font:inherit;cursor:pointer}.public-mode[aria-pressed=true]{border-color:#ac853d;box-shadow:inset 0 0 0 1px #ac853d;background:color-mix(in srgb,var(--competition-card) 95%,#b88b37)}.public-mode>small{font-size:11px;font-weight:700;letter-spacing:.08em;color:#a37c32}.public-mode h3{font-size:22px;margin:12px 0}.public-mode p{font-size:14px;line-height:1.65;margin:0 0 18px;color:var(--competition-muted)}.public-mode>span{display:block;font-size:12px;border-top:1px solid var(--competition-border);padding-top:12px}.public-builder{padding:24px;margin-top:18px}.public-builder h2{margin-bottom:22px}.public-builder h2 small{font-size:13px;color:var(--competition-muted);margin-left:12px}.public-builder fieldset{border:0;padding:0;margin:0;min-width:0;display:grid;gap:24px}.public-common-fields{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}.public-common-fields label{display:grid;gap:8px}.public-common-fields input{min-height:44px;width:100%;box-sizing:border-box}.public-rules{padding:14px 16px;border-radius:8px;background:#aa88440b;border:1px solid var(--competition-border);font-size:13px;line-height:1.65}.public-rules p{margin:6px 0 0;color:var(--competition-muted)}.public-submit{display:flex;gap:16px;align-items:center;justify-content:space-between;border-top:1px solid var(--competition-border);padding-top:20px}.public-submit>div{min-width:0}.public-submit strong{font-size:14px;overflow-wrap:anywhere}.public-submit small{display:block;margin-top:7px;font-size:12px;color:var(--competition-muted)}.public-submit button{flex:none;min-height:44px}.public-occupancy{line-height:1.6;border-left:3px solid #b58f45;padding:8px 14px;background:#b58f450a;font-size:14px}.public-limits{font-size:12px;line-height:1.7;margin-top:18px}.public-filter{display:flex;align-items:center;gap:8px;font-size:13px}.public-filter select{min-height:40px}.public-signin{padding:18px}.public-lobby button:disabled{opacity:.5;cursor:not-allowed}.public-lobby fieldset:disabled{opacity:.65}.public-create{scroll-margin-top:20px}@media(max-width:650px){.public-lobby{padding-top:22px}.public-intro{align-items:start;flex-wrap:wrap}.public-intro h1{font-size:28px}.public-join{padding:16px;flex-direction:column;align-items:stretch;gap:12px}.public-join .inline-form{width:100%}.public-mode-grid{gap:10px;grid-template-columns:1fr}.public-mode{padding:18px}.public-mode h3{font-size:20px}.public-mode p{margin-bottom:12px}.public-builder{padding:16px}.public-common-fields{grid-template-columns:1fr;gap:14px}.public-submit{align-items:stretch;flex-direction:column}.public-section-heading{flex-wrap:wrap}.public-section-heading h2{font-size:18px}.public-builder h2 small{display:block;margin:8px 0 0}.public-filter span{display:none}}
.public-filter{flex-shrink:0}.public-filter span{white-space:nowrap}
@media(max-width:650px){.public-filter span{display:block;position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}}
</style>
