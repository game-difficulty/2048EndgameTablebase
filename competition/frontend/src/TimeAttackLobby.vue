<script setup>
import { computed, ref } from 'vue';
import { api, commandId } from './api.js';
import { language, t } from './i18n.js';
import { userFacingError } from './errorMessages.js';
defineProps({ session:Object, rooms:{type:Array,default:()=>[]}, mainSiteUrl:String });
const emit = defineEmits(['navigate']);
const en = computed(()=>language.value==='en');
const name=ref(''), variant=ref('4x4'), kind=ref('tile'), value=ref(2048), minutes=ref(10), code=ref('');
const busy=ref(false), error=ref('');
let command=commandId();
const valid = computed(()=>Number.isInteger(Number(value.value)) && (kind.value==='tile'
  ? Number(value.value)>=8 && Number(value.value)<=2147483648 && Number.isInteger(Math.log2(Number(value.value)))
  : Number(value.value)>=10 && Number(value.value)<=2147483648 && Number(value.value)%2===0));
function changed(){ command=commandId(); }
async function create(){
  busy.value=true; error.value='';
  try {
    const result=await api.createTimeAttack({ name:name.value.trim() || (en.value?'Timed PB duel':'限时竞速对决'),
      variant:variant.value,target_kind:kind.value,target_value:Number(value.value),clock_seconds:Math.round(Number(minutes.value)*60),command_id:command });
    emit('navigate',`/rooms/${result.competition.room_code}`);
  } catch(e){error.value=t(userFacingError(e));} finally{busy.value=false;}
}
</script>
<template>
  <main class="page-width time-lobby">
    <header><p class="eyebrow">TIME ATTACK · PERSONAL BEST</p><h1>{{ en?'One time limit. Your fastest run.':'限时挑战，刷新你的最快单局。' }}</h1>
      <p>{{ en?'Restart as often as you like. Your best verified run decides the winner when time runs out.':'限时内无限重开，达标后仍可继续挑战。时间结束，比较双方最佳有效单局。' }}</p></header>
    <p v-if="error" class="alert" role="alert">{{error}}</p>
    <div class="time-lobby-grid">
      <form class="panel" @submit.prevent="create" @input="changed" @change="changed">
        <h2>{{en?'Create a timed duel':'创建限时对决'}}</h2>
        <fieldset :disabled="busy || !session">
          <label>{{en?'Room name':'房间名称'}}<input v-model="name" maxlength="100" minlength="2" :placeholder="en?'Timed PB duel':'限时竞速对决'" /></label>
          <label>{{en?'Board':'棋盘变体'}}<select v-model="variant"><option v-for="v in ['4x4','3x4','2x4','3x3']" :key="v" :value="v">{{v.replace('x',' × ')}}</option></select></label>
          <label>{{en?'Target type':'目标类型'}}<select v-model="kind"><option value="tile">{{en?'Target tile (at least)':'目标数字（不小于）'}}</option><option value="board_sum">{{en?'Board sum (exact)':'目标盘面和（精确等于）'}}</option></select></label>
          <label>{{en?'Target value':'具体数值'}}<input v-model="value" type="number" :min="kind==='tile'?8:10" max="2147483648" :step="kind==='tile'?1:2" required /></label>
          <p v-if="!valid" class="alert">{{en?'Choose a power of two ≥ 8, or an even board sum ≥ 10.':'目标数字须为不小于 8 的 2 的幂；盘面和须为不小于 10 的偶数。'}}</p>
          <label>{{en?'Time limit (minutes)':'总时长（分钟）'}}<input v-model="minutes" type="number" min="0.5" max="1440" step="0.5" required /></label>
          <div class="time-preview">{{variant.replace('x',' × ')}} · {{kind==='tile'?(en?'Tile':'目标数字'):(en?'Exact sum':'精确盘面和')}} {{value}} · {{minutes}} {{en?'min':'分钟'}}</div>
          <button class="primary-button" :disabled="!valid">{{busy?(en?'Creating…':'创建中…'):(en?'Create room':'创建房间')}}</button>
        </fieldset>
        <p v-if="!session"><a :href="mainSiteUrl">{{en?'Sign in to create or join':'登录后即可创建或加入'}}</a></p>
      </form>
      <aside class="panel"><h2>{{en?'How it works':'对决规则'}}</h2>
        <ul><li>{{en?'Both players ready up. The shared countdown then starts.':'双方准备后，同时开始总倒计时。'}}</li><li>{{en?'R restarts a run; no undo. Completed runs remain recorded.':'按 R 重开，不可悔棋。重开保留已确认的 PB。'}}</li><li>{{en?'A run starts when the server creates its board, not on the first move.':'单局从服务端生成棋盘时开始计时，不从第一步开始。'}}</li><li>{{en?'PB uses server-confirmed time, including network latency. Only moves received before the deadline count.':'PB 采用服务端确认用时，包含网络耗时。仅截止前收到的有效操作计入。'}}</li><li>{{en?'Disconnection does not pause time. Reconnect to resume the same attempt.':'断线不暂停计时，重连恢复同一次尝试。'}}</li><li>{{en?'Equal PBs or no valid runs on either side mean a draw.':'PB 相同或双方均无有效成绩时，判为平局。'}}</li></ul>
        <p class="muted">{{en?'One active public duel per user, across both modes. Waiting rooms expire after 30 minutes. Creation: once per minute, up to 10 per hour.':'两种公开对决共用每人一个活跃房间限制。等待 30 分钟未开始则关闭；每分钟最多创建一次，每小时最多 10 次。'}}</p>
        <form @submit.prevent="code.trim() && emit('navigate',`/rooms/${encodeURIComponent(code.trim().toUpperCase())}`)"><label>{{en?'Join by room code':'通过房间码加入'}}<input v-model="code" maxlength="12" required /></label><button class="secondary-button">{{en?'Join':'进入'}}</button></form>
      </aside>
    </div>
    <section class="panel time-history"><h2>{{en?'My timed duels':'我的限时对决'}}</h2><p v-if="!rooms.length">{{en?'No rooms yet.':'暂无房间。'}}</p><button v-for="room in rooms" :key="room.id" class="secondary-button" @click="emit('navigate',`/rooms/${room.room_code}`)">{{room.name}} · {{room.room_code}} · {{room.status}}</button></section>
  </main>
</template>
<style scoped>
.time-lobby{padding-block:32px}.time-lobby header{max-width:800px;margin-bottom:24px}.time-lobby h1{font-size:clamp(26px,4vw,42px);line-height:1.25}.time-lobby p,.time-lobby li{line-height:1.7}.time-lobby-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}.time-lobby .panel{padding:24px}.time-lobby fieldset{border:0;padding:0;margin:0;display:grid;gap:14px;min-width:0}.time-lobby label{display:grid;gap:7px}.time-lobby input,.time-lobby select{width:100%;min-width:0;box-sizing:border-box;min-height:44px}.time-lobby ul{padding-left:20px;display:grid;gap:10px}.time-preview{padding:12px;border:1px solid #a88b4b66;border-radius:8px;line-height:1.6}.time-history{margin-top:20px}.time-history button{margin:4px;max-width:100%;overflow-wrap:anywhere}@media(max-width:720px){.time-lobby-grid{grid-template-columns:1fr}.time-lobby .panel{padding:16px}}
</style>
