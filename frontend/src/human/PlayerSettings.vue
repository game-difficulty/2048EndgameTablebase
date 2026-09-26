<template>
  <div class="player-settings panel">
    <h2>{{ t('系统设置') }}</h2>
    <p v-if="preferenceSyncStatus === 'error'" role="alert">{{ language === 'zh' ? '账号设置尚未同步，请检查网络。' : 'Account settings have not synced. Check your connection.' }} <button @click="retryAccountPreferences">{{ language === 'zh' ? '重试' : 'Retry' }}</button></p>
    <div class="settings-subtabs"><button :class="{active:section==='game'}" @click="section='game'">{{ t('游戏与记录') }}</button><button :class="{active:section==='theme'}" @click="section='theme'">{{ t('界面与主题') }}</button></div>
    <div v-if="section==='game'" class="settings-rows">
      <div class="setting-line"><strong>{{ t('界面语言') }}</strong><div class="settings-options"><button :class="{active:language==='en'}" @click="setLanguage('en')">English</button><button :class="{active:language==='zh'}" @click="setLanguage('zh')">简体中文</button></div></div>
      <label class="setting-line"><strong>{{ t('方块字体比例') }} <output>{{ preferences.font_size_factor || 100 }}%</output></strong><input type="range" min="50" max="150" step="5" :value="preferences.font_size_factor || 100" @input="savePreference('font_size_factor',+$event.target.value)"></label>
      <label class="setting-line"><strong>{{ t('界面字号比例') }} <output>{{ preferences.ui_scale || 100 }}%</output></strong><input type="range" min="90" max="125" step="5" :value="preferences.ui_scale || 100" @input="savePreference('ui_scale',+$event.target.value)"></label>
      <label class="setting-line setting-checkbox"><strong>{{ t('深色模式') }}</strong><input type="checkbox" :checked="preferences.dark_mode !== false" @change="savePreference('dark_mode',$event.target.checked)"></label>
      <label class="setting-line setting-checkbox"><strong>{{ t('开启移动动画') }}</strong><input type="checkbox" :checked="preferences.do_animation !== false" @change="savePreference('do_animation',$event.target.checked)"></label>
      <h3>{{ t('对局设置') }}</h3>
      <label class="setting-line setting-checkbox"><strong>{{ t('重开确认') }}</strong><input type="checkbox" :checked="!!playSettings.alwaysConfirmRestart" @change="updatePlay({alwaysConfirmRestart:$event.target.checked})"></label>
      <label class="setting-line"><strong>{{ t('滑动灵敏度') }} <output>{{ playSettings.swipeSensitivity || 100 }}%</output></strong><input type="range" min="50" max="200" step="5" :value="playSettings.swipeSensitivity || 100" @input="updatePlay({swipeSensitivity:+$event.target.value})"></label>
      <div class="setting-line"><strong>{{ t('成绩展示阈值') }}</strong><p>{{ t('新局开始时固定展示阈值。所有正式对局仍会上传留存；低于阈值的记录不会显示在主页或榜单。') }}</p><div class="threshold-grid"><label v-for="variant in variants" :key="variant">{{ variant }} <input type="number" min="0" max="100000000" step="1" v-model.number="thresholds[variant]" @change="saveThresholds"></label></div><small v-if="thresholdError" role="alert">{{ t(thresholdError) }}</small></div>
      <label class="setting-line setting-checkbox"><strong>{{ t('显示每秒输入／移动次数') }}</strong><input type="checkbox" :checked="!!playSettings.showSpeed" @change="updatePlay({showSpeed:$event.target.checked})"></label>
      <label class="setting-line setting-checkbox"><strong>{{ t('显示出 4 比例') }}</strong><input type="checkbox" :checked="!!playSettings.showFourPercent" @change="updatePlay({showFourPercent:$event.target.checked})"></label>
    </div>
    <div v-else class="settings-rows"><div class="setting-line"><strong>{{ t('全局主题') }}</strong><div class="theme-choices"><button v-for="name in themeNames" :key="name" :class="{active:!preferences.use_custom_theme && preferences.theme===name}" @click="selectTheme(name)">{{ name }}</button><button :class="{active:preferences.use_custom_theme}" @click="savePreference('use_custom_theme',true)">✨{{ t('用户自定义') }}</button></div></div>
      <div class="setting-line"><strong>{{ t('方块颜色自定义') }}</strong><div class="custom-colors"><label v-for="index in 16" :key="index"><span>{{ 2**index }}</span><input type="color" :value="customColors[index-1]" @input="changeColor(index-1,$event.target.value)"></label></div></div>
    </div>
    <section v-if="section === 'game'" class="settings-rows verse-claim-settings">
      <h3>{{ t('继承 2048Verse 历史') }}</h3>
      <p>{{ t('每个本站账号可继承一个 Verse 账号。归属由站长审核；通过后，历史成绩进入本站排行榜、BEST 和个人主页。') }}</p>
      <div v-if="verseClaim">
        <div class="setting-line"><strong>{{ verseClaim.username }}</strong><span>{{ t(verseStatus[verseClaim.status] || verseClaim.status) }}</span></div>
        <p v-if="verseClaim.error" role="alert">{{ verseClaim.error }}</p>
        <p v-if="verseClaim.status === 'complete'">{{ t('已继承记录') }}：{{ Object.values(verseClaim.counts || {}).reduce((a,b)=>a+Number(b||0),0) }}</p>
        <p class="verse-claim-date">{{ t(verseClaim.status === 'complete' ? '继承日期' : '提交日期') }}：{{ claimDate(verseClaim.status === 'complete' ? verseClaim.updated_at : verseClaim.requested_at) }}</p>
      </div>
      <form v-else class="setting-line" @submit.prevent="requestVerseClaim">
        <label><strong>{{ t('Verse 用户名') }}</strong><input v-model.trim="verseUsername" required maxlength="64" pattern="[A-Za-z0-9_\-]+" autocomplete="off"></label>
        <button type="submit" :disabled="verseBusy">{{ t(verseBusy ? '提交中…' : '申请继承') }}</button>
      </form>
      <button type="button" :disabled="verseBusy" @click="loadVerseClaim">{{ t('刷新状态') }}</button>
      <small v-if="verseError" role="alert">{{ t(verseError) }}</small>
    </section>
    <section v-if="section === 'game'" class="settings-rows archive-application-settings">
      <h3>{{ t('补录申请') }}</h3>
      <p>{{ t('上传回放后，服务器会核对变体和最终得分；验证通过后交由站长审批。批准的补录局不参加近 168 小时榜及每周 Token 结算。') }}</p>
      <button type="button" @click="openArchiveApplication">{{ t('提交补录申请') }}</button>
      <div v-if="archiveApplications.length" class="archive-application-list">
        <div v-for="item in archiveApplications" :key="item.id">
          <strong>{{ item.variant.replace('x',' × ') }} · {{ formatNumber(item.score) }}</strong>
          <span>{{ t(archiveStatus[item.status] || item.status) }}</span>
          <small>{{ claimDate(item.ended_at) }} · {{ formatNumber(item.moves) }} {{ t('步') }}</small>
          <p v-if="item.review_note">{{ item.review_note }}</p>
        </div>
      </div>
      <small v-if="archiveLoadError" role="alert">{{ t(archiveLoadError) }}</small>
    </section>
    <Teleport to="body">
      <div v-if="archiveDialog" class="modal-backdrop" @click.self="closeArchiveApplication" @keydown.esc="closeArchiveApplication">
        <section class="modal archive-application-dialog" role="dialog" aria-modal="true" :aria-label="t('补录申请')">
          <button class="modal-close" type="button" :aria-label="t('关闭')" @click="closeArchiveApplication">×</button>
          <h2>{{ t('补录申请') }}</h2>
          <p>{{ t('回放必须能够完整解析，且服务器重算分数必须与填写分数一致。') }}</p>
          <form @submit.prevent="submitArchiveApplication">
            <label>{{ t('棋盘变体') }}<select v-model="archiveForm.variant" required><option v-for="variant in variants" :key="variant" :value="variant">{{ variant.replace('x',' × ') }}</option></select></label>
            <label>{{ t('对局结束时间') }}<input v-model="archiveForm.endedAt" type="datetime-local" required></label>
            <label>{{ t('最终得分') }}<input v-model.number="archiveForm.score" type="number" min="0" max="2000000000" step="1" required></label>
            <label>{{ t('回放文件') }}<input type="file" accept=".vrs,.txt,.hpr,application/octet-stream" required @change="archiveFile=$event.target.files?.[0]||null"></label>
            <p class="small muted">{{ t('支持 .vrs、回放代码、Verse 文本回放和本站 .hpr；文件上限 2 MB。') }}</p>
            <p v-if="archiveError" class="notice danger" role="alert">{{ t(archiveError) }}</p>
            <div class="modal-actions"><button type="button" :disabled="archiveBusy" @click="closeArchiveApplication">{{ t('关闭') }}</button><button class="primary" type="submit" :disabled="archiveBusy">{{ t(archiveBusy ? '提交中…' : '提交申请') }}</button></div>
          </form>
        </section>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted } from 'vue';
import themes from '../../../docs_and_configs/themes.json';
import { createLocalStorageStore } from '../services/storage/localStorageStore.js';
import { writeSharedTilePalette } from '../utils/sharedTilePalette.js';
import { resolveTileColors } from '../utils/tileColors.js';
import { json, request } from './client.js';
import { t, language, setLanguage } from './i18n.js';
import { preferenceSyncStatus, refreshAccountPreferences, retryAccountPreferences, saveAccountPreferences } from '../services/preferences/accountPreferences.js';
const props=defineProps({playSettings:{type:Object,required:true}});
const emit=defineEmits(['update:play-settings']);
const store=createLocalStorageStore({key:'user-preferences',version:1,defaultValue:{}});
const preferences=ref(store.read());
const themeNames=Object.keys(themes).sort();
const variants=['4x4','3x4','2x4','3x3'];
const section=ref('game'), thresholdError=ref('');
const verseUsername=ref(''), verseClaim=ref(null), verseBusy=ref(false), verseError=ref('');
const verseStatus={pending:'等待站长审核',approved:'已批准，等待导入',importing:'正在导入',complete:'继承完成',failed:'导入失败'};
const archiveApplications=ref([]), archiveDialog=ref(false), archiveBusy=ref(false), archiveError=ref(''), archiveLoadError=ref('');
const archiveFile=ref(null), archiveForm=reactive({variant:'4x4',endedAt:'',score:0});
const archiveStatus={pending:'等待站长审核',approved:'已批准并归档',rejected:'申请已拒绝',revoked:'归档资格已撤销'};
const thresholds=reactive({'4x4':0,'3x4':0,'2x4':0,'3x3':0});
const customColors=reactive(Array.from({length:16},(_,i)=>preferences.value.custom_colors?.[i] || themes.Default[i]));
function claimDate(seconds){return seconds?new Date(seconds*1000).toLocaleString(language.value==='zh'?'zh-CN':'en-US'):'—';}
const formatNumber=value=>new Intl.NumberFormat(language.value==='zh'?'zh-CN':'en-US').format(Number(value)||0);
function localDateTimeValue(date=new Date()){
  const shifted=new Date(date.getTime()-date.getTimezoneOffset()*60000);
  return shifted.toISOString().slice(0,16);
}
function refreshPreferences(){
  preferences.value=store.read();
  for(let i=0;i<16;i++) customColors[i]=preferences.value.custom_colors?.[i] || themes.Default[i];
}
function updatePlay(changes){emit('update:play-settings',{...props.playSettings,...changes});}
function savePreference(key,value){
  preferences.value=store.update(current=>{
    if(key==='use_custom_theme') return {...current,use_custom_theme:true,colors:current.custom_colors || themes.Default};
    return {...current,[key]:value};
  });
  if(key==='language')setLanguage(value);
  saveAccountPreferences(key==='use_custom_theme'
    ? {use_custom_theme:true,custom_colors:preferences.value.custom_colors || themes.Default}
    : {[key]:value});
  if(key==='theme'||key==='use_custom_theme'||key==='custom_colors'||key==='dark_mode'||key==='font_size_factor'||key==='ui_scale'||key==='do_animation'){
    if(key==='theme')writeSharedTilePalette(resolveTileColors(themes[value]));
    if(key==='custom_colors'||key==='use_custom_theme')writeSharedTilePalette(resolveTileColors(preferences.value.custom_colors || themes.Default));
    window.dispatchEvent(new Event('human-preferences-changed'));
  }
}
function selectTheme(name){
  preferences.value=store.update(current=>({...current,theme:name,use_custom_theme:false,colors:themes[name]}));
  writeSharedTilePalette(resolveTileColors(themes[name]));
  saveAccountPreferences({theme:name,use_custom_theme:false});
  window.dispatchEvent(new Event('human-preferences-changed'));
}
function changeColor(index,color){
  customColors[index]=color;
  const all=Array.from({length:36},(_,i)=>i<16?customColors[i]:preferences.value.custom_colors?.[i]||themes.Default[i]);
  preferences.value=store.update(current=>({...current,custom_colors:all,colors:all,use_custom_theme:true}));
  writeSharedTilePalette(resolveTileColors(all));
  saveAccountPreferences({custom_colors:all,use_custom_theme:true});
  window.dispatchEvent(new Event('human-preferences-changed'));
}
async function saveThresholds(){
  if(variants.some(v=>!Number.isInteger(thresholds[v])||thresholds[v]<0||thresholds[v]>100000000)){
    thresholdError.value='请输入 0 到 100000000 之间的整数。';return;
  }
  try{await json('/api/human/me/settings',{method:'PUT',body:{display_thresholds:Object.fromEntries(variants.map(v=>[v,thresholds[v]]))}});thresholdError.value='';}
  catch{thresholdError.value='展示阈值保存失败，请检查网络后重试。';}
}
async function loadVerseClaim(){
  verseBusy.value=true;verseError.value='';
  try{verseClaim.value=(await json('/api/human/me/verse-claim')).claim;}
  catch{verseError.value='无法读取继承状态，请稍后重试。';}
  finally{verseBusy.value=false;}
}
async function requestVerseClaim(){
  verseBusy.value=true;verseError.value='';
  try{verseClaim.value=(await json('/api/human/me/verse-claim',{method:'POST',body:{username:verseUsername.value}})).claim;}
  catch(e){verseError.value=e.code==='verse_claim_already_exists'?'此账号已有继承申请。':e.code==='verse_account_claimed'?'此 Verse 账号已被继承。':'继承申请失败，请稍后重试。';}
  finally{verseBusy.value=false;}
}
async function loadArchiveApplications(){
  archiveLoadError.value='';
  try{archiveApplications.value=(await json('/api/human/me/archive-applications')).applications||[];}
  catch{archiveLoadError.value='无法读取补录申请，请稍后重试。';}
}
function openArchiveApplication(){
  archiveForm.variant='4x4';archiveForm.score=0;archiveForm.endedAt=localDateTimeValue();
  archiveFile.value=null;archiveError.value='';archiveDialog.value=true;
}
function closeArchiveApplication(){if(!archiveBusy.value)archiveDialog.value=false;}
const archiveErrors={replay_score_mismatch:'回放重算得分与填写分数不一致。',replay_variant_mismatch:'回放变体与所选变体不一致。',replay_already_submitted:'这份回放已经提交或归档。',archive_application_pending_limit:'待审批的补录申请已达到上限。',archive_application_queue_full:'补录审核队列已满，请稍后再试。',archive_application_rate_limit:'今天提交的补录申请过多，请稍后再试。',replay_too_large:'回放文件不能超过 2 MB。',replay_size_invalid:'回放文件无效或超过大小限制。',invalid_ended_at:'对局结束时间无效。'};
async function submitArchiveApplication(){
  if(!archiveFile.value){archiveError.value='请选择回放文件。';return;}
  if(archiveFile.value.size>2*1024*1024){archiveError.value='回放文件不能超过 2 MB。';return;}
  const endedAt=new Date(archiveForm.endedAt).getTime()/1000;
  if(!Number.isFinite(endedAt)){archiveError.value='对局结束时间无效。';return;}
  archiveBusy.value=true;archiveError.value='';
  try{
    const query=new URLSearchParams({variant:archiveForm.variant,ended_at:String(endedAt),score:String(archiveForm.score),filename:archiveFile.value.name});
    const body=await archiveFile.value.arrayBuffer();
    const response=await request('/api/human/me/archive-applications?'+query,{method:'POST',body,binary:true,timeoutMs:25000});
    const result=await response.json();archiveApplications.value.unshift(result.application);archiveDialog.value=false;
  }catch(e){archiveError.value=archiveErrors[e.code]||'补录申请提交失败，请检查回放后重试。';}
  finally{archiveBusy.value=false;}
}
onMounted(async()=>{
  void refreshAccountPreferences();
  window.addEventListener('focus',refreshPreferences);
  window.addEventListener('storage',refreshPreferences);
  window.addEventListener('account-preferences-changed',refreshPreferences);
  try{const data=await json('/api/human/me/settings');Object.assign(thresholds,data.display_thresholds||{});}
  catch{thresholdError.value='无法读取展示阈值，请检查网络后重试。';}
  loadVerseClaim();loadArchiveApplications();
});
onUnmounted(()=>{
  window.removeEventListener('focus',refreshPreferences);
  window.removeEventListener('storage',refreshPreferences);
  window.removeEventListener('account-preferences-changed',refreshPreferences);
});
</script>

<style scoped>
.archive-application-list{display:grid;gap:8px;margin-top:8px}.archive-application-list>div{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 12px;border-top:1px solid var(--line);padding:10px 0}.archive-application-list small,.archive-application-list p{grid-column:1/-1;margin:0;color:var(--muted)}.archive-application-dialog select{display:block;width:100%;margin-top:6px}.archive-application-dialog .notice{margin:12px 0 0}
</style>
