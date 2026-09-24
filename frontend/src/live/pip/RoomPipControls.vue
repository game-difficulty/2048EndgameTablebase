<template>
  <div class="room-pip-controls">
    <button :disabled="busy" :aria-pressed="active" @click="toggle">{{ active ? t('退出小窗','Exit mini player') : t('小窗观看','Mini player') }}</button>
    <select v-model="requested" :disabled="busy || active" :aria-label="t('小窗方式','Mini player mode')">
      <option value="auto">{{ t('自动','Automatic') }}</option>
      <option value="document">{{ t('房间小窗','Room window') }}</option>
      <option value="video">{{ t('兼容视频小窗','Video fallback') }}</option>
    </select>
    <span v-if="error" role="status" class="pip-error">{{ error }}</span>
    <video ref="video" class="pip-video" muted playsinline aria-hidden="true" tabindex="-1" />
  </div>
</template>
<script setup>
import { ref, onBeforeUnmount, onMounted } from 'vue';
import { choosePipMode, moveSurface, copyRoomStyles, FramePump } from './roomPip.js';
const props = defineProps({ getSurface: Function, getFrame: Function, lang: String, title: String });
const emit = defineEmits(['active', 'surface']);
const t = (zh,en) => props.lang === 'zh' ? zh : en;
const video=ref(null), active=ref(false), busy=ref(false), requested=ref('auto'), error=ref('');
let mode=null, pipWindow=null, restore=null, themeObserver=null, stream=null, pump=null;
let timer=null, startupTimeout=null, disposed=false, attempt=0;
function refresh() {
  if (!active.value) return;
  try { pump?.tick(); }
  catch { error.value=t('小窗播放中断，请重新打开。','Mini-player playback stopped. Please open it again.'); release(); }
}
function release() {
  clearInterval(timer); clearTimeout(startupTimeout); timer=startupTimeout=null;
  themeObserver?.disconnect(); themeObserver=null;
  const windowToClose=pipWindow; pipWindow=null;
  restore?.(); restore=null;
  emit('surface',false);
  windowToClose?.removeEventListener('pagehide',left);
  if(windowToClose && !windowToClose.closed) windowToClose.close();
  if(video.value) {
    video.value.removeEventListener('enterpictureinpicture',entered);
    video.value.removeEventListener('leavepictureinpicture',left);
    video.value.pause(); video.value.srcObject=null;
  }
  stream?.getTracks().forEach(track=>track.stop()); stream=null; pump=null;
  active.value=false; emit('active',false);
}
function entered() { active.value=true; emit('active',true); }
function left() { release(); }
async function close() {
  if(document.pictureInPictureElement===video.value) {
    try { await document.exitPictureInPicture(); } catch { /* release tracks below */ }
  }
  release();
}
async function openDocument(id) {
  // Keep requestWindow directly in the click gesture; no network or await first.
  const target=await window.documentPictureInPicture.requestWindow({width:960,height:600});
  if(disposed || id!==attempt) {target.close();return;}
  pipWindow=target;
  const doc=target.document, surface=props.getSurface();
  if(!surface) throw Error('room_surface_unavailable');
  copyRoomStyles(document,doc);
  doc.title=props.title;
  const syncTheme=()=>{
    doc.documentElement.className=document.documentElement.className;
    doc.documentElement.style.cssText=document.documentElement.style.cssText;
    for(const attr of document.documentElement.attributes) if(attr.name.startsWith('data-') || attr.name==='lang') doc.documentElement.setAttribute(attr.name,attr.value);
  };
  syncTheme(); themeObserver=new MutationObserver(syncTheme); themeObserver.observe(document.documentElement,{attributes:true});
  doc.body.className=document.body.className;
  const style=doc.createElement('style');
  style.textContent=`html,body{margin:0!important;width:100%!important;height:100%!important;overflow:hidden!important;zoom:1!important;background:var(--bg-main,#0f172a);color:var(--text-main,#fff)}
    .pip-room-bar{height:48px;display:flex;align-items:center;gap:12px;padding:0 12px;font:14px sans-serif;box-sizing:border-box}
    .pip-room-bar strong{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.pip-room-bar button{padding:5px 10px;border:1px solid #64748b;border-radius:6px;cursor:pointer}
    .room-pip-surface.live-page{min-width:0!important;width:100%!important;height:calc(100vh - 48px);display:flex;align-items:center;justify-content:center;margin:0!important;padding:0!important}
    .room-pip-surface>.room-stage-viewport{flex:none;width:min(100%,calc((100vh - 48px)*16/9))!important}`;
  doc.head.append(style);
  const bar=doc.createElement('header');bar.className='pip-room-bar';
  const title=doc.createElement('strong');title.textContent=props.title;
  const back=doc.createElement('button');back.textContent=t('返回页面','Return to page');back.onclick=()=>{window.focus();close();};
  const exit=doc.createElement('button');exit.textContent=t('关闭小窗','Close');exit.onclick=close;
  bar.append(title,back,exit);doc.body.append(bar);
  const shell=surface.closest('.live-page').cloneNode(false);shell.classList.add('room-pip-surface');
  doc.body.append(shell);restore=moveSurface(surface,shell);
  emit('surface',true);
  target.addEventListener('pagehide',left,{once:true});
  entered();
}
async function openVideo(id) {
  const canvas=document.createElement('canvas'), ctx=canvas.getContext('2d'), player=video.value;
  if(!ctx) throw Error('canvas_unavailable');
  const initial=props.getFrame();canvas.width=initial.width;canvas.height=initial.height;
  stream=canvas.captureStream(10);
  pump=new FramePump(props.getFrame,frame=>{
    if(canvas.width!==frame.width || canvas.height!==frame.height) {canvas.width=frame.width;canvas.height=frame.height;}
    frame.draw(ctx);stream?.getVideoTracks()[0]?.requestFrame?.();
  });
  player.srcObject=stream;player.addEventListener('enterpictureinpicture',entered);player.addEventListener('leavepictureinpicture',left);
  pump.tick(true);
  await Promise.race([player.play(),new Promise((_,reject)=>{startupTimeout=setTimeout(()=>reject(Error('video_start_timeout')),8000);})]);
  clearTimeout(startupTimeout);
  if(disposed || id!==attempt) return;
  await player.requestPictureInPicture();
  if(disposed || id!==attempt) {await close();return;}
  timer=setInterval(refresh,100);
}
async function toggle() {
  if(active.value) {await close();return;}
  if(busy.value) return;
  error.value='';
  mode=choosePipMode(requested.value,{document:!!window.documentPictureInPicture?.requestWindow,
    video:!!(document.pictureInPictureEnabled && video.value?.requestPictureInPicture && HTMLCanvasElement.prototype.captureStream),
    renderer:!!props.getFrame?.()});
  if(!mode) {error.value=t('此浏览器或内容不支持所选小窗方式，请换一种方式。','This browser or content does not support that mode. Try another mode.');return;}
  busy.value=true;const id=++attempt;
  try {if(mode==='document') await openDocument(id);else await openVideo(id);}
  catch {error.value=t('无法打开小窗，请重试或切换小窗方式。','Could not open the mini player. Retry or choose another mode.');release();}
  finally {busy.value=false;}
}
function unload() {attempt++;release();}
onMounted(()=>window.addEventListener('pagehide',unload));
onBeforeUnmount(()=>{disposed=true;attempt++;window.removeEventListener('pagehide',unload);close();});
defineExpose({refresh,close});
</script>
<style scoped>
.room-pip-controls { position:relative;display:flex;align-items:center;gap:5px;font-size:12px; }
select { background:var(--bg-card);color:var(--text-main);border:1px solid var(--border-main);padding:6px;border-radius:6px;max-width:130px; }
.pip-error { position:absolute;right:0;top:100%;z-index:80;padding:10px;background:var(--bg-card);border:1px solid var(--border-main);width:300px;max-width:80vw;border-radius:8px; }
.pip-video { position:fixed;left:-10px;top:0;width:1px;height:1px;pointer-events:none; }
</style>
