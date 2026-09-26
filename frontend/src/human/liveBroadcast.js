import { ref } from 'vue';
import { json } from './client.js';
import { eventBytes } from './engine.js';
import { uploadBody } from './wire.js';
import { copyLiveShareText, liveShareText } from './liveShareText.js';

const LIVE_ERRORS = {
  live_room_capacity: '当前直播间已满，请稍后重试。',
  human_live_not_configured: '直播服务尚未配置完成。',
  live_run_not_current: '当前对局的写入权已改变，请刷新页面后重试。',
  live_run_unavailable: '只有已登录账号的当前正式对局可以直播。',
  live_room_not_found: '直播间已结束，请重新开启。',
};

const liveError = error => LIVE_ERRORS[error?.code || error?.message] || '直播连接失败，请稍后重试。';

export function createLiveBroadcast(session, getBest = () => 0) {
  const enabled=ref(false),state=ref('off'),room=ref(null),notice=ref('');
  let socket=null,retry=null,renewTimer=null,pingTimer=null,generation=0,sentSeq=0,ready=false,lease='',restoreJob=null,retryDelay=1500,appearance=null;
  const closeSocket=()=>{clearTimeout(retry);clearTimeout(renewTimer);clearInterval(pingTimer);ready=false;if(socket){const current=socket;socket=null;current.close();}};
  const payload=()=>{const context=session.liveContext();if(!context?.run||context.run.guest||context.run.reason)throw Error('live_run_unavailable');return{context,body:{browser:context.browser,writer:context.writer,epoch:context.run.epoch}}};
  async function prefixPacket(events){const packed=await uploadBody(eventBytes(events));const body=packed.body instanceof Uint8Array?packed.body:new Uint8Array(packed.body);const packet=new Uint8Array(5+body.byteLength);packet.set([72,76,80,49],0);packet[4]=(packed.headers['Content-Encoding']==='gzip'?1:0)|(packed.headers['X-Human-Layout']==='planes5'?2:0);packet.set(body,5);return packet;}
  async function connect(descriptor){
    const own=++generation;closeSocket();lease=descriptor.lease;room.value=descriptor;state.value='connecting';sentSeq=0;
    const {context}=payload(),events=[...session.getEvents()];
    const packet=await prefixPacket(events);if(own!==generation||!enabled.value)return;
    const ws=new WebSocket(descriptor.publish_url);socket=ws;ws.binaryType='arraybuffer';
    ws.onopen=()=>{if(socket!==ws)return;ws.send(JSON.stringify({type:'hello',lease,seq:events.length,started_at:(context.run.firstMoveAt||Date.now())/1000,appearance,best_score:Number(getBest(context.run.variant)||0)}));ws.send(packet)};
    ws.onmessage=event=>{if(socket!==ws)return;try{const data=JSON.parse(event.data);if(data.type==='ready'){ready=true;retryDelay=1500;sentSeq=data.seq;state.value='live';notice.value='';scheduleRenew();clearInterval(pingTimer);pingTimer=setInterval(()=>{if(ws.readyState===1)ws.send(JSON.stringify({type:'ping'}))},10000);publishTail();}}catch{ws.close();}};
    ws.onclose=()=>{if(socket===ws)socket=null;ready=false;if(enabled.value&&own===generation)scheduleReconnect()};
  }
  async function renew(){const {body}=payload();const descriptor=await json(`/api/human/live/${room.value.room_id}/lease`,{method:'POST',body});lease=descriptor.lease;room.value=descriptor;if(socket?.readyState===1)socket.send(JSON.stringify({type:'renew',lease}));scheduleRenew();}
  function scheduleRenew(){clearTimeout(renewTimer);renewTimer=setTimeout(()=>renew().catch(()=>socket?.close()),45000)}
  async function reconnect(){if(!enabled.value)return;const {context,body}=payload();const descriptor=await json(`/api/human/runs/${context.run.id}/live`,{method:'POST',body});await connect(descriptor)}
  function scheduleReconnect(){
    if(!enabled.value)return;clearTimeout(retry);state.value='reconnecting';const wait=retryDelay;retryDelay=Math.min(15000,Math.round(retryDelay*1.8));
    retry=setTimeout(async()=>{if(!enabled.value)return;try{await reconnect()}catch(error){notice.value=liveError(error);scheduleReconnect()}},wait);
  }
  function fail(error){state.value='error';notice.value=liveError(error)}
  async function start(){if(enabled.value&&state.value==='live')return;enabled.value=true;notice.value='';try{await reconnect()}catch(error){fail(error);enabled.value=false;throw error}}
  async function stop(){enabled.value=false;state.value='off';generation++;closeSocket();const id=room.value?.room_id;room.value=null;if(id)await json(`/api/human/live/${id}`,{method:'DELETE'}).catch(()=>{})}
  async function restore(){
    if(enabled.value)return;if(restoreJob)return restoreJob;
    restoreJob=(async()=>{try{const current=await json('/api/human/live/current');if(!current.room)return;const context=session.liveContext();if(!context?.run||context.run.id!==current.room.run_id)return;enabled.value=true;
      for(let attempt=0;attempt<6;attempt++){try{await reconnect();return}catch(error){if(error?.code!=='live_run_not_current'||attempt===5)throw error;await new Promise(resolve=>setTimeout(resolve,250*(attempt+1)))}}
    }catch{enabled.value=false;state.value='off'}finally{restoreJob=null}})();return restoreJob;
  }
  function publishTail(){if(!enabled.value||!ready||socket?.readyState!==1)return;const events=session.getEvents();if(events.length<sentSeq){socket.close();return}for(let index=sentSeq;index<events.length;index++)socket.send(eventBytes([events[index]]));sentSeq=events.length;void session.liveCheckpoint()}
  async function runChanged(){if(!enabled.value)return;try{await reconnect()}catch(error){notice.value=liveError(error);scheduleReconnect()}}
  function finish(){if(ready&&socket?.readyState===1)socket.send(JSON.stringify({type:'end'}))}
  function updateAppearance(value){appearance=value;if(ready&&socket?.readyState===1)socket.send(JSON.stringify({type:'appearance',appearance}))}
  function updateBest(value){if(ready&&socket?.readyState===1)socket.send(JSON.stringify({type:'best',best_score:Number(value||0)}))}
  async function share(lang='zh'){if(!room.value)return;const text=liveShareText(session.liveContext(),room.value,lang);const copied=await copyLiveShareText(text);notice.value=copied?(lang==='zh'?'直播分享文案已复制':'Stream message copied'):(lang==='zh'?'复制失败，请手动复制直播间地址。':'Copy failed. Please copy the room URL manually.')}
  function dispose(){generation++;closeSocket()}
  return{enabled,state,room,notice,start,stop,restore,publishTail,runChanged,finish,share,updateAppearance,updateBest,dispose};
}
