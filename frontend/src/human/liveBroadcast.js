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

export function createLiveBroadcast(session, getBest = () => 0, request = json) {
  const enabled=ref(false),state=ref('off'),room=ref(null),notice=ref('');
  let socket=null,retry=null,renewTimer=null,pingTimer=null,generation=0,sentSeq=0,ready=false,lease='',restoreJob=null,retryDelay=1500,appearance=null,publishingId=null,pendingId=null,switchSupported=false,transitionJob=Promise.resolve(),pendingAck=null,resolveAck=null;
  const clearPending=()=>{resolveAck?.();resolveAck=null;pendingAck=null;pendingId=null;pendingPrefix=null;};
  const setPending=id=>{pendingId=id;pendingAck=new Promise(resolve=>{resolveAck=resolve;});};
  const closeSocket=()=>{clearTimeout(retry);clearTimeout(renewTimer);clearInterval(pingTimer);clearPending();ready=false;if(socket){const current=socket;socket=null;current.close();}};
  const payload=()=>{const context=session.liveContext();if(!context?.run||context.run.guest||context.run.reason)throw Error('live_run_unavailable');return{context,body:{browser:context.browser,writer:context.writer,epoch:context.run.epoch}}};
  async function prefixPacket(events){const packed=await uploadBody(eventBytes(events));const body=packed.body instanceof Uint8Array?packed.body:new Uint8Array(packed.body);const packet=new Uint8Array(5+body.byteLength);packet.set([72,76,80,49],0);packet[4]=(packed.headers['Content-Encoding']==='gzip'?1:0)|(packed.headers['X-Human-Layout']==='planes5'?2:0);packet.set(body,5);return packet;}
  const currentId=()=>session.liveContext()?.run?.id;
  const isCurrent=(own,id)=>enabled.value&&own===generation&&currentId()===id;
  const hello=(type,descriptor,context,events)=>({type,lease:descriptor.lease,seq:events.length,
    ...(descriptor.resume_supported ? {resume:true} : {}),
    started_at:(context.run.firstMoveAt||Date.now())/1000,appearance,best_score:Number(getBest(context.run.variant)||0)});
  let pendingPrefix=null, announcedBest=0;
  async function sendPrefix(ws, descriptor, context, own, type) {
    const events=session.getEvents(),id=context.run.id;
    pendingPrefix={ws,events,id,own};
    announcedBest=Number(getBest(context.run.variant)||0);
    if(descriptor.resume_supported){ws.send(JSON.stringify(hello(type,descriptor,context,events)));return;}
    const packet=await prefixPacket(events);
    if(socket!==ws||!isCurrent(own,id))return;
    ws.send(JSON.stringify(hello(type,descriptor,context,events)));ws.send(packet);
  }
  async function requestedPrefix(ws,data){
    const pending=pendingPrefix;
    if(!pending||pending.ws!==ws||data.run_id!==pending.id||!isCurrent(pending.own,pending.id))return;
    if(!Number.isInteger(data.start)||data.start<0||data.start>pending.events.length||data.seq!==pending.events.length)throw Error('invalid_live_resume');
    const packet=await prefixPacket(pending.events.slice(data.start));
    if(socket===ws&&pendingPrefix===pending&&isCurrent(pending.own,pending.id)){pendingPrefix=null;ws.send(packet);}
  }
  async function connect(descriptor,context,own){
    closeSocket();lease=descriptor.lease;room.value=descriptor;state.value='connecting';sentSeq=0;
    const id=context.run.id;if(!isCurrent(own,id))return;
    const ws=new WebSocket(descriptor.publish_url);socket=ws;setPending(id);switchSupported=false;ws.binaryType='arraybuffer';
    ws.onopen=()=>{if(socket!==ws||!isCurrent(own,id))return;void sendPrefix(ws,descriptor,context,own,'hello').catch(()=>ws.close())};
    ws.onmessage=event=>{if(socket!==ws)return;try{const data=JSON.parse(event.data);if(data.type==='prefix_request'){void requestedPrefix(ws,data).catch(()=>ws.close());return;}if(data.type==='ready'){
      if(data.run_id!==pendingId)return;
      clearPending();
      if(data.run_id!==currentId())return;
      publishingId=data.run_id;switchSupported=data.switch_supported===true;
      ready=true;retryDelay=1500;sentSeq=data.seq;state.value='live';notice.value='';scheduleRenew();
      clearInterval(pingTimer);pingTimer=setInterval(()=>{if(socket===ws&&ws.readyState===1)ws.send(JSON.stringify({type:'ping'}))},10000);publishTail();updateBest(getBest(session.liveContext()?.run?.variant));
    }}catch{ws.close();}};
    ws.onclose=()=>{if(socket!==ws)return;socket=null;clearPending();ready=false;if(enabled.value)scheduleReconnect()};
  }
  async function renew(){
    const own=generation,id=currentId(),ws=socket;
    try{
      const {body}=payload();
      const descriptor=await request(`/api/human/live/${room.value.room_id}/lease`,{method:'POST',body});
      if(!isCurrent(own,id)||socket!==ws||!ready||publishingId!==id)return;
      lease=descriptor.lease;room.value=descriptor;if(ws?.readyState===1)ws.send(JSON.stringify({type:'renew',lease}));scheduleRenew();
    }catch{if(isCurrent(own,id)&&socket===ws)ws?.close()}
  }
  function scheduleRenew(){clearTimeout(renewTimer);renewTimer=setTimeout(()=>void renew(),45000)}
  async function reconnect(){
    if(!enabled.value)return;
    const own=++generation,{context,body}=payload(),id=context.run.id;
    ready=false;clearTimeout(retry);clearTimeout(renewTimer);
    // Serialize leases and wait for the previous handoff: issuing the next lease
    // revokes the previous generation, so parallel requests can strand a valid switch.
    const previous=transitionJob;let release;
    transitionJob=new Promise(resolve=>{release=resolve;});
    await previous;
    try{
      if(!isCurrent(own,id))return;
      if(pendingAck){
        let timeout;
        try{await Promise.race([pendingAck,new Promise(resolve=>{timeout=setTimeout(()=>{socket?.close();resolve();},8000);})]);}
        finally{clearTimeout(timeout);}
      }
      if(!isCurrent(own,id))return;
      const descriptor=await request(`/api/human/runs/${id}/live`,{method:'POST',body});
      if(!isCurrent(own,id))return;
      if(socket?.readyState===1&&switchSupported){
        const ws=socket;
        if(!isCurrent(own,id)||socket!==ws)return;
        room.value=descriptor;lease=descriptor.lease;setPending(id);sentSeq=0;state.value='connecting';
        await sendPrefix(ws,descriptor,context,own,'switch');
      }else await connect(descriptor,context,own);
    }finally{release();}
  }

  function scheduleReconnect(){
    if(!enabled.value)return;clearTimeout(retry);state.value='reconnecting';const wait=retryDelay;retryDelay=Math.min(15000,Math.round(retryDelay*1.8));
    retry=setTimeout(async()=>{if(!enabled.value)return;try{await reconnect()}catch(error){notice.value=liveError(error);scheduleReconnect()}},wait);
  }
  function fail(error){state.value='error';notice.value=liveError(error)}
  async function start(){if(enabled.value&&state.value==='live')return;enabled.value=true;notice.value='';try{await reconnect()}catch(error){fail(error);enabled.value=false;throw error}}
  async function stop(){enabled.value=false;state.value='off';generation++;closeSocket();const id=room.value?.room_id;room.value=null;if(id)await request(`/api/human/live/${id}`,{method:'DELETE'}).catch(()=>{})}
  async function restore(){
    if(enabled.value)return;if(restoreJob)return restoreJob;
    restoreJob=(async()=>{try{const current=await request('/api/human/live/current');if(!current.room)return;const context=session.liveContext();if(!context?.run||context.run.id!==current.room.run_id)return;enabled.value=true;
      for(let attempt=0;attempt<6;attempt++){try{await reconnect();return}catch(error){if(error?.code!=='live_run_not_current'||attempt===5)throw error;await new Promise(resolve=>setTimeout(resolve,250*(attempt+1)))}}
    }catch{enabled.value=false;state.value='off'}finally{restoreJob=null}})();return restoreJob;
  }
  function publishTail(){if(!enabled.value||!ready||socket?.readyState!==1||currentId()!==publishingId)return;const count=session.getEventCount();if(count<sentSeq){socket.close();return}const tail=session.getEvents(sentSeq,count);for(const event of tail)socket.send(eventBytes([event]));sentSeq=count;void session.liveCheckpoint()}
  async function runChanged(){if(!enabled.value)return;const id=currentId();try{await reconnect()}catch(error){if(!enabled.value||id!==currentId())return;notice.value=liveError(error);scheduleReconnect()}}
  function finish(){if(ready&&currentId()===publishingId&&socket?.readyState===1)socket.send(JSON.stringify({type:'end'}))}
  function updateAppearance(value){appearance=value;if(ready&&socket?.readyState===1)socket.send(JSON.stringify({type:'appearance',appearance}))}
  function updateBest(value){
    const best=Number(value||0),score=Number(session.liveContext()?.run?.score||0);
    if(!ready||socket?.readyState!==1||currentId()!==publishingId||best<=Math.max(announcedBest,score))return;
    socket.send(JSON.stringify({type:'best',best_score:best}));announcedBest=best;
  }
  async function share(lang='zh'){if(!room.value)return;const text=liveShareText(session.liveContext(),room.value,lang);const copied=await copyLiveShareText(text);notice.value=copied?(lang==='zh'?'直播分享文案已复制':'Stream message copied'):(lang==='zh'?'复制失败，请手动复制直播间地址。':'Copy failed. Please copy the room URL manually.')}
  function dispose(){enabled.value=false;generation++;closeSocket()}
  return{enabled,state,room,notice,start,stop,restore,publishTail,runChanged,finish,share,updateAppearance,updateBest,dispose};
}
