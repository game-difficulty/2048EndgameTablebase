// Imported only by the loopback Playwright QA session; not part of the app bundle.
import { createApp, h, shallowRef, ref } from 'vue';
import ObservedProjectBoard from '../../shared/ObservedProjectBoard.vue';
import CompetitionMatchContent from '../../../frontend/src/live/content/CompetitionMatchContent.vue';
import { MatchRuntime } from '../src/projects/matchRuntime.js';
import { ProjectStreamSender, PROJECT_STREAM_PROTOCOL } from '../src/projects/projectStream.js';
import { receivedProjectView } from '../../shared/projectStateOrder.mjs';
const delay = ms => new Promise(resolve=>setTimeout(resolve,ms));

export async function mountStreamQA({slowSide='white'}={}) {
  const boot=await (await fetch('http://127.0.0.1:8789/qa/bootstrap')).json();
  const views={yellow:shallowRef(null),white:shallowRef(null)},seen={yellow:[],white:[]},acks={},errors=[],diagnostics=[];
  const live=ref();
  const root=document.createElement('div');root.id='stream-qa';root.style.cssText='position:absolute;inset:0;z-index:9999;background:#17243a;color:white;overflow:auto';document.body.append(root);
  const app=createApp({render:()=>h('div',[
    h('div',{style:'display:flex;height:360px'},['yellow','white'].map(side=>h('section',{style:'width:45%;padding:10px'},[
      h('b',side),h(ObservedProjectBoard,{view:views[side].value,streamKey:side,
        onPresent:view=>{if(view&&seen[side].at(-1)!==view.sequence)seen[side].push(view.sequence)},
        onGap:()=>viewer.readyState===1&&viewer.send(JSON.stringify({type:'room.resync'}))})]))),
    h('div',{style:'height:720px'},[h(CompetitionMatchContent,{ref:live,lang:'zh'})])])});
  app.config.globalProperties.$t=text=>text;app.mount(root);
  const liveSeen={yellow:[],white:[]};
  const observer=new MutationObserver(()=>{
    [...root.querySelectorAll('.stream-project [data-frame-sequence]')].forEach((node,index)=>{
      const list=liveSeen[['yellow','white'][index]],sequence=Number(node.dataset.frameSequence);
      if(list&&list.at(-1)!==sequence)list.push(sequence);
    });
  });
  observer.observe(root,{subtree:true,childList:true,attributes:true,attributeFilter:['data-frame-sequence']});
  const viewer=new WebSocket('ws://127.0.0.1:8789/ws/rooms/MATCH5?dev_user='+boot.viewer);
  viewer.onmessage=event=>{const m=JSON.parse(event.data);
    if(m.type==='room.snapshot')for(const side of ['yellow','white']){const next=m.data.match?.sessions?.[side]?.public_view;if(next)views[side].value=receivedProjectView(views[side].value,next)}
    if(m.type==='project.snapshot'){const side=m.data.side;views[side].value=receivedProjectView(views[side].value,m.data.public_view)}
  };
  const relay=new WebSocket('ws://127.0.0.1:8789/qa/live');
  let liveReceived={};
  relay.onmessage=event=>{const m=JSON.parse(event.data);liveReceived=m.match?.project_public_views||{};live.value?.receive(m)};
  const senders=[],runtimes=[];
  for(let i=0;i<2;i++){
    const info=boot.players[i],bootstrap=info.room.match.my_session.runtime,side=bootstrap.side;
    const runtime=new MatchRuntime(bootstrap);runtimes.push(runtime);
    function connect(){
      const native=new WebSocket('ws://127.0.0.1:8789/ws/projects/MATCH5');
      if(side!==slowSide)return native;
      const wrapper={get readyState(){return native.readyState},get bufferedAmount(){return native.bufferedAmount},
        send:text=>setTimeout(()=>{if(native.readyState===1)native.send(text)},300),close:(code,reason)=>native.close(code,reason)};
      for(const name of ['open','message','close','error'])native.addEventListener(name,event=>setTimeout(()=>wrapper['on'+name]?.(event),name==='message'?300:0));
      return wrapper;
    }
    const sender=new ProjectStreamSender({createSocket:connect,
      authenticate:()=>({protocol:PROJECT_STREAM_PROTOCOL,instance_id:bootstrap.instance_id,dev_user:String(info.user)}),
      phaseToken:()=>info.room.match.phase_token,onAck:ack=>acks[side]=ack.accepted_sequence,
      onError:error=>errors.push({side,code:error.code,status:error.status}),onDiagnostic:d=>diagnostics.push({side,...d})});
    senders.push(sender);sender.push(runtime.accept());
  }
  const result={seen,liveSeen,acks,errors,diagnostics,runtimes,senders,views,root,liveReceived:()=>liveReceived,
    close(){observer.disconnect();senders.forEach(s=>s.close());viewer.close();relay.close();app.unmount();root.remove()}};
  result.run=async(count=90)=>{
    for(let n=0;n<count;n++){
      for(let i=0;i<2;i++){
        const runtime=runtimes[i];let packet=null;
        for(let j=0;j<4&&!packet;j++)packet=await runtime.move(['left','up','right','down'][(n+j+i)%4]);
        if(!packet)packet=runtime.action('undo');
        if(packet)senders[i].push(packet);
      }
      if(n===40)senders[slowSide==='yellow'?0:1].socket.close(4001,'qa_disconnect');
      await delay(35);
    }
    await delay(5500);
    return {sequences:runtimes.map(r=>r.sequence),acks,errors,seen,liveSeen,
      liveSequences:Object.fromEntries(Object.entries(liveReceived).map(([s,v])=>[s,v.sequence])),
      liveDisplayed:[...root.querySelectorAll('.stream-project>small')].map(x=>x.textContent),
      livePlayed:[...root.querySelectorAll('.stream-project [data-frame-sequence]')].map(x=>Number(x.dataset.frameSequence)),
      maxOutstanding:Math.max(...diagnostics.map(d=>d.outstanding||0)),
      gaps:Object.fromEntries(Object.entries(seen).map(([s,list])=>[s,list.filter((seq,i)=>i&&seq!==list[i-1]+1)])),
      liveGaps:Object.fromEntries(Object.entries(liveSeen).map(([s,list])=>[s,list.filter((seq,i)=>i&&seq!==list[i-1]+1)]))};
  };
  return result;
}
