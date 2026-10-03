// Frontend presentation registry. Backend permissions and rules remain authoritative.
export const PUBLIC_ROOM_HOME = '/duels';
export const PUBLIC_ROOM_MODES = Object.freeze([
  { kind:'duel', title:['项目对决','Project duel'], tag:'01 · PROJECT SERIES',
    description:['选择一组小游戏，与对手依次对决。','Choose a sequence of projects and play them head to head.'],
    detail:['1–15 个项目 · 依次对战 · 比较总比分','1–15 projects · Fixed order · Series score'],
    clockLabel:['每人总用时（分钟）','Time per player (minutes)'], defaultMinutes:30, minMinutes:1, step:1,
    createMethod:'createDuel', defaults:()=>({projects:[]}),
    valid:d=>Array.isArray(d.projects) && d.projects.length>0 && d.projects.length<=15 && new Set(d.projects).size===d.projects.length,
    payload:d=>({projects:[...d.projects]}),
    rules:['全部项目共用总用时，沿用 Higher 项目的差额补时规则。项目顺序在创建后固定。','All projects share the time budget, with the existing Higher time-refund rule. The order is fixed on creation.'],
    summary:(d,en)=>`${d.projects.length} ${en?'projects · Fixed order':'个项目 · 依次对战'}`,
    showProjectOrder:true,
    waitingClock:['打满全部项目，允许平局。每人总用时：','Play all projects. Draws are allowed. Time per player:'] },
  { kind:'time_attack', title:['限时竞速','Time attack'], tag:'02 · PERSONAL BEST',
    description:['在固定时间内无限重开，以最快有效单局决胜。','Restart within a shared time limit. Your fastest valid run wins.'],
    detail:['4 种棋盘 · 自定目标 · 比较最佳 PB','4 boards · Custom target · Best run'],
    clockLabel:['比赛总时长（分钟）','Match time limit (minutes)'], defaultMinutes:10, minMinutes:.5, step:.5,
    createMethod:'createTimeAttack', defaults:()=>({variant:'4x4',target_kind:'tile',target_value:2048}),
    valid:d=>['4x4','3x4','2x4','3x3'].includes(d.variant) && Number.isInteger(Number(d.target_value)) && Number(d.target_value)<=2147483648 &&
      (d.target_kind==='tile' ? Number(d.target_value)>=8 && Number.isInteger(Math.log2(Number(d.target_value)))
        : d.target_kind==='board_sum' && Number(d.target_value)>=10 && Number(d.target_value)%2===0),
    payload:d=>({variant:d.variant,target_kind:d.target_kind,target_value:Number(d.target_value)}),
    rules:['无限重开，不可悔棋。单局从服务端生成棋盘时计时；PB 包含网络耗时，操作须在截止前送达。断线不暂停，达标后仍可继续挑战。','Unlimited restarts, no undo. Runs start when the server creates the board. PB includes network latency; moves must arrive before the deadline. Disconnects do not pause time. Keep playing after reaching the target.'],
    summary:(d,en)=>`${d.variant.replace('x',' × ')} · ${d.target_kind==='tile'?(en?'Tile ≥':'数字 ≥'):(en?'Exact sum =':'精确盘面和 =')} ${d.target_value}`,
    showProjectOrder:false,
    waitingClock:['共同比赛时限：','Shared time limit:'] },
]);
export const publicRoomMode = kind => PUBLIC_ROOM_MODES.find(mode=>mode.kind===kind);
export const isPublicRoom = room => !!publicRoomMode(room?.room_kind);
export const isOpenRoom = room => !['FINISHED','CANCELLED'].includes(room.status);
export const publicLobbyRoute = path => /^\/(duels|time-attacks)\/?$/.test(path);
export const legacyMode = path => /^\/time-attacks\/?$/.test(path) ? 'time_attack' : null;
export const modeText = (pair, language) => pair?.[language==='en'?1:0] || '';
export function publicRoomStatus(status, language) {
  const en=language==='en';
  if(['CREATED','SEATING','READY_CHECK'].includes(status))return en?'Waiting for players':'等待准备';
  if(status==='FINISHED')return en?'Finished':'已结束';
  if(status==='CANCELLED')return en?'Closed':'已关闭';
  if(/_READY$/.test(status))return en?'Between games':'局间准备';
  if(/_RESULT$/.test(status))return en?'Game result':'单局结算';
  return en?'In progress':'进行中';
}
export function creationValid(mode, draft, minutes, name) {
  const seconds=Number(minutes)*60;
  return !!mode && mode.valid(draft) && Number.isInteger(seconds) && seconds>=mode.minMinutes*60 && seconds<=86400
    && (String(name).trim().length===0 || String(name).trim().length>=2) && String(name).trim().length<=100;
}

export async function personalPublicRooms(rooms, readRoom) {
  const candidates=rooms.filter(isPublicRoom), result=[];
  let next=0;
  // Privileged listings may contain everyone; do not mistake those rooms for
  // the viewer's occupancy. Owners always retain their own seat in public rooms.
  await Promise.all(Array.from({length:Math.min(4,candidates.length)},async()=>{
    while(next<candidates.length){
      const row=candidates[next++], detail=await readRoom(row.room_code);
      if(detail.me?.seat)result.push({...row,status:detail.status,can_close:detail.me.can_close});
    }
  }));
  return result;
}
