import { liveTileColors } from '../tilePalette.js';
import { participantName, showEndNotice } from './participants.js';

// Optional compatibility renderer for classic 2048 content. Browser/session
// ownership stays in the room; other content types can supply a different renderer.
export function boardPipFrame({ slots, selectedLane = 0, layout = 'focus', state, lang, title, now = Date.now() }) {
  const key = JSON.stringify([layout, selectedLane, state, lang, title,
    slots.map(s => [s.lane, s.name, s.status, s.run?.run_id, s.run?.seq, showEndNotice(s.run,now), s.run?.source])]);
  const t = (zh, en) => lang === 'zh' ? zh : en;
  return { key, width: 960, height: 540, draw(ctx) {
    ctx.fillStyle = '#0f172a'; ctx.fillRect(0, 0, 960, 540);
    ctx.textBaseline = 'middle'; ctx.textAlign = 'left'; ctx.fillStyle = '#fff'; ctx.font = 'bold 20px sans-serif';
    ctx.fillText(title || '2048 LIVE', 18, 24, 740);
    const render = (slot, x, y, size) => {
      const run = slot?.run, unit = size / 480;
      ctx.fillStyle = '#fff'; ctx.textAlign = 'left'; ctx.font = `bold ${Math.max(12, size / 22)}px sans-serif`;
      ctx.fillText(`${participantName(slot)} · ${Number(run?.score || 0).toLocaleString()}`, x, y - 15, size);
      ctx.fillStyle = '#1e293b'; ctx.fillRect(x,y,size,size);
      for (let i=0; i<16; i++) {
        const value=run?.board[i] || 0, colors=liveTileColors(value);
        const tx=x+(12+i%4*117)*unit, ty=y+(12+Math.floor(i/4)*117)*unit;
        ctx.fillStyle=value ? colors.background : '#182333'; ctx.fillRect(tx,ty,105*unit,105*unit);
        if (!value) continue;
        ctx.fillStyle=colors.color; ctx.font=`bold ${(value>=10000?24:value>=1000?32:40)*unit}px sans-serif`;
        ctx.textAlign='center'; ctx.fillText(String(value),tx+52.5*unit,ty+54*unit);
      }
      const label = run?.ended_at ? (showEndNotice(run,now) ? t('本局结束','Game over') : '') : state !== 'live' ? ({paused:t('直播已暂停','Paused'),offline:t('等待主播','Offline'),loading:t('正在连接','Connecting'),reconnecting:t('正在重连','Reconnecting')}[state] || state)
        : !run || slot.status === 'recovering' ? t('AI 正在准备','Preparing AI') : '';
      if (label) {
        ctx.fillStyle='#000b'; ctx.fillRect(x,y+size*.4,size,size*.2);
        ctx.fillStyle='#fff'; ctx.font=`bold ${Math.max(12,size*.055)}px sans-serif`; ctx.textAlign='center';
        ctx.fillText(label,x+size/2,y+size/2,size-12);
      }
    };
    if (slots.length === 1) render(slots[0],245,65,470);
    else if (layout === 'equal') slots.forEach((slot,i)=>render(slot,15+i*315,135,300));
    else {
      render(slots.find(s=>s.lane===selectedLane) || slots[0],360,65,470);
      slots.filter(s=>s.lane!==selectedLane).forEach((slot,i)=>render(slot,125,75+i*240,210));
    }
  }};
}
