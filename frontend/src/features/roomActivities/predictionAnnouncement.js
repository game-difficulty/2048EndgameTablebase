// Text segments are rendered with Vue interpolation, never user-supplied HTML.
export function predictionAnnouncementSegments(event, lang = 'zh') {
  const zh = lang === 'zh';
  const combo = event.tier === '65k+32k';
  const target = combo ? '65k＋32k' : '65k';
  const tokens = (Number(event.total_units || 0) / 1000).toLocaleString(zh ? 'zh-CN' : 'en-US', { maximumFractionDigits: 1 });
  const parts = [{text:'🎉 '},{text:event.player_name,kind:'player'},
    {text:zh ? ` 合出 ${target}！独立加注奖励${combo ? '升级' : ''}：` : ` made ${target}! Side-bet ${combo ? 'reward upgrade' : 'rewards'}: `}];
  (event.names || []).slice(0,3).forEach((name,index) => {
    if(index) parts.push({text:zh ? '、' : ', '});
    parts.push({text:name,kind:'name'});
  });
  if(event.recipient_count > 3) parts.push({text:zh ? `…等 ${event.recipient_count} 人` : `… (${event.recipient_count} viewers)`});
  parts.push({text:zh ? '共获得 ' : ' received '},{text:`${tokens} Token`,kind:'amount'},{text:zh ? '。' : ' in total.'});
  return parts;
}

export const predictionAnnouncement = (event, lang = 'zh') => predictionAnnouncementSegments(event, lang).map(part => part.text).join('');
