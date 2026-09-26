export function liveShareText(context, room, lang = 'zh') {
  const variant = String(context?.run?.variant || room?.variant || '4x4').replace('x', '×');
  const score = Number(context?.run?.score || 0).toLocaleString(lang === 'zh' ? 'zh-CN' : 'en-US');
  return lang === 'zh'
    ? `我正在直播 2048 ${variant} 对局，目前 ${score} 分。来看看这局能走多远！\n${room.url}`
    : `I'm streaming a 2048 ${variant} game at ${score} points. Come see how far this run goes!\n${room.url}`;
}

export async function copyLiveShareText(value, platform = navigator, page = document) {
  try { if (platform.clipboard?.writeText) { await platform.clipboard.writeText(value); return true; } } catch {}
  const field=page.createElement('textarea');field.value=value;field.readOnly=true;field.style.cssText='position:fixed;left:-9999px;top:0;opacity:0';page.body.appendChild(field);field.select();
  try{return page.execCommand('copy')}finally{field.remove()}
}
