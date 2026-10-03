// Both spreadsheet and advanced imports use the existing atomic roster API.
function rowError(row, detail) {
  throw new Error(`第 ${row} 行：${detail}`);
}

function cells(line, row) {
  const separator = line.includes('\t') ? /\t/ : /[,，]/;
  const result = [];
  let value = '', quoted = false, closed = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (quoted) {
      if (char === '"' && line[i + 1] === '"') { value += '"'; i++; }
      else if (char === '"') { quoted = false; closed = true; }
      else value += char;
    } else if (separator.test(char)) {
      result.push(value.trim()); value = ''; closed = false;
    } else if (char === '"' && !value.trim() && !closed) {
      value = ''; quoted = true;
    } else {
      if (closed && char.trim()) rowError(row, '引号后的内容无效。');
      value += char;
    }
  }
  if (quoted) rowError(row, '引号未闭合。');
  result.push(value.trim());
  return result;
}

const teamHeader = values => /^(队名|队伍名称|team(?: name)?)$/i.test(values[0])
  && values.length > 1 && values.slice(1).every(value => /^(队长|队员\d*|选手\d*|captain|(?:player|member)\s*\d*)$/i.test(value));

export function parseRosterImport(text, { format = 'teams', identityMode = 'username', teamSize = 3 } = {}) {
  const entries = [], sourceRows = [], teams = new Set();
  if (!['teams', 'advanced'].includes(format) || !['username', 'user_id'].includes(identityMode)) throw new Error('请选择有效的导入方式。');
  if (!Number.isInteger(teamSize) || teamSize < 1 || teamSize > 16) throw new Error('本赛事的队伍人数无效。');
  const identity = (value, row) => {
    if (!value || (identityMode === 'user_id' && (!/^[1-9]\d*$/.test(value) || !Number.isSafeInteger(Number(value))))) rowError(row, '请填写有效的当前用户名或用户 ID。');
    return { [identityMode]: identityMode === 'username' ? value : Number(value) };
  };
  let first = true;
  for (const [index, line] of text.replace(/^\uFEFF/, '').split(/\r\n|\n|\r/).entries()) {
    if (!line.trim()) continue;
    const row = index + 1, values = cells(line, row);
    if (format === 'teams') {
      // Ignore unused spreadsheet columns but never remove a missing player in the middle.
      while (values.length > 1 && !values.at(-1)) values.pop();
      if (first && teamHeader(values)) { first = false; continue; }
      first = false;
      const [team, ...players] = values;
      if (!team) rowError(row, '请填写队名。');
      if (teams.has(team)) rowError(row, '队名重复，请将同一队的选手放在一行。');
      if (players.length !== teamSize || players.some(player => !player)) rowError(row, `每队须填写 ${teamSize} 位选手。`);
      teams.add(team);
      players.forEach((player, i) => {
        entries.push({ ...identity(player, row), team_name: team, is_external: false, captain: i === 0, position: i + 1 });
        sourceRows.push(row);
      });
    } else {
      const [id, team = '', ext = '0', cap = '0', pos = '', extra] = values;
      if (extra !== undefined || !['0', '1'].includes(ext) || !['0', '1'].includes(cap)
        || (pos && (!/^\d+$/.test(pos) || Number(pos) < 1 || Number(pos) > teamSize))) rowError(row, '请检查队名、外援0或1、队长0或1和队内序号。');
      entries.push({ ...identity(id, row), team_name: team, is_external: ext === '1', captain: cap === '1', position: pos ? Number(pos) : null });
      sourceRows.push(row);
    }
  }
  if (!entries.length || entries.length > 500) throw new Error('请提供 1–500 位选手。');
  return { entries, sourceRows };
}

export function selectRosterCaptain(entries, userId) {
  const selected = entries.find(entry => entry.user_id === userId);
  if (!selected?.team_name) return;
  const team = entries.filter(entry => entry.team_name === selected.team_name);
  if (selected.position != null && selected.position !== 1) {
    const first = team.find(entry => entry.position === 1);
    if (first) first.position = selected.position;
    selected.position = 1;
  }
  for (const entry of team) entry.captain = entry === selected;
}

export function rosterImportError(message, sourceRows) {
  return message.replace(/第 (\d+) 行/g, (match, number) => sourceRows[Number(number) - 1] ? `第 ${sourceRows[Number(number) - 1]} 行` : match);
}

export function rosterCsvCell(value) {
  const text = String(value);
  return /[,，\t\r\n"]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}
