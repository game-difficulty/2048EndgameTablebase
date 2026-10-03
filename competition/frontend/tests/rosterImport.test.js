import test from 'node:test';
import assert from 'node:assert/strict';
import { parseRosterImport, selectRosterCaptain, rosterImportError, rosterCsvCell } from '../src/rosterImport.js';

test('the seven-team spreadsheet expands into 21 players with captain and positions', () => {
  const text = `SSS\tsumeka\tSMCU\tstarcry_
PKU\tP-shiyi592\tps\t20230154
miles是啥子对\tmingsuovo\tqwertyuiop666100\t90-dedetoxaniith
SCL\tSaturn\tCQ_cris\tling1ap
鲜\tsynbsynb778\tyuan\t5033
荷塘月色\t鸠鸠爱摆烂\tmmmcccc\tautumn2
九月猫\t九九归一\tYQQH_miduokaese\tZearch Linro`;
  const { entries, sourceRows } = parseRosterImport(text);
  assert.equal(entries.length, 21);
  assert.deepEqual(entries.slice(0, 3), [
    { username: 'sumeka', team_name: 'SSS', is_external: false, captain: true, position: 1 },
    { username: 'SMCU', team_name: 'SSS', is_external: false, captain: false, position: 2 },
    { username: 'starcry_', team_name: 'SSS', is_external: false, captain: false, position: 3 },
  ]);
  assert.equal(entries[5].username, '20230154');
  assert.equal(entries[14].username, '5033');
  assert.equal(entries[20].username, 'Zearch Linro');
  assert.deepEqual(sourceRows.slice(-3), [7, 7, 7]);
  assert.equal(entries.filter(entry => entry.captain).length, 7);
});

test('Excel headers, CRLF, blank lines, and unused trailing columns retain physical source rows', () => {
  const { entries, sourceRows } = parseRosterImport('\uFEFF队名\t队长\t队员2\t队员3\r\n\r\nSSS\tOne\tTwo\tThree\t\t\r\n');
  assert.equal(entries.length, 3);
  assert.deepEqual(sourceRows, [3, 3, 3]);
  assert.equal(rosterImportError('第 2 行「Two」：找不到有效的当前用户名。未导入任何名单。', sourceRows), '第 3 行「Two」：找不到有效的当前用户名。未导入任何名单。');
});

test('comma and full-width comma rows support quoted usernames without splitting spaces', () => {
  const { entries } = parseRosterImport('"Team, A","One, A","Two ""B""",Zearch Linro\nTeam B，Four，Five，Six');
  assert.equal(entries[0].team_name, 'Team, A');
  assert.deepEqual(entries.slice(0, 3).map(entry => entry.username), ['One, A', 'Two "B"', 'Zearch Linro']);
  assert.equal(entries[3].username, 'Four');
  assert.throws(() => parseRosterImport('Team,"One,Two,Three'), /第 1 行：引号未闭合/);
});

test('invalid teams block parsing instead of importing partial rows', () => {
  for (const row of ['Team,One,Two', 'Team,One,,Three', 'Team,One,Two,Three,Four']) {
    assert.throws(() => parseRosterImport(`Valid,One,Two,Three\n${row}`), /第 2 行：每队须填写 3 位选手/);
  }
  assert.throws(() => parseRosterImport('Team,One,Two,Three\nTeam,Four,Five,Six'), /队名重复/);
  assert.throws(() => parseRosterImport(',One,Two,Three'), /请填写队名/);
  assert.throws(() => parseRosterImport(''), /1–500/);
});

test('team size comes from the event, and numeric usernames become IDs only in ID mode', () => {
  const input = 'Team,1,2,3,4,5';
  assert.deepEqual(parseRosterImport(input, { teamSize: 5 }).entries.map(entry => entry.position), [1, 2, 3, 4, 5]);
  assert.equal(parseRosterImport(input, { teamSize: 5 }).entries[0].username, '1');
  assert.equal(parseRosterImport(input, { teamSize: 5, identityMode: 'user_id' }).entries[0].user_id, 1);
  assert.throws(() => parseRosterImport('Team,1,2,0', { identityMode: 'user_id' }), /有效的当前用户名或用户 ID/);
  assert.throws(() => parseRosterImport('Team,1,2,9007199254740992', { identityMode: 'user_id' }), /有效的当前用户名或用户 ID/);
});

test('advanced import retains unassigned entries and guest/captain/position flags', () => {
  const { entries } = parseRosterImport('Player One\nGuest,Team,1,0,5\nCaptain,Team,0,1,1', { format: 'advanced', teamSize: 5 });
  assert.deepEqual(entries[0], { username: 'Player One', team_name: '', is_external: false, captain: false, position: null });
  assert.equal(entries[1].is_external, true);
  assert.equal(entries[1].position, 5);
  assert.equal(entries[2].captain, true);
});

test('changing captain swaps position 1 and preserves guests and other teams', () => {
  const entries = parseRosterImport('A,1,2,3\nB,4,5,6', { identityMode: 'user_id' }).entries;
  entries[2].is_external = true;
  selectRosterCaptain(entries, 3);
  assert.deepEqual(entries.slice(0, 3).map(entry => [entry.position, entry.captain]), [[3, false], [2, false], [1, true]]);
  assert.equal(entries[2].is_external, true);
  assert.equal(entries[3].captain, true);
  selectRosterCaptain(entries, 2);
  assert.deepEqual(entries.slice(0, 3).map(entry => [entry.position, entry.captain]), [[3, false], [1, true], [2, false]]);
});

test('loading an existing roster preserves punctuation in team names', () => {
  const csv = [123, 'Team, "A"', 0, 1, 1].map(rosterCsvCell).join(',');
  const { entries } = parseRosterImport(csv, { format: 'advanced', identityMode: 'user_id' });
  assert.equal(entries[0].team_name, 'Team, "A"');
  assert.equal(entries[0].user_id, 123);
});
