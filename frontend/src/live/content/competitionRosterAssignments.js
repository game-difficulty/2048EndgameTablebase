// A player can appear more than once in BO5 / BO7 lineups.
export function rosterAssignments(games, side, projectName) {
  const assignments = {};
  for (const game of games || []) {
    const position = game.players?.[side]?.position;
    if (!position) continue;
    const name = projectName(game.project_key).replace(/\s*[（(]\d+\s*[×x]\s*\d+[）)]\s*$/, '');
    (assignments[position] ||= []).push(`${game.game_key} · ${name}`);
  }
  return assignments;
}
