"""Transactional SQLite constraint upgrade; preserve old matches and indexes."""
import re


def migrate_room_constraints(db):
    tables = ('competition_seats', 'competition_lineups', 'competition_match_control',
              'competition_game_readiness', 'competition_game_sessions', 'competition_game_results',
              'competition_result_confirmations', 'tournament_roster_positions')
    for table in tables:
        row = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if not row:
            continue
        old = row[0]
        new = re.sub(r'position BETWEEN 1 AND 3\b', 'position BETWEEN 1 AND 16', old)
        new = re.sub(r"\b(current_game_key|game_key) IN \('A', 'B', 'C'\)",
                     lambda match: f"length({match[1]}) = 1 AND {match[1]} BETWEEN 'A' AND 'O'", new)
        if table == 'competition_lineups':
            new = re.sub(r'UNIQUE\s*\(competition_id, side, position\)\s*,', '', new)
        if new == old:
            continue
        indexes = [r[0] for r in db.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", (table,))]
        triggers = [r[0] for r in db.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name=? AND sql IS NOT NULL", (table,))]
        temp = table + '_rules_upgrade'
        new = re.sub(r'CREATE TABLE\s+(?:IF NOT EXISTS\s+)?["`\[]?'+table+r'["`\]]?', 'CREATE TABLE '+temp, new, count=1, flags=re.I)
        db.execute(new)
        columns = ','.join('"'+r[1]+'"' for r in db.execute(f'PRAGMA table_info({table})'))
        db.execute(f'INSERT INTO {temp} ({columns}) SELECT {columns} FROM {table}')
        db.execute(f'DROP TABLE {table}')
        db.execute(f'ALTER TABLE {temp} RENAME TO {table}')
        for index in indexes + triggers:
            db.execute(index)
    violations = db.execute('PRAGMA foreign_key_check').fetchall()
    if violations:
        raise RuntimeError('Room rules migration violated foreign keys')
