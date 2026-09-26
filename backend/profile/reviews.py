"""Post-publication review; only new profile changes enter the queue."""
from backend.auth.db import auth_db
from .service import iso
from .validation import canonical_display_name_key
from .storage import delete_avatar_file


def list_reviews(page=1, status='pending'):
    page = max(1, int(page))
    where = '' if status == 'all' else 'WHERE r.status = ?'
    params = () if status == 'all' else (status,)
    with auth_db() as db:
        total = db.execute(f'SELECT COUNT(*) FROM profile_change_reviews r {where}', params).fetchone()[0]
        rows = db.execute(f'''SELECT e.id, e.user_id, e.change_type, e.old_value, e.new_value,
            e.ip_address, e.created_at,
            u.display_name, p.avatar_key, r.status, r.reviewed_at FROM profile_change_reviews r
            JOIN user_profile_change_events e ON e.id=r.event_id JOIN users u ON u.id=e.user_id
            JOIN user_profiles p ON p.user_id=e.user_id
            {where} ORDER BY e.id DESC LIMIT 20 OFFSET ?''', (*params, (page - 1) * 20)).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item['is_current'] = (item.pop('avatar_key') == item['new_value']
                              if item['change_type'] == 'avatar'
                              else item['display_name'] == item['new_value'])
        items.append(item)
    return dict(items=items, total=total)


def decide_review(event_id, action, admin_id):
    if action not in {'keep', 'revoke'}:
        raise ValueError('Invalid review action')
    old_key = None
    with auth_db() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('''SELECT e.*, r.status FROM profile_change_reviews r
            JOIN user_profile_change_events e ON e.id=r.event_id WHERE e.id=?''', (event_id,)).fetchone()
        if row is None:
            raise ValueError('Review not found')
        if row['status'] in {'revoked', 'superseded'}:
            return {'status': row['status']}
        status, now = 'reviewed', iso()
        if action == 'revoke':
            profile = db.execute('SELECT * FROM user_profiles WHERE user_id=?', (row['user_id'],)).fetchone()
            user = db.execute('SELECT * FROM users WHERE id=?', (row['user_id'],)).fetchone()
            kinds = ('avatar', 'avatar_remove', 'avatar_cooldown_reset') if row['change_type'] == 'avatar' else ('display_name', 'display_name_admin_reset')
            placeholders = ','.join('?' for _ in kinds)
            latest = db.execute(f'''SELECT MAX(id) FROM user_profile_change_events
                WHERE user_id=? AND change_type IN ({placeholders})''', (row['user_id'], *kinds)).fetchone()[0]
            current = profile['avatar_key'] if row['change_type'] == 'avatar' else user['display_name']
            if latest != event_id or current != row['new_value']:
                status = 'superseded'
            elif row['change_type'] == 'avatar':
                old_key = current
                db.execute('UPDATE user_profiles SET avatar_key=NULL, avatar_sha256=NULL, updated_at=? WHERE user_id=?', (now, row['user_id']))
                status = 'revoked'
            else:
                name, suffix = f"User{row['user_id']}", 0
                while db.execute('SELECT 1 FROM users WHERE display_name_key=? AND id!=?',
                                 (canonical_display_name_key(name), row['user_id'])).fetchone():
                    suffix += 1
                    name = f"User{row['user_id']}_{suffix}"
                db.execute('UPDATE users SET display_name=?, display_name_key=?, updated_at=? WHERE id=?',
                           (name, canonical_display_name_key(name), now, row['user_id']))
                db.execute('UPDATE leaderboard_entries SET display_name=? WHERE user_id=?', (name, row['user_id']))
                status = 'revoked'
            if status == 'revoked':
                db.execute('''INSERT INTO user_profile_change_events
                    (user_id, change_type, old_value, new_value, ip_address, user_agent, created_at)
                    VALUES (?, ?, ?, ?, '', 'admin profile review', ?)''',
                    (row['user_id'], 'avatar_remove' if row['change_type'] == 'avatar' else 'display_name_admin_reset',
                     row['new_value'], None if row['change_type'] == 'avatar' else name, now))
        db.execute('UPDATE profile_change_reviews SET status=?, reviewed_by=?, reviewed_at=? WHERE event_id=?',
                   (status, admin_id, now, event_id))
    if old_key:
        delete_avatar_file(old_key)
    return {'status': status}
