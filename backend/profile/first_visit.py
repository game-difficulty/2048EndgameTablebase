"""Versioned account-level first-visit outcomes. Only explicitly registered flows are accepted."""
import time
from backend.auth.db import auth_db

FLOWS = {"play_rules": {"version": 1, "states": {"acknowledged"}, "required": True}}

def read_flows(user_id):
    with auth_db() as db:
        rows = db.execute("SELECT flow,version,status,completed_at FROM account_first_visit WHERE user_id=?", (user_id,)).fetchall()
    saved = {(r["flow"], r["version"]): dict(r) for r in rows}
    return [{"id": key, "version": spec["version"], "required": spec["required"],
             "status": saved.get((key, spec["version"]), {}).get("status"),
             "completed_at": saved.get((key, spec["version"]), {}).get("completed_at")}
            for key, spec in FLOWS.items()]

def complete_flow(user_id, flow, version, status):
    spec = FLOWS.get(flow)
    if not spec or type(version) is not int or version != spec["version"] or status not in spec["states"]:
        raise ValueError("invalid_first_visit_flow")
    with auth_db() as db:
        db.execute("INSERT OR IGNORE INTO account_first_visit(user_id,flow,version,status,completed_at) VALUES (?,?,?,?,?)",
                   (user_id, flow, version, status, time.time()))
    return read_flows(user_id)
