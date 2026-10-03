"""Create a PostgreSQL custom-format backup and verify it in a fresh disposable DB.

Only Docker-local PostgreSQL is supported by this helper. No production deletion,
backup expiry, or overwrite of an existing database/file is performed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from uuid import uuid4
import psycopg
from psycopg import sql
from sqlalchemy.engine import make_url


def run(args, env=None):
    result = subprocess.run(args, env=env, capture_output=True)
    if result.returncode:
        # Do not echo commands, connection strings, or credentials.
        raise RuntimeError(
            "PostgreSQL backup/restore command failed; exit code "
            + str(result.returncode)
        )
    return result.stdout


def inventory(conn):
    tables = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname='public' AND (tablename LIKE 'forum_%' OR tablename='alembic_version') ORDER BY tablename"
    ).fetchall()
    result = {}
    for (table,) in tables:
        # Compare canonical full-row digests, including attachment bytes and private permission state.
        digest = hashlib.sha256()
        count = 0
        query = sql.SQL(
            "SELECT row_to_json(t)::text FROM {} t ORDER BY row_to_json(t)::text"
        ).format(sql.Identifier(table))
        with conn.cursor(name="verify_" + uuid4().hex) as cursor:
            cursor.execute(query)
            for (row,) in cursor:
                digest.update(row.encode())
                digest.update(b"\n")
                count += 1
        result[table] = {"rows": count, "sha256": digest.hexdigest()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    url = make_url(os.environ["FORUM_DATABASE_URL"])
    if url.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("This helper only accepts a loopback PostgreSQL URL.")
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stamp = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    archive = out / ("forum-" + stamp + ".dump")
    if archive.exists():
        raise SystemExit("Refusing to overwrite backup.")
    remote = "/tmp/forum-backup-" + uuid4().hex + ".dump"
    database = "forum_restore_check_" + uuid4().hex
    assert re.fullmatch(r"forum_restore_check_[0-9a-f]{32}", database)
    env = {**os.environ, "PGPASSWORD": url.password or ""}
    prefix = ["docker", "exec", "-e", "PGPASSWORD", args.container]
    source_url = url.set(drivername="postgresql").render_as_string(hide_password=False)
    admin = psycopg.connect(source_url, autocommit=True)
    created = False
    try:
        # Export one snapshot so row verification and pg_dump see precisely the same state.
        with psycopg.connect(source_url) as source:
            source.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            snapshot = source.execute("SELECT pg_export_snapshot()").fetchone()[0]
            expected = inventory(source)
            run(
                prefix
                + [
                    "pg_dump",
                    "-U",
                    url.username,
                    "-d",
                    url.database,
                    "-Fc",
                    "--no-owner",
                    "--no-acl",
                    "--snapshot",
                    snapshot,
                    "-f",
                    remote,
                ],
                env,
            )
        run(["docker", "cp", args.container + ":" + remote, str(archive)])
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        created = True
        run(
            prefix
            + [
                "pg_restore",
                "-U",
                url.username,
                "-d",
                database,
                "--no-owner",
                "--no-acl",
                "--exit-on-error",
                remote,
            ],
            env,
        )
        with psycopg.connect(
            url.set(drivername="postgresql", database=database).render_as_string(
                hide_password=False
            )
        ) as restored:
            actual = inventory(restored)
        if actual != expected:
            raise RuntimeError(
                "Restored table inventory does not match exported snapshot."
            )
        report = {
            "verified": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "database": url.database,
            "archive": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "tables": actual,
        }
        report_path = archive.with_suffix(".verified.json")
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "verified": True,
                    "archive": str(archive),
                    "report": str(report_path),
                    "tables": len(actual),
                    "bytes": archive.stat().st_size,
                }
            )
        )
    finally:
        if created:
            assert (
                re.fullmatch(r"forum_restore_check_[0-9a-f]{32}", database)
                and database != url.database
            )
            admin.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                    sql.Identifier(database)
                )
            )
        admin.close()
        # Only remove this invocation's generated /tmp artifact, never an archive directory.
        if re.fullmatch(r"/tmp/forum-backup-[0-9a-f]{32}\.dump", remote):
            run(["docker", "exec", args.container, "rm", "--", remote])


if __name__ == "__main__":
    main()
