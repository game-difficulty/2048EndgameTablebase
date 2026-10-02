import os
from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

url = os.environ.get("FORUM_DATABASE_URL", "")
if not url.startswith("postgresql+psycopg://"):
    raise RuntimeError("FORUM_DATABASE_URL must be a PostgreSQL psycopg URL")

if context.is_offline_mode():
    context.configure(url=url, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_engine(url, poolclass=NullPool, hide_parameters=True)
    with engine.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            connection.exec_driver_sql("SELECT pg_advisory_xact_lock(2048, 17001)")
            context.run_migrations()
    engine.dispose()
