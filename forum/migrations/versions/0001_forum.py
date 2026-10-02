"""Forum foundation; frozen SQL is intentionally independent of application models."""
from pathlib import Path
from alembic import op

revision = "0001_forum"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    sql = Path(__file__).with_suffix(".sql").read_text(encoding="utf-8")
    for statement in sql.split(";\n"):
        if statement.strip():
            op.execute(statement.strip())


def downgrade():
    # An application rollback must never silently delete community content.
    raise RuntimeError("Destructive downgrade is disabled. Restore a verified backup into a separate database.")
