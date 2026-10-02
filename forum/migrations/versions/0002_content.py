"""Attachments, subscriptions, operations and durable delivery."""

from pathlib import Path
from alembic import op

revision = "0002_content"
down_revision = "0001_forum"
branch_labels = None
depends_on = None


def upgrade():
    for statement in (
        Path(__file__).with_suffix(".sql").read_text(encoding="utf-8").split(";\n")
    ):
        if statement.strip():
            op.execute(statement.strip())


def downgrade():
    raise RuntimeError(
        "Restore a verified backup into a separate database; content must not be dropped."
    )
