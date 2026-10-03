"""Discovery, polls, Q&A, announcements, privacy and typed notifications."""

from pathlib import Path
from alembic import op

revision = "0004_operations"
down_revision = "0003_social"
branch_labels = None
depends_on = None


def upgrade():
    for statement in (
        Path(__file__).with_suffix(".sql").read_text(encoding="utf-8").split(";\n")
    ):
        if statement.strip():
            op.execute(statement.strip())


def downgrade():
    raise RuntimeError("Restore a verified backup into a separate database.")
