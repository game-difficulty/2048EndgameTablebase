"""Social relationships, private reading state, reply drafts and appeals."""

from pathlib import Path
from alembic import op

revision = "0003_social"
down_revision = "0002_content"
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
