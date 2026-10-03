"""Preserve the system category on public announcement notifications."""

from alembic import op

revision = "0005_notice_category"
down_revision = "0004_operations"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE forum_notifications DROP CONSTRAINT forum_notifications_kind_check"
    )
    op.execute(
        "ALTER TABLE forum_notifications ADD CONSTRAINT forum_notifications_kind_check CHECK(kind IN ('reply','subscription','follow','mention','system'))"
    )


def downgrade():
    raise RuntimeError("Restore a verified backup into a separate database.")
