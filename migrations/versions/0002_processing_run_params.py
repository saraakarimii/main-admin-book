"""Persist Notebook-v2 parameter set per processing run."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0002_run_params"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

def upgrade():
    bind = op.get_bind()
    columns = {c["name"] for c in inspect(bind).get_columns("processing_runs")}
    if "params_json" not in columns:
        op.add_column("processing_runs", sa.Column("params_json", sa.Text(), nullable=False, server_default="{}"))

def downgrade():
    bind = op.get_bind()
    columns = {c["name"] for c in inspect(bind).get_columns("processing_runs")}
    if "params_json" in columns:
        op.drop_column("processing_runs", "params_json")
