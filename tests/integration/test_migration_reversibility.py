import os
import subprocess

import pytest
from sqlalchemy import create_engine, inspect, text

from ariadne.settings import get_settings

pytestmark = pytest.mark.integration

SCRATCH_DB = "ariadne_migration_check"


def _alembic(command: str, url: str) -> None:
    subprocess.run(
        ["alembic", *command.split()],
        check=True,
        env={**os.environ, "ALEMBIC_DATABASE_URL": url},
        capture_output=True,
    )


def test_migrations_apply_and_reverse_cleanly():
    """A migration that cannot be reversed is a migration nobody can safely iterate on."""
    base = get_settings().test_database_url.rsplit("/", 1)[0]
    url = f"{base}/{SCRATCH_DB}"

    admin = create_engine(f"{base}/postgres", isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)'))
        conn.execute(text(f'CREATE DATABASE "{SCRATCH_DB}"'))

    try:
        _alembic("upgrade head", url)
        engine = create_engine(url)
        assert "films" in inspect(engine).get_table_names()
        engine.dispose()

        _alembic("downgrade base", url)
        engine = create_engine(url)
        remaining = set(inspect(engine).get_table_names()) - {"alembic_version"}
        engine.dispose()
        assert remaining == set()
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)'))
        admin.dispose()
