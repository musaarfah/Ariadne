import os
import subprocess
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session

from ariadne.settings import get_settings


def _admin_url(url: str) -> str:
    """Same server, but connected to the default database so we can create/drop others."""
    return url.rsplit("/", 1)[0] + "/postgres"


@pytest.fixture(scope="session")
def test_engine() -> Iterator[Engine]:
    """A scratch database built by running the real migrations.

    Migrations rather than create_all, so the test schema is the one that ships. A drift
    between models and migrations should fail here, not in production.
    """
    url = get_settings().test_database_url
    db_name = url.rsplit("/", 1)[1]

    admin = create_engine(_admin_url(url), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
        conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    admin.dispose()

    subprocess.run(
        ["alembic", "upgrade", "head"],
        check=True,
        env={**os.environ, "ALEMBIC_DATABASE_URL": url},
        capture_output=True,
    )

    engine = create_engine(url)
    yield engine
    engine.dispose()

    admin = create_engine(_admin_url(url), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture
def session(test_engine: Engine) -> Iterator[Session]:
    """A session whose writes are rolled back, so tests cannot leak into each other."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
