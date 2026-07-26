import time
from pathlib import Path

import typer
from redis import Redis
from sqlalchemy import inspect, text

from ariadne.core.catalog.fixtures import FIXTURE_DIR, capture_all
from ariadne.core.catalog.tmdb import TmdbAuthError, TmdbClient, TmdbError
from ariadne.core.ingest.export import parse_export
from ariadne.core.ingest.persist import persist_export
from ariadne.db.session import get_engine, session_scope
from ariadne.settings import get_settings

app = typer.Typer(help="Ariadne — find the below-the-line crew that explains your film taste.")


@app.callback()
def _root() -> None:
    """Present so Typer keeps subcommand dispatch with only one command registered."""


@app.command("db-check")
def db_check() -> None:
    """Verify Postgres and Redis are reachable and the schema is applied."""
    settings = get_settings()
    failures: list[str] = []

    try:
        engine = get_engine()
        with engine.connect() as conn:
            version = conn.execute(text("SHOW server_version")).scalar_one()
            has_trgm = conn.execute(
                text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')")
            ).scalar_one()
        tables = sorted(inspect(engine).get_table_names())

        typer.echo(f"postgres    ok   server {version}")
        typer.echo(f"pg_trgm     {'ok  ' if has_trgm else 'MISSING'} required by the resolver")
        typer.echo(f"tables      {len(tables)}    {', '.join(tables) or 'none'}")

        if not has_trgm:
            failures.append("pg_trgm extension is missing")
        if "alembic_version" not in tables:
            failures.append("migrations have not been applied (run: alembic upgrade head)")
    except Exception as exc:
        typer.echo(f"postgres    FAIL {exc}")
        failures.append("postgres unreachable")

    try:
        Redis.from_url(settings.redis_url).ping()
        typer.echo("redis       ok")
    except Exception as exc:
        typer.echo(f"redis       FAIL {exc}")
        failures.append("redis unreachable")

    if failures:
        typer.echo("")
        for failure in failures:
            typer.echo(f"  - {failure}")
        raise typer.Exit(code=1)


@app.command("ingest")
def ingest(
    path: Path = typer.Argument(..., help="Letterboxd export zip or extracted directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse and report without writing"),
) -> None:
    """Parse a Letterboxd export and load its ratings, diary and likes."""
    started = time.perf_counter()
    parsed = parse_export(path)
    parse_seconds = time.perf_counter() - started
    stats = parsed.stats

    typer.echo(f"ratings            {stats.ratings}")
    typer.echo(f"diary entries      {stats.diary_entries}  ({stats.rewatches} rewatches)")
    typer.echo(f"likes              {stats.likes}")
    typer.echo(
        f"diary join         {stats.diary_films_matched}/{stats.diary_unique_films} films "
        f"matched to a rating by name+year"
    )
    if stats.diary_films_unmatched:
        typer.echo(f"  unmatched        {stats.diary_films_unmatched}")
    typer.echo(f"files read         {', '.join(stats.files_read) or 'none'}")
    if stats.files_missing:
        typer.echo(f"files missing      {', '.join(stats.files_missing)}")
    if stats.skipped:
        for reason, count in sorted(stats.skipped.items()):
            typer.echo(f"skipped            {count}  {reason}")
    typer.echo(f"parse time         {parse_seconds:.2f}s")

    if dry_run:
        typer.echo("\ndry run — nothing written")
        return

    if not stats.ratings:
        typer.echo("\nno ratings parsed; refusing to create an empty upload")
        raise typer.Exit(code=1)

    with session_scope() as session:
        upload = persist_export(session, parsed)
        token = upload.token

    typer.echo(f"\nupload token       {token}")


@app.command("tmdb-check")
def tmdb_check() -> None:
    """Verify the TMDB key works and report what a known film looks like."""
    try:
        client = TmdbClient()
    except TmdbAuthError as exc:
        typer.echo(f"tmdb        FAIL {exc}")
        raise typer.Exit(code=1) from exc

    try:
        movie = client.get_movie(238)  # The Godfather
        credits = client.get_credits(238)
    except TmdbError as exc:
        typer.echo(f"tmdb        FAIL {exc}")
        raise typer.Exit(code=1) from exc

    crew = credits.get("crew", [])
    departments = sorted({member.get("department", "?") for member in crew})

    typer.echo(f"tmdb        ok   rate limit {get_settings().tmdb_rate_limit:g} req/s")
    typer.echo(f"film        {movie.get('title')} ({(movie.get('release_date') or '?')[:4]})")
    typer.echo(f"votes       {movie.get('vote_average')} from {movie.get('vote_count')}")
    typer.echo(f"crew        {len(crew)} credits across {len(departments)} departments")
    typer.echo(f"departments {', '.join(departments)}")
    typer.echo(f"requests    {client.request_count} ({client.retry_count} retried)")


@app.command("tmdb-capture")
def tmdb_capture() -> None:
    """Record the seed TMDB responses into fixtures/ so tests can run offline."""
    client = TmdbClient()
    written = capture_all(client)

    for name in written:
        typer.echo(f"  {name}")
    typer.echo(f"\n{len(written)} fixtures written to {FIXTURE_DIR}")
    typer.echo(f"requests    {client.request_count} ({client.retry_count} retried)")


if __name__ == "__main__":
    app()
