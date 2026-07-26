import time
from pathlib import Path

import typer
from redis import Redis
from sqlalchemy import inspect, select, text

from ariadne.core.catalog.credits import CreditsStats, ingest_credits_for_upload
from ariadne.core.catalog.fixtures import FIXTURE_DIR, capture_all
from ariadne.core.catalog.pipeline import ResolveStats, resolve_upload
from ariadne.core.catalog.tmdb import TmdbAuthError, TmdbClient, TmdbError
from ariadne.core.evaluation.audit import read_verdicts, sample_cases, write_review_file
from ariadne.core.evaluation.coverage import build_coverage, pooled_regions, role_order
from ariadne.core.evaluation.dataset import load_dataset
from ariadne.core.evaluation.harness import MODEL_VERSION, evaluate, run_payload
from ariadne.core.evaluation.level1 import build_report, upload_by_token
from ariadne.core.evaluation.metrics import (
    GATE_K,
    GATE_THRESHOLD,
    PRODUCT_K,
    PRODUCT_THRESHOLD,
    precision_grid,
)
from ariadne.core.ingest.export import parse_export
from ariadne.core.ingest.persist import persist_export
from ariadne.db.models import AnalysisRun, Upload
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


@app.command("resolve")
def resolve(
    token: str = typer.Argument(..., help="Upload token from `ariadne ingest`"),
    limit: int | None = typer.Option(None, "--limit", help="Resolve only the first N films"),
    offline: bool = typer.Option(
        False, "--offline", help="Use only the local catalog; make no TMDB calls"
    ),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Re-attempt previously unresolved and television entries"
    ),
) -> None:
    """Resolve an upload's rated films to TMDB ids."""
    client = None if offline else TmdbClient()

    def progress(done: int, total: int, stats: ResolveStats) -> None:
        if done % 100 == 0 or done == total:
            typer.echo(
                f"  {done}/{total}  resolved {stats.resolved}  "
                f"failed {stats.failed}  tv {stats.television}  api {stats.from_api}"
            )

    started = time.perf_counter()
    with session_scope() as session:
        upload = session.execute(select(Upload).where(Upload.token == token)).scalar_one_or_none()
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)

        stats = resolve_upload(
            session,
            client,
            upload.id,
            limit=limit,
            on_progress=progress,
            retry_failed=retry_failed,
        )
    elapsed = time.perf_counter() - started

    typer.echo("")
    typer.echo(f"films               {stats.total}")
    typer.echo(f"resolved            {stats.resolved}  ({stats.resolution_rate * 100:.1f}%)")
    typer.echo(f"television excluded {stats.television}")
    typer.echo(f"unresolved          {stats.failed}")
    if stats.errors:
        typer.echo(f"errors              {stats.errors}")
    typer.echo("")
    typer.echo(f"from cache          {stats.from_cache}")
    typer.echo(f"from local catalog  {stats.from_local}")
    typer.echo(f"from tmdb api       {stats.from_api}")
    if client is not None:
        typer.echo(f"requests            {client.request_count} ({client.retry_count} retried)")
    typer.echo("")
    for method, count in sorted(stats.methods.items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {method:<12} {count}")
    typer.echo("")
    typer.echo(f"elapsed             {elapsed:.1f}s")


@app.command("metrics")
def metrics(token: str = typer.Argument(..., help="Upload token")) -> None:
    """Level 1 data-quality metrics for an upload."""
    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)
        report = build_report(session, upload.id)

    typer.echo("RESOLUTION")
    typer.echo(f"  entries                   {report.entries}")
    typer.echo(
        f"  resolved                  {report.resolved}"
        f"   {report.resolution_rate * 100:.1f}% of all entries"
    )
    typer.echo(f"  television                {report.television}")
    typer.echo(f"  unresolved                {report.unresolved}")
    typer.echo(
        f"  resolved excl. television {report.resolved}/{report.film_entries}"
        f"   {report.resolution_rate_excluding_television * 100:.1f}%   <- target >97%"
    )

    typer.echo("")
    typer.echo("METHOD")
    for method, count in report.methods.most_common():
        typer.echo(f"  {method:<12} {count:>5}")
    typer.echo(
        f"  fuzzy fallback {report.fuzzy_fallback}/{report.resolved}"
        f"   {report.fuzzy_fallback_rate * 100:.1f}% of resolved"
    )

    typer.echo("")
    typer.echo("CONFIDENCE")
    for value, count in sorted(report.confidences.items(), reverse=True):
        typer.echo(f"  {value:<12} {count:>5}")

    typer.echo("")
    typer.echo("CATALOG")
    typer.echo(f"  films stored              {report.films_in_catalog}")
    typer.echo(f"  with vote counts          {report.films_with_votes}")
    typer.echo(f"  with a country            {report.films_with_country}")

    typer.echo("")
    typer.echo("RESOLVED FILMS BY DECADE")
    for decade in sorted(report.by_decade):
        count = report.by_decade[decade]
        bar = "#" * round(count / max(report.by_decade.values()) * 40)
        typer.echo(f"  {decade}s {count:>5}  {bar}")

    typer.echo("")
    typer.echo("ID COLLISIONS (two entries resolving to one film — almost always an error)")
    if not report.collisions:
        typer.echo("  none")
    for tmdb_id, film, claimants in report.collisions:
        typer.echo(f"  {film}  id={tmdb_id}")
        for claimant in claimants:
            typer.echo(f"      {claimant}")

    typer.echo("")
    typer.echo("JOIN INTEGRITY (diary attaches on name+year)")
    typer.echo(f"  diary entries             {report.diary_entries}")
    typer.echo(
        f"  diary films matched       {report.diary_films_matched}/{report.diary_films}"
        f"   {report.diary_join_rate * 100:.1f}%"
    )


@app.command("audit-sample")
def audit_sample(
    token: str = typer.Argument(..., help="Upload token"),
    out: Path = typer.Option(
        Path("research/audits/resolution-audit.csv"), "--out", help="Review file to write"
    ),
    exact: int = typer.Option(50, "--exact", help="How many exact matches to sample"),
) -> None:
    """Write a stratified sample of resolutions for hand audit.

    Every non-exact outcome is included, since those are the risk pool. Exact matches are
    sampled with a fixed seed so the same sample is reproducible.
    """
    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)
        cases = sample_cases(session, upload.id, exact_sample=exact)

    write_review_file(cases, out)

    from collections import Counter

    by_method = Counter(case.method for case in cases)
    auto = Counter(case.auto_verdict for case in cases)

    typer.echo(f"{len(cases)} cases written to {out}")
    typer.echo("")
    for method, count in by_method.most_common():
        typer.echo(f"  {method:<12} {count:>4}")
    typer.echo("")
    typer.echo(f"  auto-correct  {auto['correct']:>4}  (folded title and year both identical)")
    typer.echo(f"  need review   {auto['review']:>4}")


@app.command("audit-report")
def audit_report(
    path: Path = typer.Argument(Path("research/audits/resolution-audit.csv"), help="Reviewed file"),
) -> None:
    """Score a reviewed audit file."""
    result = read_verdicts(path)

    typer.echo(f"cases  {result.total}")
    typer.echo("")
    typer.echo(
        f"{'method':<13}{'n':>4}{'correct':>9}{'wrong':>7}{'unsure':>8}{'todo':>6}{'acc':>8}"
    )
    for method in sorted(result.by_method):
        tally = result.by_method[method]
        accuracy = f"{tally.accuracy * 100:.1f}%" if tally.judged else "-"
        typer.echo(
            f"{method:<13}{tally.total:>4}{tally.correct:>9}{tally.wrong:>7}"
            f"{tally.unsure:>8}{tally.unfilled:>6}{accuracy:>8}"
        )

    resolved = result.resolved
    refusals = result.refusals

    typer.echo("")
    typer.echo("PRECISION — of the film matches made, how many are right")
    if resolved.judged:
        typer.echo(
            f"  {resolved.correct}/{resolved.judged}   {resolved.accuracy * 100:.2f}%"
            f"   <- target >99%"
        )
    else:
        typer.echo("  no verdicts recorded")
    if resolved.unsure:
        typer.echo(f"  {resolved.unsure} excluded as unsure")

    typer.echo("")
    typer.echo("REFUSALS — of the entries declined, how many were right to decline")
    if refusals.judged:
        typer.echo(f"  {refusals.correct}/{refusals.judged}   {refusals.accuracy * 100:.1f}%")
    else:
        typer.echo("  no verdicts recorded")

    if result.wrong_cases:
        typer.echo("")
        typer.echo("WRONG")
        for label, method, note in result.wrong_cases:
            typer.echo(f"  [{method}] {label}  —  {note}")

    if result.unsure_cases:
        typer.echo("")
        typer.echo("UNSURE")
        for label, method, note in result.unsure_cases:
            typer.echo(f"  [{method}] {label}  —  {note}")


@app.command("credits")
def credits_command(
    token: str = typer.Argument(..., help="Upload token"),
    limit: int | None = typer.Option(None, "--limit", help="Only the first N films"),
    refresh: bool = typer.Option(
        False, "--refresh", help="Refetch films that already have credits"
    ),
) -> None:
    """Fetch crew credits for every resolved film on an upload."""
    client = TmdbClient()

    def progress(done: int, total: int, stats: CreditsStats) -> None:
        if done % 100 == 0 or done == total:
            typer.echo(
                f"  {done}/{total}  fetched {stats.fetched}  cached {stats.skipped_cached}  "
                f"credits {stats.credits_stored}"
            )

    started = time.perf_counter()
    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)
        stats = ingest_credits_for_upload(
            session, client, upload.id, limit=limit, on_progress=progress, refresh=refresh
        )
    elapsed = time.perf_counter() - started

    typer.echo("")
    typer.echo(f"films               {stats.films}")
    typer.echo(f"fetched             {stats.fetched}")
    typer.echo(f"already had credits {stats.skipped_cached}")
    if stats.missing:
        typer.echo(f"missing on tmdb     {stats.missing}")
    if stats.errors:
        typer.echo(f"errors              {stats.errors}")
    typer.echo("")
    typer.echo(f"credits stored      {stats.credits_stored}")
    typer.echo(f"people seen         {stats.people_stored}")
    typer.echo(f"median crew / film  {stats.median_crew:.0f}")
    typer.echo(f"requests            {client.request_count} ({client.retry_count} retried)")
    typer.echo(f"elapsed             {elapsed:.1f}s")


@app.command("coverage")
def coverage(token: str = typer.Argument(..., help="Upload token")) -> None:
    """Crew coverage per role, by decade and region."""
    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)
        report = build_coverage(session, upload.id)

    roles = role_order()

    typer.echo("CATALOG")
    typer.echo(f"  films                 {report.films}")
    typer.echo(f"  with any credit       {report.films_with_any_credit}")
    typer.echo(f"  credits               {report.credits}")
    typer.echo(f"  distinct people       {report.people}")
    typer.echo(f"  median crew / film    {report.median_crew:.0f}")

    typer.echo("")
    typer.echo("ROLE COVERAGE  <- the number that decides how much of the product is possible")
    for role in roles:
        rate = report.role_rate(role)
        bar = "#" * round(rate * 40)
        multi = report.multi_credit_films[role]
        typer.echo(
            f"  {role:<20} {report.by_role[role]:>5}  {rate * 100:5.1f}%  {bar:<40}"
            f"  {multi} films with >1"
        )

    typer.echo("")
    typer.echo("BY DECADE")
    header = "  decade  films  " + "".join(f"{r[:6]:>8}" for r in roles)
    typer.echo(header)
    for decade in sorted(report.decade_films):
        films = report.decade_films[decade]
        cells = "".join(f"{report.by_decade[decade][r] / films * 100:7.0f}%" for r in roles)
        typer.echo(f"  {decade:<7} {films:>5}  {cells}")

    typer.echo("")
    typer.echo("BY REGION")
    named, pooled = pooled_regions(report)
    typer.echo("  region  films  " + "".join(f"{r[:6]:>8}" for r in roles))
    for region in named:
        films = report.region_films[region]
        cells = "".join(f"{report.by_region[region][r] / films * 100:7.0f}%" for r in roles)
        typer.echo(f"  {region:<7} {films:>5}  {cells}")
    if pooled:
        typer.echo(f"  (pooled: {pooled} films in regions with fewer than 15)")


@app.command("evaluate")
def evaluate_command(
    token: str = typer.Argument(..., help="Upload token"),
    k: int = typer.Option(20, "--k", help="k for Precision@k"),
    save: bool = typer.Option(True, "--save/--no-save", help="Persist the run to analysis_runs"),
    grid: bool = typer.Option(False, "--grid", help="Print the full P@k grid for each predictor"),
) -> None:
    """Score the baseline ladder on both splits."""
    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)

        films = load_dataset(session, upload.id)
        if not films:
            typer.echo("no resolved rated films; run `ariadne resolve` first")
            raise typer.Exit(code=1)

        results = evaluate(films, k=k)
        payload = run_payload(results, films=len(films))

        if save:
            session.add(
                AnalysisRun(
                    upload_id=upload.id,
                    model_version=MODEL_VERSION,
                    config={"k": k},
                    metrics=payload,
                )
            )

    typer.echo(f"films  {len(films)}   model  {MODEL_VERSION}")

    for split in results:
        typer.echo("")
        typer.echo(f"=== {split.split} ===")
        typer.echo(f"  {split.note}")
        typer.echo(f"  train {split.train_n}   test {split.test_n}", nl=False)
        typer.echo(f"   dropped (no log date) {split.dropped}" if split.dropped else "")

        drift = split.drift
        typer.echo(
            f"  drift: train mean {drift.train.mean:.3f} sd {drift.train.sd:.3f} "
            f"5.0s {drift.train.top_rating_share * 100:.1f}%"
        )
        typer.echo(
            f"         test  mean {drift.test.mean:.3f} sd {drift.test.sd:.3f} "
            f"5.0s {drift.test.top_rating_share * 100:.1f}%"
        )
        typer.echo(
            f"         shift {drift.mean_shift:+.3f} mean, "
            f"{drift.top_rating_share_shift * 100:+.1f}pp at 5.0"
        )

        typer.echo("")
        typer.echo(
            f"  {'predictor':<16}{'GATE':>7}{'base':>7}{'lift':>8}"
            f"{'P@' + str(k):>7}{'rho':>8}{'MAE':>7}{'MAEc':>7}"
        )
        for result in split.results:
            m = result.metrics
            typer.echo(
                f"  {result.name:<16}{m.gate_precision:>7.3f}{m.gate_base_rate:>7.3f}"
                f"{m.gate_lift:>+8.3f}{m.precision_at_k:>7.3f}{m.spearman:>8.3f}"
                f"{m.mae:>7.3f}{m.mae_centred:>7.3f}"
            )

    if grid:
        import numpy as np

        from ariadne.core.evaluation.baselines import ladder
        from ariadne.core.evaluation.splits import temporal_split

        grid_split = temporal_split(films)
        actual = np.array([f.rating for f in grid_split.test], dtype=float)
        typer.echo("")
        typer.echo("=== P@k grid, temporal split ===")
        for predictor in ladder():
            predictor.fit(grid_split.train)
            table = precision_grid(predictor.predict(grid_split.test), actual)
            typer.echo(f"  {predictor.name}")
            for threshold, row in table.items():
                cells = "  ".join(f"k={k}:{v:.3f}" for k, v in row.items())
                typer.echo(f"    >={threshold}  {cells}")

    typer.echo("")
    typer.echo(
        f"GATE = Precision@{GATE_K} at rating >= {GATE_THRESHOLD}. The go/no-go metric, chosen from"
    )
    typer.echo(
        f"the measured grid before the crew model existed. P@{PRODUCT_K} at >= {PRODUCT_THRESHOLD}"
        " is what a user experiences,"
    )
    typer.echo(
        "but it saturates: director_only already reaches 0.950 there, one film from the ceiling."
    )
    typer.echo("MAE stays secondary — 71.9% of ratings are whole stars, 222 films tie at 5.0.")
    if save:
        typer.echo("")
        typer.echo("run saved to analysis_runs")


if __name__ == "__main__":
    app()
