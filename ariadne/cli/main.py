import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from redis import Redis
from sqlalchemy import inspect, select, text

from ariadne.constants import TEMPORAL_SPLIT_DATE
from ariadne.core.catalog.credits import CreditsStats, ingest_credits_for_upload
from ariadne.core.catalog.fixtures import FIXTURE_DIR, capture_all
from ariadne.core.catalog.pipeline import ResolveStats, resolve_upload
from ariadne.core.catalog.tmdb import TmdbAuthError, TmdbClient, TmdbError
from ariadne.core.evaluation.audit import read_verdicts, sample_cases, write_review_file
from ariadne.core.evaluation.controls import build_null, shuffle_test, sweep
from ariadne.core.evaluation.coverage import build_coverage, pooled_regions, role_order
from ariadne.core.evaluation.dataset import load_dataset, person_names
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
from ariadne.core.recommend.decomposition import DECOMPOSITION_RESAMPLES, decompose
from ariadne.core.taste.crew import MIN_FILMS_TO_REPORT, CrewModel
from ariadne.db.models import AnalysisRun, CrewEffect, Upload
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


@app.command("backfill-cast")
def backfill_cast_command() -> None:
    """Store cast from payloads already in the database. No API calls.

    `get_movie(append="credits")` returned cast all along and the whole payload went into
    `films.raw`; only `store_credits` ignored it (D98).
    """
    from ariadne.core.catalog.credits import backfill_cast

    def progress(index: int, total: int, processed: int, written: int) -> None:
        if index % 2000 == 0 or index == total:
            typer.echo(f"  {index}/{total} films   with cast {processed}   credits {written}")

    started = time.perf_counter()
    with session_scope() as session:
        processed, written = backfill_cast(session, on_progress=progress)
    elapsed = time.perf_counter() - started

    typer.echo("")
    typer.echo(f"films with a stored payload  {processed}")
    typer.echo(f"cast credits written         {written}")
    typer.echo("api calls                    0")
    typer.echo(f"elapsed                      {elapsed:.1f}s")


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
    cut: datetime = typer.Option(
        TEMPORAL_SPLIT_DATE.isoformat(),
        formats=["%Y-%m-%d"],
        help="Temporal split date. Must match whatever `decompose` is given.",
    ),
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

        results = evaluate(films, k=k, cut=cut.date())
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

        from ariadne.core.evaluation.baselines import full_ladder
        from ariadne.core.evaluation.splits import temporal_split

        grid_split = temporal_split(films, cut.date())
        actual = np.array([f.rating for f in grid_split.test], dtype=float)
        typer.echo("")
        typer.echo("=== P@k grid, temporal split ===")
        for predictor in full_ladder():
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


@app.command("fit")
def fit(
    token: str = typer.Argument(..., help="Upload token"),
    role: str | None = typer.Option(None, "--role", help="Show one role only"),
    top: int = typer.Option(10, "--top", help="How many people to list per role"),
    save: bool = typer.Option(True, "--save/--no-save", help="Persist effects to crew_effects"),
) -> None:
    """Fit Track 1 crew effects and show them per role."""
    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)

        films = load_dataset(session, upload.id)
        model = CrewModel()
        model.fit(films)

        wanted = (role,) if role else model.roles
        # Names are fetched for what will actually be shown. Fetching by top-effect rank instead
        # missed the reportable people entirely, because a 1-film person can outrank them on raw
        # magnitude even after shrinkage.
        shown = [e for r in wanted for e in model.reportable_effects(r)[:top]]
        needed: set[int] = {e.person_id for e in shown}
        needed |= {e.inseparable_from for e in shown if e.inseparable_from is not None}
        names = person_names(session, needed)

        if save:
            run = AnalysisRun(
                upload_id=upload.id,
                model_version="1.7-track1",
                config={"roles": list(model.roles), "min_films": MIN_FILMS_TO_REPORT},
                metrics={},
            )
            session.add(run)
            session.flush()
            session.add_all(
                CrewEffect(
                    run_id=run.id,
                    person_id=e.person_id,
                    role=e.role,
                    effect=e.effect,
                    stderr=e.stderr,
                    n_films=e.n_films,
                    inseparable_from_person_id=e.inseparable_from,
                )
                for r in model.roles
                for e in model.effects(r)
            )

    typer.echo(f"films {len(films)}   reporting threshold: {MIN_FILMS_TO_REPORT}+ films")

    for r in wanted:
        effects = model.effects(r)
        reportable = [e for e in effects if e.reportable]
        typer.echo("")
        typer.echo(f"=== {r} ===")
        typer.echo(
            f"  {len(effects)} people fitted, {len(reportable)} with {MIN_FILMS_TO_REPORT}+ films"
        )
        if not reportable:
            typer.echo("  INSUFFICIENT DATA — nobody in this role clears the threshold")
            continue

        typer.echo(f"  {'effect':>8}{'raw':>8}{'n':>5}{'se':>7}  person")
        for e in reportable[:top]:
            label = names.get(e.person_id, str(e.person_id))
            typer.echo(
                f"  {e.effect:>+8.3f}{e.raw_mean:>+8.3f}{e.n_films:>5}{e.stderr:>7.3f}  {label}"
            )
            if e.inseparable_from == e.person_id:
                # A writer-director: their writing cannot be told apart from their own work
                # behind the camera, which is a real limit rather than a missing feature.
                typer.echo(
                    f"           cannot be separated from their own direction "
                    f"({e.n_films}/{e.n_films} films)"
                )
            elif e.inseparable_from is not None:
                other = names.get(e.inseparable_from, str(e.inseparable_from))
                typer.echo(
                    f"           cannot be separated from {other} ({e.n_films}/{e.n_films} films)"
                )

        withheld = len(effects) - len(reportable)
        if withheld:
            typer.echo(f"  ({withheld} people withheld: fewer than {MIN_FILMS_TO_REPORT} films)")

    if save:
        typer.echo("")
        typer.echo("effects saved to crew_effects")


@app.command("filmographies")
def filmographies(
    token: str = typer.Argument(..., help="Upload token"),
    min_films: int = typer.Option(3, "--min-films", help="Films in the library to qualify"),
    limit: int | None = typer.Option(None, "--limit", help="Only the first N people"),
) -> None:
    """Fetch what else the library's crew have worked on, so recommendations have candidates."""
    from ariadne.core.catalog.filmography import (
        FilmographyStats,
        fetch_genre_map,
        ingest_filmographies,
        people_worth_traversing,
    )

    client = TmdbClient()

    def progress(done: int, total: int, stats: FilmographyStats) -> None:
        if done % 100 == 0 or done == total:
            typer.echo(
                f"  {done}/{total} people   films stored {stats.films_stored}   "
                f"credits {stats.credits_stored}"
            )

    started = time.perf_counter()
    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)

        people = people_worth_traversing(session, upload.id, min_films=min_films)
        if limit is not None:
            people = people[:limit]
        typer.echo(f"{len(people)} people with {min_films}+ films in a modelled role")

        genre_map = fetch_genre_map(client)
        stats = ingest_filmographies(session, client, people, genre_map, on_progress=progress)
    elapsed = time.perf_counter() - started

    typer.echo("")
    typer.echo(f"people fetched      {stats.fetched}/{stats.people}")
    if stats.missing or stats.errors:
        typer.echo(f"missing / errors    {stats.missing} / {stats.errors}")
    typer.echo(f"new films stored    {stats.films_stored}")
    typer.echo(f"credits stored      {stats.credits_stored}")
    typer.echo(f"skipped, few votes  {stats.skipped_low_votes}")
    typer.echo(f"requests            {client.request_count} ({client.retry_count} retried)")
    typer.echo(f"elapsed             {elapsed:.1f}s")


@app.command("decompose")
def decompose_command(
    token: str = typer.Argument(..., help="Upload token"),
    resamples: int = typer.Option(DECOMPOSITION_RESAMPLES, help="Paired bootstrap resamples"),
    cut: datetime = typer.Option(
        TEMPORAL_SPLIT_DATE.isoformat(),
        formats=["%Y-%m-%d"],
        help="Temporal split date. The default is the reference account's backfill boundary and is "
        "wrong for any other library — choose it from the log-date distribution first.",
    ),
) -> None:
    """What explains your taste. Tier 2 — every number carries an interval.

    Contributions are marginal: each layer is added alone to the same consensus base, so no number
    depends on the order the layers are listed in. They therefore do not sum to the total, and the
    overlap is reported rather than hidden.
    """
    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)
        films = load_dataset(session, upload.id)

    for report in decompose(films, resamples=resamples, cut=cut.date()):
        typer.echo("")
        typer.echo(f"=== {report.split}   test {report.test_n} films")
        typer.echo(f"  {report.note}")
        typer.echo("")
        typer.echo(
            f"  What everyone else thought of the film explains "
            f"{report.base_explained:.1%} of the variation in your ratings on its own"
            f"  (gate {report.base_gate:.3f})."
        )
        typer.echo("  Everything below is what each kind of information adds to that:")
        typer.echo("")
        typer.echo(f"  {'':<26}{'explained':>10} {'95% CI':>17}{'gate':>9} {'95% CI':>17}")

        for layer in report.layers:
            explained, ranking = layer.explained, layer.ranking
            verdict = "clears zero" if layer.helps else ("makes it worse" if layer.hurts else "")
            typer.echo(
                f"  {layer.label:<26}{explained.observed_diff:>+10.3f} "
                f"[{explained.ci_low:>+6.3f},{explained.ci_high:>+6.3f}]"
                f"{ranking.observed_diff:>+9.3f} "
                f"[{ranking.ci_low:>+6.3f},{ranking.ci_high:>+6.3f}]  {verdict}"
            )

        combined = report.combined
        if combined is not None:
            typer.echo("")
            typer.echo(
                f"  {'all of them together':<26}{combined.explained.observed_diff:>+10.3f} "
                f"[{combined.explained.ci_low:>+6.3f},{combined.explained.ci_high:>+6.3f}]"
                f"{combined.ranking.observed_diff:>+9.3f} "
                f"[{combined.ranking.ci_low:>+6.3f},{combined.ranking.ci_high:>+6.3f}]"
            )
            typer.echo(
                f"  the layers measured one at a time sum to {report.sum_of_layers:+.3f}, so "
                f"{report.overlap:+.3f} of that is the same information counted twice"
            )

    typer.echo("")
    typer.echo("Explained = share of the variation in held-out ratings, over every film.")
    typer.echo("Gate = Precision@100 at >= 4.5, over the top 100 only. A layer can move one and")
    typer.echo(
        "not the other: explaining a rating and improving a recommendation are different jobs."
    )


def _print_portrait(report: Any, names: dict[int, str]) -> None:
    from ariadne.core.recommend.portrait import MIN_VOTES_FOR_RESIDUAL

    typer.echo(f"{report.films} rated films.  Everything below is a count or a direct observation:")
    typer.echo("no estimate, no interval, nothing that could turn out to be a different number.")

    revealed = report.revealed
    if revealed is not None:
        typer.echo("")
        typer.echo("WHAT YOU SAY vs WHAT YOU DO")
        typer.echo(
            f"  You rated {revealed.top_rated_total} films {5.0:g}. Of the "
            f"{revealed.top_rated_in_diary} your diary covers, "
            f"**{revealed.top_rated_never_revisited} you never went back to.**"
        )
        typer.echo("  Films you actually returned to:")
        for film in revealed.rewatched[:6]:
            typer.echo(f"    {film.rewatches}x  {film.title[:44]:<46} rated {film.rating}")

    typer.echo("")
    typer.echo("PEOPLE YOU HAVE BEEN FOLLOWING")
    for loyalty in report.loyalties:
        typer.echo(
            f"  {loyalty.films:>3} films   {names.get(loyalty.person_id, loyalty.person_id):<26}"
            f"{loyalty.role}"
        )

    typer.echo("")
    typer.echo("WHERE YOU DISAGREE WITH THE WORLD")
    typer.echo(
        f"  (films with at least {MIN_VOTES_FOR_RESIDUAL} votes; "
        f"{report.excluded_low_votes} excluded for having too few)"
    )
    typer.echo("  you rate far ABOVE what this kind of film usually gets from you:")
    for item in report.above[:5]:
        typer.echo(
            f"    {item.residual:>+5.2f}  {item.film.title[:40]:<42} "
            f"you {item.film.rating:<4} world {item.film.vote_average:.1f}/10"
        )
    typer.echo("  and far BELOW:")
    for item in report.below[:5]:
        typer.echo(
            f"    {item.residual:>+5.2f}  {item.film.title[:40]:<42} "
            f"you {item.film.rating:<4} world {item.film.vote_average:.1f}/10"
        )

    typer.echo("")
    typer.echo("YOUR BLIND SPOTS — rated above your average, barely watched")
    for spot in report.blind_spots[:6]:
        typer.echo(
            f"  {spot.label:<18} {spot.films:>3} films   avg {spot.mean_rating:.2f}"
            f"   {spot.lift:+.2f} vs your library average of {spot.library_mean:.2f}"
        )

    style = report.style
    if style is not None:
        typer.echo("")
        typer.echo("HOW YOU RATE")
        typer.echo(
            f"  {style.whole_star_share * 100:.0f}% whole stars, "
            f"{style.levels_used} of {style.levels_available} levels used  —  "
            + ("decisive rather than calibrating" if style.is_decisive else "you use the scale")
        )
        typer.echo(
            f"  mean {style.mean:.2f}, spread {style.sd:.2f}, "
            f"{style.top_rating_share * 100:.0f}% of everything you rate is a {5.0:g}"
        )


@app.command("portrait")
def portrait(
    token: str = typer.Argument(..., help="Upload token"),
) -> None:
    """What your library knows about you. Tier 1 only — facts, no estimates.

    Nothing here is a prediction. Four hypotheses about the predictive value of who-made-what were
    tested and rejected (F70, F72, F73), so this section deliberately contains no modelled claim.
    """
    from ariadne.core.catalog.roles import PRODUCT_ROLES
    from ariadne.core.recommend.portrait import build_portrait
    from ariadne.core.taste.expectation import fit_rich_expectation

    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)

        films = load_dataset(session, upload.id)
        expectation = fit_rich_expectation(films)
        report = build_portrait(films, expectation, PRODUCT_ROLES)
        names = person_names(session, {loyalty.person_id for loyalty in report.loyalties})

    _print_portrait(report, names)


@app.command("analyze")
def analyze(
    path: Path = typer.Argument(..., help="Letterboxd export zip or extracted directory"),
    offline: bool = typer.Option(
        False, "--offline", help="Skip TMDB entirely — resolve against the local catalog only"
    ),
) -> None:
    """Ingest a Letterboxd export and print everything tier 1 can say about it, in one command.

    Runs ingest, resolve, credits, then the portrait — the same four steps `ingest` / `resolve` /
    `credits` / `portrait` run separately, chained for a library nobody has tuned anything for.

    Deliberately stops at tier 1. `evaluate` and `decompose` need a temporal cut chosen from this
    account's own log-date history, before any model has run (D112) — a cut this command guessed
    would be a cut chosen to flatter whatever it guessed. The month-by-month log-date counts are
    printed at the end so that choice can be made by looking at the account, not by looking away.
    """
    typer.echo(f"=== ingest: {path} ===")
    started = time.perf_counter()
    parsed = parse_export(path)
    stats = parsed.stats
    typer.echo(
        f"ratings {stats.ratings}   diary {stats.diary_entries} ({stats.rewatches} rewatches)"
        f"   likes {stats.likes}"
    )
    if stats.diary_films_unmatched:
        typer.echo(f"diary join unmatched  {stats.diary_films_unmatched}")
    if not stats.ratings:
        typer.echo("\nno ratings parsed; refusing to create an empty upload")
        raise typer.Exit(code=1)

    with session_scope() as session:
        upload = persist_export(session, parsed)
        token = upload.token
    typer.echo(f"upload token   {token}")

    client = None if offline else TmdbClient()

    def resolve_progress(done: int, total: int, resolve_stats: ResolveStats) -> None:
        if done == total:
            rate = resolve_stats.resolution_rate * 100
            typer.echo(
                f"resolved {resolve_stats.resolved}/{total} ({rate:.1f}%)"
                f"   television {resolve_stats.television}   unresolved {resolve_stats.failed}"
            )

    typer.echo("\n=== resolve ===")
    with session_scope() as session:
        resolve_upload_row = upload_by_token(session, token)
        assert resolve_upload_row is not None  # noqa: S101 — just created it above
        resolve_upload(session, client, resolve_upload_row.id, on_progress=resolve_progress)

    if offline:
        typer.echo("\n=== credits: skipped (--offline) ===")
        typer.echo("loyalties and disagreements below will be sparse without TMDB credit data.")
    else:
        assert client is not None  # noqa: S101 — only None when offline

        def credits_progress(done: int, total: int, credit_stats: CreditsStats) -> None:
            if done == total:
                typer.echo(
                    f"fetched {credit_stats.fetched}   already held {credit_stats.skipped_cached}"
                    f"   credits stored {credit_stats.credits_stored}"
                )

        typer.echo("\n=== credits ===")
        with session_scope() as session:
            credits_upload = upload_by_token(session, token)
            assert credits_upload is not None  # noqa: S101
            ingest_credits_for_upload(
                session, client, credits_upload.id, on_progress=credits_progress
            )

    from ariadne.core.catalog.roles import PRODUCT_ROLES
    from ariadne.core.recommend.portrait import build_portrait
    from ariadne.core.taste.expectation import fit_rich_expectation

    typer.echo("\n=== portrait: what your library knows about you ===")
    typer.echo("tier 1 only — counts and direct observations, no modelled claim (F70, F72, F73)\n")
    with session_scope() as session:
        portrait_upload = upload_by_token(session, token)
        assert portrait_upload is not None  # noqa: S101
        films = load_dataset(session, portrait_upload.id)
        expectation = fit_rich_expectation(films)
        report = build_portrait(films, expectation, PRODUCT_ROLES)
        names = person_names(session, {loyalty.person_id for loyalty in report.loyalties})
    _print_portrait(report, names)

    typer.echo(f"\nelapsed   {time.perf_counter() - started:.1f}s")
    typer.echo(f"upload token   {token}")

    months = Counter(
        f"{film.logged_date.year}-{film.logged_date.month:02d}"
        for film in films
        if film.logged_date
    )
    if months:
        typer.echo("")
        typer.echo("ratings logged per month, most recent first (pick a cut here, not a score):")
        for month in sorted(months, reverse=True)[:8]:
            typer.echo(f"  {month}  {months[month]:>4}  {'#' * (months[month] // 10)}")
    typer.echo("")
    typer.echo(
        "To go further — director/crew effects, and how much of a rating each explains — choose "
        "a cut date from the histogram above (the month after the last visible backfill burst) "
        "and run:"
    )
    typer.echo(f"  ariadne evaluate {token} --cut YYYY-MM-DD")
    typer.echo(f"  ariadne decompose {token} --cut YYYY-MM-DD")


@app.command("recommend")
def recommend(
    token: str = typer.Argument(..., help="Upload token"),
    top: int = typer.Option(20, "--top", help="How many to show"),
) -> None:
    """Recommend unseen films by walking the crew graph, with Level 3 metrics."""
    from ariadne.core.catalog.roles import DIRECTOR
    from ariadne.core.evaluation.baselines import DirectorOnly
    from ariadne.core.recommend.adjacency import (
        build_recommendations,
        find_disagreements,
        resolve_directors,
    )

    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)

        films = load_dataset(session, upload.id)
        model = CrewModel()
        model.fit(films)

        report = build_recommendations(session, upload.id, model, films)
        chosen = report.top(top)

        # Candidates arrive from below-the-line filmographies, so nothing is known about who
        # directed them. Resolving the shortlist is what makes non-obviousness measurable.
        familiar = {p for f in films for p in f.people_in(DIRECTOR)}
        resolved = resolve_directors(chosen, TmdbClient(), familiar)

        director_model = DirectorOnly()
        director_model.fit(films)
        gaps = find_disagreements(films, model, director_model)[:5]

        wanted = {c.reason_person for c in chosen if c.reason_person}
        names = person_names(session, wanted)

    typer.echo(f"catalog {report.catalog_films} films   already rated {report.already_rated}")
    typer.echo(f"scoreable candidates {report.scoreable}   coverage {report.coverage * 100:.1f}%")

    typer.echo("")
    typer.echo("LEVEL 3 METRICS")
    typer.echo(
        f"  novelty            {report.novelty(top) * 100:.1f}th percentile of your library"
        "   (low is the target)"
    )
    typer.echo(
        "  ranked by crew contribution, not predicted rating — see Candidate.crew_adjustment"
    )
    obvious = report.non_obviousness(top)
    if obvious is None:
        typer.echo("  non-obviousness    unknown — no director resolved for any recommendation")
    else:
        typer.echo(
            f"  non-obviousness    {obvious * 100:.0f}% have a director you have never rated"
            f"   ({resolved}/{len(chosen)} directors looked up)"
        )

    typer.echo("")
    typer.echo(f"TOP {top}")
    for candidate in chosen:
        who = names.get(candidate.reason_person or 0, str(candidate.reason_person))
        flag = "  [familiar director]" if candidate.is_non_obvious is False else ""
        typer.echo(
            f"  {candidate.crew_adjustment:>+6.3f}  {candidate.film.title[:42]:<43}"
            f"{str(candidate.film.year or '?'):>6}  pred {candidate.score:.2f}{flag}"
        )
        typer.echo(
            f"         because of {who} ({candidate.reason_role}, {candidate.reason_effect:+.3f})"
            f"   novelty {candidate.novelty_percentile * 100:.0f}th"
        )

    typer.echo("")
    typer.echo("WHERE CREW AND DIRECTOR DISAGREE MOST (films you have already rated)")
    for gap in gaps:
        typer.echo(
            f"  {gap.film.title[:44]:<45} you {gap.film.rating:>4.1f}   "
            f"crew {gap.crew_score:>5.2f}   director {gap.director_score:>5.2f}"
            f"   gap {gap.gap:+.2f}"
        )


@app.command("variants")
def variants(
    token: str = typer.Argument(..., help="Upload token"),
    resamples: int = typer.Option(1500, "--resamples", help="Paired bootstrap resamples"),
) -> None:
    """Compare crew-model variants against the default, so choices are measured not argued.

    Varies the expectation model, how co-credited people combine, and whether the target includes
    a rewatch bonus. Every variant is scored on both splits and bootstrapped against the default.
    """
    import numpy as np

    from ariadne.core.evaluation.metrics import GATE_K, GATE_THRESHOLD, compare, precision_at_k
    from ariadne.core.evaluation.splits import random_split, temporal_split

    definitions: list[tuple[str, str, str, str]] = [
        ("default", "rich", "mean", "rating"),
        ("simple expectation", "simple", "mean", "rating"),
        ("combine=max", "rich", "max", "rating"),
        ("combine=weighted", "rich", "weighted", "rating"),
        ("target=preference", "rich", "mean", "preference"),
        ("preference + weighted", "rich", "weighted", "preference"),
    ]

    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)
        films = load_dataset(session, upload.id)

    covered = sum(1 for f in films if f.in_diary)
    typer.echo(
        f"films {len(films)}   diary covers {covered} ({covered / len(films) * 100:.0f}%), "
        f"so the rewatch bonus applies to those only"
    )

    for split in (temporal_split(films), random_split(films)):
        # Scored against the raw rating throughout, even for the preference variant. A model that
        # optimises a target of its own choosing must still be judged on the thing users care
        # about, or the comparison is rigged.
        actual = np.array([f.rating for f in split.test], dtype=float)

        typer.echo("")
        typer.echo(f"=== {split.name} ===   train {len(split.train)}   test {len(split.test)}")
        typer.echo("")
        typer.echo(f"  {'variant':<24}{'GATE':>7}{'vs default':>12}{'95% CI':>18}  verdict")

        predictions: dict[str, np.ndarray] = {}
        for label, expectation, combine, target in definitions:
            model = CrewModel(expectation=expectation, combine=combine, target=target)
            model.fit(split.train)
            predictions[label] = model.predict(split.test)

        for label, _, _, _ in definitions:
            gate = precision_at_k(predictions[label], actual, GATE_K, GATE_THRESHOLD)
            if label == "default":
                typer.echo(f"  {label:<24}{gate:>7.3f}{'—':>12}{'—':>18}  reference")
                continue
            result = compare(
                label,
                predictions[label],
                "default",
                predictions["default"],
                actual,
                resamples=resamples,
            )
            verdict = "differs" if result.significant else "within noise"
            typer.echo(
                f"  {label:<24}{gate:>7.3f}{result.observed_diff:>+12.3f}"
                f"{f'[{result.ci_low:+.3f}, {result.ci_high:+.3f}]':>18}  {verdict}"
            )

    typer.echo("")
    typer.echo("All variants scored against the raw rating, including the preference one.")


@app.command("attribution")
def attribution(
    token: str = typer.Argument(..., help="Upload token"),
) -> None:
    """Does each person's effect survive once directors compete for the same variance?

    Track 1 estimates people independently and can only flag inseparability structurally. Ridge
    fits everyone at once, so refitting with directors as features shows quantitatively whether an
    effect was really the director's (D8, D9).
    """
    from ariadne.core.taste.ridge import attribute

    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)
        films = load_dataset(session, upload.id)

        model = CrewModel()
        model.fit(films)
        track1 = {
            (r, e.person_id): (e.effect, e.n_films)
            for r in model.roles
            for e in model.reportable_effects(r)
        }
        results = attribute(films, track1)
        names = person_names(session, {a.person_id for a in results})

    typer.echo(
        f"films {len(films)}   {len(results)} people above the {MIN_FILMS_TO_REPORT}-film threshold"
    )
    typer.echo("")
    typer.echo(
        f"  {'person':<24}{'role':<18}{'n':>4}{'track1':>9}"
        f"{'ridge':>9}{'+dir':>9}{'kept':>7}  verdict"
    )
    for a in results:
        label = names.get(a.person_id, str(a.person_id))[:23]
        kept = f"{a.retained_share:>7.2f}" if a.retained_share is not None else f"{'—':>7}"
        typer.echo(
            f"  {label:<24}{a.role:<18}{a.n_films:>4}{a.track1_effect:>+9.3f}"
            f"{a.ridge_without_director:>+9.3f}{a.ridge_with_director:>+9.3f}"
            f"{kept}  {a.verdict}"
        )

    attributable = [a for a in results if a.attributable]
    survived = sum(1 for a in attributable if a.survives)
    typer.echo("")
    typer.echo(
        f"  {survived}/{len(attributable)} attributable effects survive with directors in the model"
    )
    typer.echo(f"  ({len(results) - len(attributable)} too small to attribute)")
    typer.echo("  'kept' is the ridge-with-director coefficient as a share of the Track 1 effect.")


@app.command("gate")
def gate(
    token: str = typer.Argument(..., help="Upload token"),
    resamples: int = typer.Option(2000, "--resamples", help="Paired bootstrap resamples"),
) -> None:
    """The go/no-go, with intervals.

    Reported against both comparisons (D69): crew versus director-only is the thesis, crew versus
    the best baseline is whether it is useful.
    """
    import numpy as np

    from ariadne.core.evaluation.baselines import full_ladder
    from ariadne.core.evaluation.metrics import compare
    from ariadne.core.evaluation.splits import random_split, temporal_split

    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)
        films = load_dataset(session, upload.id)

    for split in (temporal_split(films), random_split(films)):
        actual = np.array([f.rating for f in split.test], dtype=float)
        predictions: dict[str, np.ndarray] = {}
        for predictor in full_ladder():
            predictor.fit(split.train)
            predictions[predictor.name] = predictor.predict(split.test)

        typer.echo("")
        typer.echo(f"=== {split.name} ===   train {len(split.train)}   test {len(split.test)}")
        typer.echo("")
        typer.echo(f"  {'comparison':<34}{'diff':>8}{'95% CI':>18}{'P(a>b)':>9}  verdict")

        pairs = [
            ("crew", "director_only", "THESIS: crew beats the director"),
            ("crew", "context", "USEFULNESS: crew beats the context baseline"),
            ("crew", "genre_only", "crew vs genre_only"),
            ("crew_ridge", "director_only", "Track 2 vs director"),
            ("crew", "crew_ridge", "Track 1 vs Track 2"),
            ("genre_only", "director_only", "genre vs director"),
        ]
        for a, b, label in pairs:
            if a not in predictions or b not in predictions:
                continue
            result = compare(a, predictions[a], b, predictions[b], actual, resamples=resamples)
            verdict = "significant" if result.significant else "within noise"
            typer.echo(
                f"  {label:<34}{result.observed_diff:>+8.3f}"
                f"{f'[{result.ci_low:+.3f}, {result.ci_high:+.3f}]':>18}"
                f"{result.prob_a_better:>9.2f}  {verdict}"
            )

    typer.echo("")
    typer.echo("Paired bootstrap on Precision@100 at >=4.5. Both predictors are scored on the same")
    typer.echo("resampled test set each time, so the interval is on the difference itself.")


@app.command("significance")
def significance(
    token: str = typer.Argument(..., help="Upload token"),
    permutations: int = typer.Option(200, "--permutations", help="Permutations for the null"),
) -> None:
    """Test observed crew effects against a permutation null.

    The null is over the largest effect per permutation, which corrects for testing hundreds of
    people at once: clearing it means beating the best that noise managed anywhere.
    """
    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)
        films = load_dataset(session, upload.id)

        model = CrewModel()
        model.fit(films)

        shown = [e for r in model.roles for e in model.reportable_effects(r)[:3]]
        names = person_names(session, {e.person_id for e in shown})

    typer.echo(
        f"films {len(films)}   permutations {permutations}   threshold {MIN_FILMS_TO_REPORT}+"
    )
    typer.echo("")
    typer.echo(f"  {'role':<20}{'null p50':>10}{'null p95':>10}{'observed':>10}   verdict")

    any_significant = False
    detail: list[str] = []
    for r in model.roles:
        null = build_null(films, r, permutations=permutations)
        reportable = model.reportable_effects(r)
        if not reportable:
            typer.echo(
                f"  {r:<20}{null.percentile(50):>10.3f}{null.critical_value:>10.3f}"
                f"{'—':>10}   no one above the film threshold"
            )
            continue

        best = max(reportable, key=lambda e: abs(e.effect))
        significant = abs(best.effect) > null.critical_value
        any_significant = any_significant or significant
        verdict = "SIGNIFICANT" if significant else "not distinguishable from noise"
        typer.echo(
            f"  {r:<20}{null.percentile(50):>10.3f}{null.critical_value:>10.3f}"
            f"{abs(best.effect):>10.3f}   {verdict}"
        )
        detail.append(
            f"  {r}: {names.get(best.person_id, str(best.person_id))} "
            f"{best.effect:+.3f} on {best.n_films} films, p = {null.p_value(best.effect):.3f}"
        )

    typer.echo("")
    typer.echo("STRONGEST EFFECT PER ROLE")
    for line in detail:
        typer.echo(line)

    typer.echo("")
    if any_significant:
        typer.echo("At least one role has an effect beyond what noise produced.")
    else:
        typer.echo("No role has an effect beyond the permutation null. On this data, at this")
        typer.echo("sample size, no below-the-line crew effect can be distinguished from noise.")


@app.command("controls")
def controls(
    token: str = typer.Argument(..., help="Upload token"),
    role: str = typer.Option("editor", "--role", help="Role for the synthetic sweep"),
    trials: int = typer.Option(12, "--trials", help="Trials per sweep cell"),
    skip_sweep: bool = typer.Option(False, "--skip-sweep", help="Shuffle test only"),
) -> None:
    """Run the negative controls: shuffle test, then the synthetic detection-floor sweep."""
    with session_scope() as session:
        upload = upload_by_token(session, token)
        if upload is None:
            typer.echo(f"no upload with token {token}")
            raise typer.Exit(code=1)
        films = load_dataset(session, upload.id)

    typer.echo("=== SHUFFLE TEST ===")
    typer.echo("  ratings permuted between films; every effect must collapse toward zero")
    result = shuffle_test(films)
    typer.echo("")
    typer.echo(f"  {'':<24}{'real':>10}{'shuffled':>10}")
    typer.echo(
        f"  {'largest effect':<24}{result.real_max_effect:>10.3f}"
        f"{result.shuffled_max_effect:>10.3f}"
    )
    typer.echo(
        f"  {'reportable people':<24}{result.real_reportable:>10}{result.shuffled_reportable:>10}"
    )
    typer.echo(
        f"  {f'effects >= {result.threshold}':<24}{result.real_above_threshold:>10}"
        f"{result.shuffled_above_threshold:>10}"
    )
    typer.echo("")
    typer.echo(f"  above-threshold ratio {result.count_ratio:.3f}  (pass condition: <= 0.1)")
    typer.echo(f"  largest-effect ratio  {result.max_ratio:.3f}  (informational only)")
    typer.echo("      a null maximum rises with how many people are tested, so this ratio is")
    typer.echo("      not comparable across role scopes. See build_null for the rigorous test.")
    verdict = "PASS — signal collapsed" if result.collapsed else "FAIL — SHRINKAGE IS BROKEN"
    typer.echo(f"  VERDICT: {verdict}")

    if skip_sweep:
        return

    typer.echo("")
    typer.echo(f"=== SYNTHETIC SWEEP ({role}) ===")
    typer.echo("  a known effect is planted in one person, then recovered or not")
    swept = sweep(films, role, trials=trials)

    sizes = sorted({c.effect_size for c in swept.cells})
    counts = sorted({c.n_films for c in swept.cells})
    typer.echo("")
    typer.echo("  recovery rate (planted person in top 3 of the role)")
    typer.echo("  effect  " + "".join(f"{f'n={n}':>9}" for n in counts))
    for size in sizes:
        cells = {c.n_films: c for c in swept.cells if c.effect_size == size}
        row = "".join(
            f"{cells[n].rate:>8.2f}{'*' if cells[n].detected else ' '}"
            if n in cells
            else f"{'-':>9}"
            for n in counts
        )
        typer.echo(f"  {size:>+6.2f}  {row}")
    typer.echo("")
    typer.echo("  * detected: recovered in at least 80% of trials")
    typer.echo("")
    typer.echo("  DETECTION FLOOR — fewest films needed per effect size")
    for size in sizes:
        floor = swept.floor_for(size)
        typer.echo(
            f"    {size:>+5.2f} stars: {floor if floor else 'not detected at any tested count'}"
        )


if __name__ == "__main__":
    app()
