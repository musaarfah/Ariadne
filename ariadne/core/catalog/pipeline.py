"""Resolving an upload's films, cheapest source first.

Order matters for cost, not correctness: a cached resolution is free, a local catalog match
costs one indexed query, and only a miss reaches the TMDB API. Once several accounts have been
processed most films are already known, which is the whole reason resolutions are keyed by
Letterboxd URI rather than per upload.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ariadne.core.catalog.normalize import comparison_forms
from ariadne.core.catalog.resolve import (
    EXACT_TITLE_SIMILARITY,
    MAX_YEAR_GAP,
    ResolutionOutcome,
    resolve_from_results,
)
from ariadne.core.catalog.similarity import similarity
from ariadne.core.catalog.store import (
    cached_resolution,
    local_candidates,
    record_resolution,
    upsert_film,
)
from ariadne.core.catalog.tmdb import TmdbClient, TmdbError
from ariadne.db.models import MatchMethod, Rating, Resolution


@dataclass
class ResolveStats:
    total: int = 0
    from_cache: int = 0
    from_local: int = 0
    from_api: int = 0
    resolved: int = 0
    television: int = 0
    failed: int = 0
    errors: int = 0
    methods: Counter[str] = field(default_factory=Counter)

    @property
    def resolution_rate(self) -> float:
        return self.resolved / self.total if self.total else 0.0

    @property
    def attempted(self) -> int:
        """Films where a resolution was genuinely computed rather than read from cache."""
        return self.from_local + self.from_api


def _looks_like_television(results: list[dict[str, Any]], title: str, year: int | None) -> bool:
    """Whether a TV search result plausibly *is* the entry we failed to resolve as a film."""
    query_forms = comparison_forms(title)

    for result in results:
        names = [result.get("name"), result.get("original_name")]
        first_air = (result.get("first_air_date") or "")[:4]
        result_year = int(first_air) if first_air.isdigit() else None

        if year is not None and result_year is not None and abs(result_year - year) > MAX_YEAR_GAP:
            continue

        for name in names:
            if not isinstance(name, str):
                continue
            for candidate_form in comparison_forms(name):
                if any(
                    similarity(q, candidate_form) >= EXACT_TITLE_SIMILARITY for q in query_forms
                ):
                    return True
    return False


def resolve_one(
    session: Session,
    client: TmdbClient | None,
    letterboxd_uri: str,
    title: str,
    year: int | None,
    stats: ResolveStats,
    *,
    check_television: bool = True,
    retry_failed: bool = False,
) -> Resolution:
    stats.total += 1

    existing = cached_resolution(session, letterboxd_uri)
    # A cached failure is worth retrying after the resolver changes, but a cached success is
    # not — re-resolving what already worked only spends rate budget.
    if existing is not None and retry_failed and existing.tmdb_id is None:
        existing = None
    if existing is not None:
        stats.from_cache += 1
        _count_outcome(stats, existing.match_method)
        return existing

    outcome = resolve_from_results(local_candidates(session, title, year), title, year)

    # Counted as local either because the catalog answered, or because there is no client to
    # ask — an offline run should not report API calls it never made.
    if outcome.resolved or client is None:
        stats.from_local += 1
    else:
        stats.from_api += 1
        results = client.search_movies(title, year)
        outcome = resolve_from_results(results, title, year)

        if outcome.resolved:
            # Store the payload we matched so the catalog can answer this locally next time.
            matched = next((r for r in results if r.get("id") == outcome.tmdb_id), None)
            if matched is not None:
                upsert_film(session, matched)
        elif (
            check_television
            and outcome.no_film_candidate
            and _looks_like_television(client.search_tv(title, year), title, year)
        ):
            outcome = ResolutionOutcome(
                None,
                MatchMethod.TELEVISION,
                None,
                "matched a television series, excluded from the film catalog",
            )

    _count_outcome(stats, outcome.method)
    return record_resolution(session, letterboxd_uri, title, year, outcome)


def _count_outcome(stats: ResolveStats, method: MatchMethod) -> None:
    stats.methods[method.value] += 1
    if method is MatchMethod.TELEVISION:
        stats.television += 1
    elif method is MatchMethod.UNRESOLVED:
        stats.failed += 1
    else:
        stats.resolved += 1


def resolve_upload(
    session: Session,
    client: TmdbClient | None,
    upload_id: Any,
    limit: int | None = None,
    on_progress: Any = None,
    retry_failed: bool = False,
) -> ResolveStats:
    """Resolve every rated film on an upload, taking titles and years from its ratings."""
    stats = ResolveStats()

    films = list(
        session.execute(
            select(Rating.letterboxd_uri, Rating.name, Rating.year)
            .where(Rating.upload_id == upload_id)
            .order_by(Rating.id)
        ).all()
    )
    if limit is not None:
        films = films[:limit]

    for index, (uri, name, year) in enumerate(films, start=1):
        try:
            resolve_one(session, client, uri, name, year, stats, retry_failed=retry_failed)
        except TmdbError:
            stats.errors += 1
        if on_progress is not None:
            on_progress(index, len(films), stats)

    return stats
