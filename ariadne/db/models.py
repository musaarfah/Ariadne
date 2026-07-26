"""Schema for Ariadne.

Two groups of tables that are kept deliberately separate:

- The shared catalog (films, people, credits, resolutions) is identical for every user and
  grows monotonically. Title+year to TMDB id is the same answer for everyone, so caching
  resolutions globally means the 15th recruited account resolves almost entirely from cache.
- Per-upload data (uploads, ratings, diary_entries) and results. There is no users table:
  Ariadne has no accounts, so an upload token is the unit of identity.
"""

import enum
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ariadne.db.base import Base

# --- shared catalog ---------------------------------------------------------------------

# Only films live here. Television is never stored: TMDB movie ids and TV ids are separate
# namespaces that can collide on the same integer, and tmdb_id is this table's primary key.
# A Letterboxd entry identified as television is recorded on resolutions with method
# TELEVISION, which keeps the exclusion count queryable without risking a wrong join.


class Film(Base):
    __tablename__ = "films"

    tmdb_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    original_title: Mapped[str | None] = mapped_column(String(500))

    # Unicode-normalised, dashes folded, articles handled. Carries a trigram index so the
    # resolver's fuzzy fallback is a similarity() query rather than a full scan in Python.
    normalized_title: Mapped[str] = mapped_column(String(500))

    year: Mapped[int | None] = mapped_column(Integer)
    vote_average: Mapped[float | None] = mapped_column(Float)
    vote_count: Mapped[int | None] = mapped_column(Integer)
    country: Mapped[str | None] = mapped_column(String(2))

    raw: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_films_normalized_title", "normalized_title"),
        Index("ix_films_year", "year"),
        # The btree above serves exact-title lookups; this one makes similarity() cheap for
        # the resolver's fuzzy fallback. Declared here rather than only in the migration so
        # autogenerate does not try to drop it.
        Index(
            "ix_films_normalized_title_trgm",
            "normalized_title",
            postgresql_using="gin",
            postgresql_ops={"normalized_title": "gin_trgm_ops"},
        ),
    )


class Person(Base):
    __tablename__ = "people"

    tmdb_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(500))


class Credit(Base):
    __tablename__ = "credits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    film_id: Mapped[int] = mapped_column(ForeignKey("films.tmdb_id", ondelete="CASCADE"))
    person_id: Mapped[int] = mapped_column(ForeignKey("people.tmdb_id", ondelete="CASCADE"))

    # TMDB's own strings, stored unnormalised. All departments are fetched and kept even
    # though the model uses a handful of roles: one API call returns the whole credit list,
    # so filtering at fetch time would save nothing and would mean refetching everything
    # after any change to role scope.
    department: Mapped[str] = mapped_column(String(100))
    job: Mapped[str] = mapped_column(String(200))

    __table_args__ = (
        UniqueConstraint("film_id", "person_id", "job", name="uq_credit"),
        Index("ix_credits_person", "person_id"),
        Index("ix_credits_film", "film_id"),
    )


class MatchMethod(enum.StrEnum):
    EXACT = "exact"
    TRIGRAM = "trigram"
    # Several TMDB entries shared the title and year; one overwhelmed the rest by vote count.
    # Kept distinct from EXACT so these can be counted and audited separately.
    DOMINANT = "dominant"
    MANUAL = "manual"
    # Identified as television, so deliberately not resolved to a film. Distinct from
    # UNRESOLVED so the exclusion count is a query rather than a guess.
    TELEVISION = "television"
    UNRESOLVED = "unresolved"


class Resolution(Base):
    """Letterboxd film to TMDB id. User-independent, so cached across all uploads.

    Failures are rows with tmdb_id NULL and method UNRESOLVED rather than missing rows —
    the resolution-rate metric needs the denominator.
    """

    __tablename__ = "resolutions"

    letterboxd_uri: Mapped[str] = mapped_column(String(200), primary_key=True)
    name: Mapped[str] = mapped_column(String(500))
    year: Mapped[int | None] = mapped_column(Integer)
    tmdb_id: Mapped[int | None] = mapped_column(ForeignKey("films.tmdb_id", ondelete="SET NULL"))
    match_method: Mapped[MatchMethod] = mapped_column(Enum(MatchMethod, name="match_method"))
    confidence: Mapped[float | None] = mapped_column(Float)

    # Why the resolver decided what it did. Read by the 100-match hand audit in 1.4, where
    # "which rule fired" is the difference between a useful audit and staring at ids.
    reason: Mapped[str | None] = mapped_column(Text)
    resolved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# --- per-upload data -------------------------------------------------------------------


class UploadStatus(enum.StrEnum):
    PENDING = "pending"
    RESOLVING = "resolving"
    FETCHING_CREDITS = "fetching_credits"
    MODELLING = "modelling"
    COMPLETE = "complete"
    FAILED = "failed"


class Upload(Base):
    """One submitted export. Ariadne has no accounts, so this is the unit of identity.

    The token is the only handle on a result and is unguessable. Uploads are persisted
    rather than discarded so a run can be re-analysed with consent when the model improves
    — otherwise Phase 4 would mean re-recruiting every participant.
    """

    __tablename__ = "uploads"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[UploadStatus] = mapped_column(
        Enum(UploadStatus, name="upload_status"), default=UploadStatus.PENDING
    )
    film_count: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Rating(Base):
    __tablename__ = "ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    upload_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("uploads.id", ondelete="CASCADE"))

    # Not a foreign key to resolutions: ratings land at ingest, before anything is resolved.
    letterboxd_uri: Mapped[str] = mapped_column(String(200))

    # The title and year exactly as Letterboxd exported them. Kept here rather than seeding
    # resolutions, so that table holds only resolver output and "row exists" unambiguously
    # means "already attempted".
    name: Mapped[str] = mapped_column(String(500))
    year: Mapped[int | None] = mapped_column(Integer)

    rating: Mapped[Decimal] = mapped_column(Numeric(2, 1))

    # When the rating was logged, which is not when the film was watched. Only diary rows
    # carry a real watch date. See docs/DATA_FINDINGS.MD F7.
    logged_date: Mapped[date | None] = mapped_column(Date)

    __table_args__ = (Index("ix_ratings_upload", "upload_id"),)


class DiaryEntry(Base):
    """Watch dates and rewatch flags.

    Keyed by name+year rather than URI: diary.csv uses diary-entry links, which share no
    namespace with the film links in ratings.csv (zero overlap in the reference export).
    See docs/DATA_FINDINGS.MD F6.
    """

    __tablename__ = "diary_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    upload_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("uploads.id", ondelete="CASCADE"))
    film_key: Mapped[str] = mapped_column(String(500))
    watched_date: Mapped[date | None] = mapped_column(Date)
    is_rewatch: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (Index("ix_diary_upload", "upload_id"),)


class Like(Base):
    """Liked films — a positive-only signal, separate from ratings.

    Stored even though nothing reads it until the Phase 1.7 ablation: uploads are kept so a
    run can be re-analysed when the model improves, and a participant's export cannot be
    re-requested later without asking them again.
    """

    __tablename__ = "likes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    upload_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("uploads.id", ondelete="CASCADE"))
    letterboxd_uri: Mapped[str] = mapped_column(String(200))

    __table_args__ = (Index("ix_likes_upload", "upload_id"),)


# --- results ---------------------------------------------------------------------------


class AnalysisRun(Base):
    """One evaluation or fitting run. Never overwritten.

    metrics is JSONB so the whole baseline ladder, both splits, and the negative controls
    land in one versioned row. The writeup shows how numbers moved, not just where they
    ended up.
    """

    __tablename__ = "analysis_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    upload_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("uploads.id", ondelete="CASCADE"))
    model_version: Mapped[str] = mapped_column(String(50))
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_runs_upload", "upload_id"),)


class CrewEffect(Base):
    __tablename__ = "crew_effects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"))
    person_id: Mapped[int] = mapped_column(ForeignKey("people.tmdb_id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(100))

    effect: Mapped[float] = mapped_column(Float)
    stderr: Mapped[float | None] = mapped_column(Float)
    n_films: Mapped[int] = mapped_column(Integer)

    # Set when a person's contribution cannot be told apart from a collaborator's — for
    # example a cinematographer who has only ever shot for one director. Surfaced to the
    # user rather than resolved by guessing.
    inseparable_from_person_id: Mapped[int | None] = mapped_column(
        ForeignKey("people.tmdb_id", ondelete="SET NULL")
    )

    __table_args__ = (Index("ix_effects_run_role", "run_id", "role"),)


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"))
    film_id: Mapped[int] = mapped_column(ForeignKey("films.tmdb_id", ondelete="CASCADE"))
    score: Mapped[float] = mapped_column(Float)
    primary_reason_person_id: Mapped[int | None] = mapped_column(
        ForeignKey("people.tmdb_id", ondelete="SET NULL")
    )

    # TMDB vote_count percentile. Low is the goal: a recommender that surfaces The Godfather
    # to a Letterboxd power user is accurate and useless.
    novelty_pct: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (Index("ix_recs_run", "run_id"),)
