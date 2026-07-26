"""Parsing a Letterboxd data export.

This module is the privacy boundary. An export contains profile.csv, which holds the user's
email address, real name and location. ExportSource refuses to open any file outside
EXPORT_FILES_USED, so nothing downstream can receive those fields even by accident.
"""

import csv
import io
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ariadne.constants import EXPORT_FILES_USED, MAX_RATING, MIN_RATING

RATINGS_FILE = "ratings.csv"
DIARY_FILE = "diary.csv"
LIKES_FILE = "likes/films.csv"

_HALF_STAR = Decimal("0.5")
_MIN = Decimal(str(MIN_RATING))
_MAX = Decimal(str(MAX_RATING))


def film_key(name: str, year: int | None) -> str:
    """Join key for diary rows.

    diary.csv identifies films by diary-entry URI, which shares no namespace with the film
    URIs in ratings.csv — the two intersect at zero. Name+year is the only available join.
    See docs/DATA_FINDINGS.MD F6.
    """
    return f"{name.strip().casefold()}::{year if year is not None else ''}"


@dataclass(frozen=True)
class ParsedRating:
    letterboxd_uri: str
    name: str
    year: int | None
    rating: Decimal
    logged_date: date | None

    @property
    def key(self) -> str:
        return film_key(self.name, self.year)


@dataclass(frozen=True)
class ParsedDiaryEntry:
    name: str
    year: int | None
    watched_date: date | None
    is_rewatch: bool

    @property
    def key(self) -> str:
        return film_key(self.name, self.year)


@dataclass(frozen=True)
class ParsedLike:
    letterboxd_uri: str
    name: str
    year: int | None


@dataclass(frozen=True)
class IngestStats:
    ratings: int = 0
    diary_entries: int = 0
    likes: int = 0
    rewatches: int = 0

    # Join integrity for the name+year attach. Counted rather than assumed so a silent
    # failure shows up as a metric instead of as missing rows.
    diary_unique_films: int = 0
    diary_films_matched: int = 0

    skipped: Mapping[str, int] = field(default_factory=dict)
    files_read: tuple[str, ...] = ()
    files_missing: tuple[str, ...] = ()

    @property
    def diary_films_unmatched(self) -> int:
        return self.diary_unique_films - self.diary_films_matched


@dataclass(frozen=True)
class ParsedExport:
    ratings: tuple[ParsedRating, ...]
    diary: tuple[ParsedDiaryEntry, ...]
    likes: tuple[ParsedLike, ...]
    stats: IngestStats


class ExportSource:
    """Read-only access to an export, restricted to the files Ariadne is allowed to open.

    Accepts either the downloaded zip or an already-extracted directory. Exports are
    sometimes nested under a single top-level folder, which is stripped so callers always
    work with export-relative paths.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._zip: zipfile.ZipFile | None = None

        if path.is_dir():
            members = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
        elif zipfile.is_zipfile(path):
            self._zip = zipfile.ZipFile(path)
            members = {name for name in self._zip.namelist() if not name.endswith("/")}
        else:
            raise ValueError(f"{path} is neither a directory nor a zip archive")

        self._members = _strip_common_root(members)

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def __enter__(self) -> "ExportSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def has(self, relative_path: str) -> bool:
        return relative_path in self._members

    def read_text(self, relative_path: str) -> str:
        # The privacy boundary. profile.csv is not in EXPORT_FILES_USED, so no caller can
        # reach the user's email address, name or location through this class.
        if relative_path not in EXPORT_FILES_USED:
            raise PermissionError(
                f"{relative_path} is not an Ariadne input; allowed: {EXPORT_FILES_USED}"
            )

        member = self._members[relative_path]
        if self._zip is not None:
            raw = self._zip.read(member)
        else:
            raw = (self._path / member).read_bytes()
        return raw.decode("utf-8-sig")


def _strip_common_root(members: set[str]) -> dict[str, str]:
    """Map export-relative path to the real member path."""
    if not members:
        return {}

    first_segments = {m.split("/")[0] for m in members}
    if len(first_segments) == 1 and all("/" in m for m in members):
        root = first_segments.pop()
        return {m[len(root) + 1 :]: m for m in members}
    return {m: m for m in members}


def _rows(source: ExportSource, relative_path: str) -> Iterator[dict[str, str]]:
    yield from csv.DictReader(io.StringIO(source.read_text(relative_path)))


def _parse_year(raw: str) -> int | None:
    raw = raw.strip()
    return int(raw) if raw.isdigit() else None


def _parse_date(raw: str) -> date | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_rating(raw: str) -> Decimal | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None
    if not _MIN <= value <= _MAX:
        return None
    if value % _HALF_STAR != 0:
        return None
    return value


def parse_export(path: Path) -> ParsedExport:
    """Read an export into normalized rows, discarding everything Ariadne does not use."""
    with ExportSource(path) as source:
        skipped: Counter[str] = Counter()
        files_read: list[str] = []
        files_missing: list[str] = []

        ratings = tuple(_parse_ratings(source, skipped, files_read, files_missing))
        diary = tuple(_parse_diary(source, skipped, files_read, files_missing))
        likes = tuple(_parse_likes(source, skipped, files_read, files_missing))

    diary_keys = {entry.key for entry in diary}
    rating_keys = {rating.key for rating in ratings}

    stats = IngestStats(
        ratings=len(ratings),
        diary_entries=len(diary),
        likes=len(likes),
        rewatches=sum(1 for entry in diary if entry.is_rewatch),
        diary_unique_films=len(diary_keys),
        diary_films_matched=len(diary_keys & rating_keys),
        skipped=dict(skipped),
        files_read=tuple(files_read),
        files_missing=tuple(files_missing),
    )
    return ParsedExport(ratings=ratings, diary=diary, likes=likes, stats=stats)


def _parse_ratings(
    source: ExportSource,
    skipped: Counter[str],
    files_read: list[str],
    files_missing: list[str],
) -> Iterator[ParsedRating]:
    if not source.has(RATINGS_FILE):
        files_missing.append(RATINGS_FILE)
        return
    files_read.append(RATINGS_FILE)

    for row in _rows(source, RATINGS_FILE):
        uri = row.get("Letterboxd URI", "").strip()
        name = row.get("Name", "").strip()
        rating = _parse_rating(row.get("Rating", ""))

        if not uri or not name:
            skipped["ratings:missing_uri_or_name"] += 1
            continue
        if rating is None:
            skipped["ratings:unusable_rating"] += 1
            continue

        yield ParsedRating(
            letterboxd_uri=uri,
            name=name,
            year=_parse_year(row.get("Year", "")),
            rating=rating,
            # Date here is when the rating was logged, not when the film was watched.
            # Only diary rows carry a real watch date (F7).
            logged_date=_parse_date(row.get("Date", "")),
        )


def _parse_diary(
    source: ExportSource,
    skipped: Counter[str],
    files_read: list[str],
    files_missing: list[str],
) -> Iterator[ParsedDiaryEntry]:
    if not source.has(DIARY_FILE):
        files_missing.append(DIARY_FILE)
        return
    files_read.append(DIARY_FILE)

    for row in _rows(source, DIARY_FILE):
        name = row.get("Name", "").strip()
        if not name:
            skipped["diary:missing_name"] += 1
            continue

        yield ParsedDiaryEntry(
            name=name,
            year=_parse_year(row.get("Year", "")),
            watched_date=_parse_date(row.get("Watched Date", "")),
            is_rewatch=row.get("Rewatch", "").strip().casefold() in {"yes", "true", "1"},
        )


def _parse_likes(
    source: ExportSource,
    skipped: Counter[str],
    files_read: list[str],
    files_missing: list[str],
) -> Iterator[ParsedLike]:
    if not source.has(LIKES_FILE):
        files_missing.append(LIKES_FILE)
        return
    files_read.append(LIKES_FILE)

    for row in _rows(source, LIKES_FILE):
        uri = row.get("Letterboxd URI", "").strip()
        name = row.get("Name", "").strip()
        if not uri or not name:
            skipped["likes:missing_uri_or_name"] += 1
            continue

        yield ParsedLike(letterboxd_uri=uri, name=name, year=_parse_year(row.get("Year", "")))
