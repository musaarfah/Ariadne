"""Recording TMDB responses to disk so the test suite never touches the network.

The seed list is chosen for the resolver regression set: same-title collisions, unicode and
dash handling, and one television entry that movie search should not match.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ariadne.core.catalog.tmdb import TmdbClient, TmdbNotFound

FIXTURE_DIR = Path("fixtures/tmdb")


@dataclass(frozen=True)
class SeedSearch:
    title: str
    year: int | None
    note: str


# Whiplash is the important one: 2013 is the short film, 2014 is the feature. Any resolver
# that treats year as a tiebreaker rather than a constraint gets this wrong.
SEED_SEARCHES: tuple[SeedSearch, ...] = (
    SeedSearch("Whiplash", 2013, "short film"),
    SeedSearch("Whiplash", 2014, "feature"),
    SeedSearch("Nosferatu", 1922, "same title, different era"),
    SeedSearch("Nosferatu", 2024, "same title, different era"),
    SeedSearch("Gangs of Wasseypur – Part 1", 2012, "en dash, non-anglophone"),
    SeedSearch("Bāhubali: The Beginning", 2015, "macron"),
    SeedSearch("Alien³", 1992, "superscript digit"),
    SeedSearch("WALL·E", 2008, "interpunct"),
    SeedSearch("Salò, or the 120 Days of Sodom", 1975, "grave accent, comma"),
    SeedSearch("Léon: The Professional", 1994, "accent, subtitle"),
    SeedSearch("The Godfather", 1972, "control case"),
    SeedSearch("Obi-Wan Kenobi", 2022, "television, should not match a movie"),
)

# Films whose full detail and credits are recorded, for credit-parsing tests.
SEED_FILMS: tuple[int, ...] = (
    238,  # The Godfather
    244786,  # Whiplash (2014)
    310131,  # The Witch
    503919,  # The Lighthouse
)


def search_fixture_name(title: str, year: int | None) -> str:
    # ASCII-only so fixture filenames are identical on every platform and filesystem
    # normalisation form. The unicode lives inside the file, not in its name.
    safe = "".join(c if c.isascii() and c.isalnum() else "_" for c in title).strip("_").lower()
    return f"search_{safe}_{year if year is not None else 'noyear'}.json"


def write_fixture(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_fixture(name: str, directory: Path = FIXTURE_DIR) -> Any:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def capture_all(client: TmdbClient, directory: Path = FIXTURE_DIR) -> list[str]:
    """Record every seed response. Returns the fixture names written."""
    written: list[str] = []

    for seed in SEED_SEARCHES:
        results = client.search_movies(seed.title, seed.year)
        name = search_fixture_name(seed.title, seed.year)
        write_fixture(
            {"query": seed.title, "year": seed.year, "note": seed.note, "results": results},
            directory / name,
        )
        written.append(name)

    for tmdb_id in SEED_FILMS:
        try:
            movie = client.get_movie(tmdb_id)
            credits = client.get_credits(tmdb_id)
        except TmdbNotFound:
            continue
        write_fixture(movie, directory / f"movie_{tmdb_id}.json")
        write_fixture(credits, directory / f"credits_{tmdb_id}.json")
        written.extend([f"movie_{tmdb_id}.json", f"credits_{tmdb_id}.json"])

    return written
