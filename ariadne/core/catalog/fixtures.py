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


# The resolver regression set: every title from the reference export that is either
# non-ASCII or collides with another film of the same name, plus a control and one
# television entry. Titles are byte-exact copies of what Letterboxd exported, including the
# no-break spaces in two Star Wars entries.
SEED_SEARCHES: tuple[SeedSearch, ...] = (
    # Same-title collisions. Whiplash is the sharpest: 2013 is a short film, 2014 the
    # feature, one year apart with identical titles.
    SeedSearch("Whiplash", 2013, "collision: short film"),
    SeedSearch("Whiplash", 2014, "collision: feature"),
    SeedSearch("Nosferatu", 1922, "collision: century apart"),
    SeedSearch("Nosferatu", 2024, "collision: century apart"),
    SeedSearch("Joker", 2012, "collision"),
    SeedSearch("Joker", 2019, "collision"),
    SeedSearch("Aladdin", 1992, "collision: animated original"),
    SeedSearch("Aladdin", 2019, "collision: live action remake"),
    SeedSearch("Frozen", 2010, "collision"),
    SeedSearch("Frozen", 2013, "collision"),
    SeedSearch("Beauty and the Beast", 1991, "collision: animated original"),
    SeedSearch("Beauty and the Beast", 2017, "collision: live action remake"),
    SeedSearch("Les Misérables", 1998, "collision AND accent"),
    SeedSearch("Les Misérables", 2012, "collision AND accent"),
    # En dash, the most common oddity in the export (12 titles).
    SeedSearch("Mission: Impossible – Dead Reckoning", 2023, "en dash"),
    SeedSearch("Mission: Impossible – Rogue Nation", 2015, "en dash"),
    SeedSearch("Mission: Impossible – Ghost Protocol", 2011, "en dash"),
    SeedSearch("Mission: Impossible – Fallout", 2018, "en dash"),
    SeedSearch("Mission: Impossible – The Final Reckoning", 2025, "en dash, very recent"),
    SeedSearch("John Wick: Chapter 3 – Parabellum", 2019, "en dash"),
    SeedSearch("Gangs of Wasseypur – Part 1", 2012, "en dash, non-anglophone"),
    SeedSearch("Gangs of Wasseypur – Part 2", 2012, "en dash, non-anglophone"),
    SeedSearch("Golmaal – Fun Unlimited", 2006, "en dash, non-anglophone"),
    SeedSearch("Star Wars: Episode I – The Phantom Menace", 1999, "en dash"),
    # These two additionally contain U+00A0 after the dash, invisible on screen. Written as
    # an explicit escape so no edit or copy-paste can silently lose it.
    SeedSearch(
        "Star Wars: Episode II –\u00a0Attack of the Clones", 2002, "en dash + no-break space"
    ),
    SeedSearch(
        "Star Wars: Episode III –\u00a0Revenge of the Sith", 2005, "en dash + no-break space"
    ),
    # Accents and other non-ASCII, none of which NFKC alters.
    SeedSearch("Léon: The Professional", 1994, "accent"),
    SeedSearch("The Double Life of Véronique", 1991, "accent, translated title"),
    SeedSearch("Salò, or the 120 Days of Sodom", 1975, "accent; Letterboxd year disagrees"),
    SeedSearch("Bāhubali: The Beginning", 2015, "macron"),
    SeedSearch("Bāhubali 2: The Conclusion", 2017, "macron"),
    SeedSearch("Üç Harfliler: Fal", 2025, "Turkish, obscure, very recent"),
    # NFKC folds these two to ASCII.
    SeedSearch("Alien³", 1992, "superscript three folds to 3"),
    SeedSearch("WALL·E", 2008, "interpunct"),
    # Controls.
    SeedSearch("The Godfather", 1972, "control: should be trivial"),
    SeedSearch("Obi-Wan Kenobi", 2022, "television: must NOT match a movie"),
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
