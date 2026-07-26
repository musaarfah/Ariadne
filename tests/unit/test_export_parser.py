import zipfile
from decimal import Decimal
from pathlib import Path

import pytest

from ariadne.core.ingest.export import (
    ExportSource,
    film_key,
    parse_export,
)

RATINGS_CSV = """Date,Name,Year,Letterboxd URI,Rating
2022-07-07,The Witch,2015,https://boxd.it/aaaa,5
2023-11-26,The Lighthouse,2019,https://boxd.it/bbbb,4
2024-02-01,Whiplash,2014,https://boxd.it/cccc,4.5
2024-02-02,Gangs of Wasseypur – Part 1,2012,https://boxd.it/dddd,5
"""

DIARY_CSV = """Date,Name,Year,Letterboxd URI,Rating,Rewatch,Tags,Watched Date
2024-02-02,Gangs of Wasseypur – Part 1,2012,https://boxd.it/entry1,5,Yes,,2024-01-30
2024-02-01,Whiplash,2014,https://boxd.it/entry2,4.5,,,2024-01-28
2024-03-01,A Film Not Rated,2001,https://boxd.it/entry3,,,,2024-02-20
"""

LIKES_CSV = """Date,Name,Year,Letterboxd URI
2023-11-25,The Witch,2015,https://boxd.it/aaaa
"""

PROFILE_CSV = """Date Joined,Username,Given Name,Email Address,Location,Bio
2022-07-07,someone,Real Name,secret@example.com,Somewhere,A bio
"""

# Exports contain these, and their filenames collide with the files we do want.
DELETED_DIARY_CSV = """Date,Name,Year,Letterboxd URI,Rating,Rewatch,Tags,Watched Date
2020-01-01,Deleted Entry,1999,https://boxd.it/gone,1,,,2020-01-01
"""


def write_export_dir(root: Path, *, nested: bool = False) -> Path:
    base = root / "letterboxd-someone-2026-01-01" if nested else root
    (base / "likes").mkdir(parents=True, exist_ok=True)
    (base / "deleted").mkdir(parents=True, exist_ok=True)
    (base / "orphaned").mkdir(parents=True, exist_ok=True)

    (base / "ratings.csv").write_text(RATINGS_CSV, encoding="utf-8")
    (base / "diary.csv").write_text(DIARY_CSV, encoding="utf-8")
    (base / "likes" / "films.csv").write_text(LIKES_CSV, encoding="utf-8")
    (base / "profile.csv").write_text(PROFILE_CSV, encoding="utf-8")
    (base / "deleted" / "diary.csv").write_text(DELETED_DIARY_CSV, encoding="utf-8")
    (base / "orphaned" / "diary.csv").write_text(DELETED_DIARY_CSV, encoding="utf-8")
    return root


def write_export_zip(root: Path, *, nested: bool = False) -> Path:
    staged = write_export_dir(root / "staged", nested=nested)
    archive = root / "export.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(staged.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staged))
    return archive


# --- the privacy boundary --------------------------------------------------------------


def test_profile_csv_cannot_be_read(tmp_path: Path):
    write_export_dir(tmp_path)
    with ExportSource(tmp_path) as source, pytest.raises(PermissionError):
        source.read_text("profile.csv")


def test_no_pii_survives_parsing(tmp_path: Path):
    write_export_dir(tmp_path)
    parsed = parse_export(tmp_path)

    blob = repr(parsed)
    for secret in ("secret@example.com", "Real Name", "Somewhere", "A bio", "someone"):
        assert secret not in blob


def test_reviews_and_comments_are_also_refused(tmp_path: Path):
    write_export_dir(tmp_path)
    with ExportSource(tmp_path) as source:
        for path in ("reviews.csv", "comments.csv", "watched.csv"):
            with pytest.raises(PermissionError):
                source.read_text(path)


# --- file selection --------------------------------------------------------------------


def test_deleted_and_orphaned_diaries_are_not_read(tmp_path: Path):
    """Exports contain deleted/diary.csv and orphaned/diary.csv alongside the real one."""
    write_export_dir(tmp_path)
    parsed = parse_export(tmp_path)

    names = {entry.name for entry in parsed.diary}
    assert "Deleted Entry" not in names
    assert parsed.stats.diary_entries == 3


def test_nested_export_root_is_stripped(tmp_path: Path):
    write_export_dir(tmp_path, nested=True)
    parsed = parse_export(tmp_path)
    assert parsed.stats.ratings == 4


def test_zip_and_directory_agree(tmp_path: Path):
    from_dir = parse_export(write_export_dir(tmp_path / "dir"))
    from_zip = parse_export(write_export_zip(tmp_path / "zip"))
    assert from_dir.stats.ratings == from_zip.stats.ratings
    assert from_dir.stats.diary_entries == from_zip.stats.diary_entries
    assert from_dir.stats.likes == from_zip.stats.likes


def test_nested_zip_root_is_stripped(tmp_path: Path):
    parsed = parse_export(write_export_zip(tmp_path, nested=True))
    assert parsed.stats.ratings == 4


def test_missing_files_are_reported_not_fatal(tmp_path: Path):
    (tmp_path / "ratings.csv").write_text(RATINGS_CSV, encoding="utf-8")
    parsed = parse_export(tmp_path)

    assert parsed.stats.ratings == 4
    assert parsed.stats.diary_entries == 0
    assert set(parsed.stats.files_missing) == {"diary.csv", "likes/films.csv"}


def test_rejects_a_path_that_is_neither_zip_nor_directory(tmp_path: Path):
    junk = tmp_path / "notanexport.txt"
    junk.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_export(junk)


# --- row parsing -----------------------------------------------------------------------


def test_ratings_are_parsed_as_exact_half_stars(tmp_path: Path):
    write_export_dir(tmp_path)
    parsed = parse_export(tmp_path)

    by_name = {r.name: r for r in parsed.ratings}
    assert by_name["Whiplash"].rating == Decimal("4.5")
    assert by_name["The Witch"].rating == Decimal("5")
    assert by_name["Whiplash"].year == 2014


def test_unicode_titles_survive_intact(tmp_path: Path):
    """Normalization belongs to the resolver; ingest must not mangle the source title."""
    write_export_dir(tmp_path)
    parsed = parse_export(tmp_path)
    assert any(r.name == "Gangs of Wasseypur – Part 1" for r in parsed.ratings)


@pytest.mark.parametrize(
    "rating_value",
    ["", "0", "6", "abc", "3.7"],
)
def test_unusable_ratings_are_skipped_and_counted(tmp_path: Path, rating_value: str):
    (tmp_path / "ratings.csv").write_text(
        "Date,Name,Year,Letterboxd URI,Rating\n"
        f"2024-01-01,Some Film,2001,https://boxd.it/xxxx,{rating_value}\n",
        encoding="utf-8",
    )
    parsed = parse_export(tmp_path)
    assert parsed.stats.ratings == 0
    assert parsed.stats.skipped["ratings:unusable_rating"] == 1


def test_rows_without_a_uri_are_skipped(tmp_path: Path):
    (tmp_path / "ratings.csv").write_text(
        "Date,Name,Year,Letterboxd URI,Rating\n2024-01-01,Some Film,2001,,4\n",
        encoding="utf-8",
    )
    parsed = parse_export(tmp_path)
    assert parsed.stats.ratings == 0
    assert parsed.stats.skipped["ratings:missing_uri_or_name"] == 1


def test_missing_year_is_tolerated(tmp_path: Path):
    (tmp_path / "ratings.csv").write_text(
        "Date,Name,Year,Letterboxd URI,Rating\n2024-01-01,Some Film,,https://boxd.it/xxxx,4\n",
        encoding="utf-8",
    )
    parsed = parse_export(tmp_path)
    assert parsed.stats.ratings == 1
    assert parsed.ratings[0].year is None


def test_rewatch_flag_is_read(tmp_path: Path):
    write_export_dir(tmp_path)
    parsed = parse_export(tmp_path)

    assert parsed.stats.rewatches == 1
    rewatched = [e for e in parsed.diary if e.is_rewatch]
    assert rewatched[0].name == "Gangs of Wasseypur – Part 1"


def test_watched_date_comes_from_diary_only(tmp_path: Path):
    write_export_dir(tmp_path)
    parsed = parse_export(tmp_path)

    whiplash = next(e for e in parsed.diary if e.name == "Whiplash")
    assert whiplash.watched_date is not None
    assert whiplash.watched_date.isoformat() == "2024-01-28"

    # ratings.csv Date is a log date, not a watch date (F7).
    rated = next(r for r in parsed.ratings if r.name == "Whiplash")
    assert rated.logged_date is not None
    assert rated.logged_date.isoformat() == "2024-02-01"


def test_bom_encoded_file_is_handled(tmp_path: Path):
    (tmp_path / "ratings.csv").write_text(RATINGS_CSV, encoding="utf-8-sig")
    parsed = parse_export(tmp_path)
    assert parsed.stats.ratings == 4


# --- the name+year join ----------------------------------------------------------------


def test_diary_join_is_counted(tmp_path: Path):
    write_export_dir(tmp_path)
    stats = parse_export(tmp_path).stats

    # Two of three diary films are rated; "A Film Not Rated" is not.
    assert stats.diary_unique_films == 3
    assert stats.diary_films_matched == 2
    assert stats.diary_films_unmatched == 1


def test_film_key_ignores_case_and_padding():
    assert film_key(" The Witch ", 2015) == film_key("the witch", 2015)


def test_film_key_separates_same_title_different_year():
    # Whiplash 2013 is the short film; 2014 is the feature.
    assert film_key("Whiplash", 2013) != film_key("Whiplash", 2014)
