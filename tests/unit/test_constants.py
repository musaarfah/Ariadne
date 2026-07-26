from datetime import date

from ariadne.constants import (
    EXPORT_FILES_USED,
    MAX_RATING,
    MIN_RATING,
    RANDOM_SEED,
    TEMPORAL_SPLIT_DATE,
)


def test_temporal_split_is_the_documented_date():
    # Changing this silently would invalidate every metric comparison in research/.
    assert TEMPORAL_SPLIT_DATE == date(2024, 1, 1)


def test_rating_bounds_match_letterboxd():
    assert (MIN_RATING, MAX_RATING) == (0.5, 5.0)


def test_seed_is_fixed():
    assert isinstance(RANDOM_SEED, int)


def test_profile_csv_is_never_read():
    # profile.csv holds the user's email address, real name and location.
    assert "profile.csv" not in EXPORT_FILES_USED
