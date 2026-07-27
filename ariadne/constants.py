"""Values that are fixed by the research design rather than configurable."""

from datetime import date

# Every stochastic operation seeds from this. The writeup has to be reproducible.
RANDOM_SEED = 20260726

# The reference account's ratings are two distributions, not one: 58% were entered in a
# 2023 backfill burst, rated from memory rather than logged on watch. The live-logged half
# is more generous (mean 3.59 vs 3.35) with nearly double the 5.0 rate. Splitting here
# separates the two regimes; see docs/DATA_FINDINGS.MD F3.
#
# This date is therefore a fact about ONE account, not about Letterboxd. Any other library has its
# own backfill boundary, and every command that splits accepts --cut so it can be given one. Choose
# it from the log-date distribution BEFORE looking at any model output: a cut chosen after seeing
# the scores is a cut chosen to flatter them.
TEMPORAL_SPLIT_DATE = date(2024, 1, 1)

# Ratings are half-stars from 0.5 to 5.0. There is no 0.
MIN_RATING = 0.5
MAX_RATING = 5.0

# Only these files are read from an export. Everything else is discarded at ingest,
# including profile.csv, which carries the user's email address, name and location.
EXPORT_FILES_USED = ("ratings.csv", "diary.csv", "likes/films.csv")
