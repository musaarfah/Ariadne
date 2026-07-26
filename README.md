# Ariadne

Find the people who actually made the films you love.

Everyone knows their favorite director. Almost nobody knows their favorite **editor**,
**cinematographer**, **composer**, or **production designer** — even though those people often explain
taste better than the director does.

Ariadne reads your own Letterboxd export and finds the below-the-line collaborators that genuinely
predict what you rate highly, then recommends films along crew edges no other tool traverses.

---

## What makes this harder than a `GROUP BY`

Averaging your ratings per crew member fails in two ways, and handling both is the point of the
project.

**Consensus carries no information.** A 5-star rating for *The Godfather* tells us almost nothing about
you — nearly everyone rates it near the ceiling. A 5-star rating for something the world is lukewarm on
is dense with signal. So the model is fit on **rating minus expectation**, not raw rating.

**Crews are sticky.** A cinematographer who has only ever shot for one director inherits that
director's entire effect. When two people are mathematically inseparable in your library, Ariadne says
so — *"cannot be separated from [director] (2/2 films)"* — instead of picking one and sounding
confident.

Everything is validated against a ladder of baselines, including a **director-only** model. If
below-the-line crew doesn't beat director-only on a temporal split, the project's central claim is
false, and we report that.

---

## Status

**Stage 1: in development.** Nothing is usable yet.

| Stage | Scope |
|---|---|
| **1 — solo** | Full pipeline validated on a single 1,345-film reference account |
| **2 — multi-user** | 15–30 recruited accounts; hierarchical pooling; publishable result |
| **3 — web** | Upload-a-zip web interface. Only after the model is validated |

See `docs/Ariadne.MD` for the full spec, `docs/PHASES.md` for the execution plan, and
`docs/DATA_FINDINGS.MD` for the data analysis the design decisions rest on.

---

## Usage (planned)

Get your data: **letterboxd.com → Settings → Import & Export → Export Your Data**

```bash
ariadne analyze path/to/letterboxd-export.zip
```

No Letterboxd login. No password. No OAuth. No scraping. Just the zip you already own.

## Development

```bash
docker compose -f infra/docker-compose.yml up -d   # Postgres 16 + Redis
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                               # add your TMDB API key
alembic upgrade head
ariadne db-check                                   # verifies Postgres, pg_trgm, Redis, schema
```

Checks:

```bash
ruff check . && ruff format --check .
mypy ariadne
alembic check          # fails if models and migrations have drifted
pytest -q              # integration tests need the containers up
```

---

## Privacy

A Letterboxd export's `profile.csv` contains your **email address**, real name, and location. Ariadne
**strips these at ingest, before anything is written to disk.** Only `ratings.csv`, `diary.csv`, and
`likes/films.csv` are read; everything else in the export is discarded.

Personal exports are gitignored and never committed.

---

## Data sources

- **Your Letterboxd export** — ratings, diary, likes
- **[TMDB](https://www.themoviedb.org/)** — crew credits and film metadata

This product uses the TMDB API but is not endorsed or certified by TMDB.

---

## License

TBD
