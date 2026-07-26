# Ariadne — Decision Log

Chronological and **append-only**. Entries are never edited or deleted; when a decision changes, the
old entry is marked superseded and a new one is added. The point is the chain — what we thought, why,
what changed our minds.

This file lives in `research/` rather than `docs/` for two reasons. `research/` is tracked in git, so
every entry carries a verifiable timestamp. And the reasoning is shareable: a log where a decision gets
reversed by evidence says more than a log where everything was right first time.

Format per entry — **Context**, **Decision**, **Alternatives**, **Why**, **Status**.

Related: `docs/DATA_FINDINGS.MD` (empirical findings F1–F14), `docs/Ariadne.MD` (current spec),
`docs/PHASES.md` (plan), `docs/CURRENT_STATUS.md` (where we are now).

---

## Phase: choosing the project — 2026-07-26

### D1 — Build a film-domain project rather than cold-messaging alone
**Context.** Goal is to be hired into CS/SWE/data work, ideally somewhere in film tech, starting from a
well-kept Letterboxd account and cold outreach on LinkedIn.
**Decision.** Build a substantial film-data project first and let it speak for the outreach.
**Why.** A message with a link that makes someone ask "how did you do that?" is worth roughly fifty
without one.
**Status.** Active.

### D2 — Optimize for producing a *finding*, not a tool
**Context.** Portfolio projects are abundant; nothing about "a nice dashboard of your stats" travels.
**Decision.** Target work that produces knowledge — a dataset or a measured result — not only software.
**Why.** Tools impress; findings get shared, cited, and replied to.
**Status.** Active. Drives the Phase 1.9 writeup gate and the whole evaluation design.

### D3 — Three shortlisted ideas are one system, not three projects
**Context.** Three ideas survived a longer list: below-the-line crew graph, automated cinemetrics
(visual craft), Letterboxd review-corpus NLP.
**Decision.** Treat them as three modalities of one question — taste explained by human networks, by
visual craft, and by language — with a cross-modality study at the end.
**Alternatives.** Three unrelated repos.
**Why.** They share a join key (the user's export) and a spine (film identity resolution), and together
they permit a question none can ask alone: *what actually explains film taste best — genre, crew, craft,
or critical language?*
**Status.** Active. Recorded in `../PROJECTS.md`.

### D4 — Strictly one project at a time, taken all the way
**Context.** Three-in-one is a scope trap; the failure mode is 70% through all three and publishing
none.
**Decision.** Project A ships and is written up before Project B starts. No parallel work.
**Why.** Three shipped things beat one unfinished cathedral, and the outreach improves every month
instead of waiting on a big bang.
**Status.** Active.

### D5 — Project A (crew graph) goes first
**Context.** Crew graph is cheapest; cinemetrics is hardest; review corpus is in between.
**Decision.** Crew graph first.
**Why.** Cheapest to a live link, most immediately striking output, and it forces the film-identity
resolution spine that the other two both need — built while the stakes are low.
**Status.** Active.

---

## Phase: statistical design — 2026-07-26

### D6 — Fit on residuals, never on raw ratings
**Context.** The naive approach averages a user's ratings per crew member.
**Decision.** Target variable is rating minus expectation, where expectation comes from the film's
global average plus the user's own centre and spread.
**Alternatives.** Raw rating means with a popularity covariate.
**Why.** Consensus carries no personal information. A 5.0 for *The Godfather* is expected and tells us
nothing; a 5.0 for *Gangs of Wasseypur* is far above consensus and is dense with signal. **The model
learns from disagreement with consensus, not agreement.**
**Status.** Active. The central idea of the project.

### D7 — Empirical Bayes shrinkage, hand-written
**Context.** Reference account has ~1,345 rated films; the modal crew member appears **once**.
**Decision.** Shrink per-person effects toward the user's mean, strength estimated from data. Implement
by hand (~20 lines) rather than adopting a hierarchical modelling framework.
**Why.** With that sparsity, shrinkage is not a refinement, it is the whole ballgame. A framework would
be over-engineering for Stage 1 — revisit at Track 3.
**Status.** Active.

### D8 — Two model tracks, and report their disagreement
**Context.** Long-term director/DP/editor partnerships are the norm, so collaborators' effects are
confounded.
**Decision.** Track 1 = shrunken residual means (interpretable, shown to the user). Track 2 = ridge over
sparse crew indicators (splits credit jointly). Report where they disagree.
**Why.** Someone ranking high on Track 1 and collapsing on Track 2 had an effect that was really their
collaborator's. That is diagnostic information, not noise.
**Status.** Active.

### D9 — Refuse to guess when two people are inseparable
**Context.** Jarin Blaschke has shot only Robert Eggers films; in this library they cannot be told
apart.
**Decision.** Report "cannot be separated from [director] (n/n films)" instead of picking one.
**Alternatives.** Silently attribute to whichever scores higher, as every other recommender does.
**Why.** The honest statement is more useful than a confident wrong number — and it is a differentiator,
not a caveat.
**Status.** Active. Reinforced by D26.

---

## Phase: reference-data analysis — 2026-07-26

### D10 — Drop the watched-but-unrated ablation
**Context.** Planned as a way to expand training signal.
**Decision.** Removed.
**Why.** F1 — the reference account has 2 unrated films out of 1,347. There is no unrated tail to
exploit. The lever was designed for a problem this data does not have.
**Status.** Active.

### D11 — Precision@20 replaces MAE as the primary metric
**Context.** Original plan made MAE/RMSE the headline.
**Decision.** Precision@20 primary; Spearman secondary; MAE/RMSE demoted.
**Why.** F4 — 16.5% of ratings are exactly 5.0 (22.6% recently) and 71.9% are whole stars. Effective
resolution is ~5 levels, not 10, and the top level is an unordered 222-film mass. A model can win on MAE
by predicting 3.5 forever and be useless. The product ranks, so rank metrics measure the real job.
**Status.** Active. **Supersedes** the original metric choice.

### D12 — Temporal split respecified, not abandoned
**Context.** Temporal split was designated the honest headline.
**Decision.** Cut at 2024-01-01 (791/554), centre each block separately, report drift statistics
alongside the metric.
**Why.** F3 — the ratings are two distributions. 58% were entered in a 2023 backfill burst, rated from
memory; the live-logged half is more generous (mean 3.59 vs 3.35) with nearly double the 5.0 rate. A
naive split moves the error for reasons unrelated to model quality.
**Status.** Active. **Supersedes** the unqualified temporal split.

### D13 — Rewatch count becomes a secondary target
**Context.** 256 rewatch entries were unused.
**Decision.** Add as a secondary target, tested as an ablation.
**Why.** F9 — choosing to rewatch is arguably a stronger preference statement than 5 stars, and it is the
only available signal that discriminates *within* the unordered 5.0 mass from D11.
**Status.** Active.

### D14 — Two explicit validation stages; recruit for divergence
**Context.** n=1. The single biggest limitation, and unfixable with more of one person's data.
**Decision.** Stage 1 solo (pipeline + go/no-go). Stage 2 recruits 15 minimum, 25–30 ideal — chosen for
*divergent* taste, older cinema, non-Anglophone libraries.
**Why.** F2 — Ariadne is a per-user model, so more users do not improve any individual's estimates. What
they buy is a *distribution* of results instead of an anecdote, plus heterogeneity analysis and Track 3
pooling. Recruiting for divergence is also the only real fix for the era skew (83% post-2000).
**Status.** Active.

### D15 — Publish a negative result
**Context.** The thesis is falsifiable and might be false.
**Decision.** If crew does not beat director-only, publish anyway with the detection-floor analysis.
**Why.** Decided *before* any data was fit, which is far easier than deciding it after four weeks of
work. A well-measured negative is still a finding.
**Status.** Active.

### D16 — Scope all claims to post-2000 cinema
**Why.** F5 — 83.1% of the reference library is post-2000 and only 5.9% predates 1980, which is also
where TMDB credits are thinnest.
**Status.** Active.

---

## Phase: privacy and data sourcing — 2026-07-26

### D17 — Bring-your-own export; never scrape accounts
**Decision.** Users supply their own Letterboxd export. No login, password, OAuth, or scraping.
**Why.** Letterboxd has no open public API and scraping is ToS-hostile. Exports are legally clean,
faster, and choosing this deliberately is itself a signal.
**Status.** Active.

### D18 — PII boundary is structural, not procedural
**Context.** `profile.csv` contains email address, real name, location, bio.
**Decision.** `ExportSource` raises on any path outside `EXPORT_FILES_USED`; `ParsedExport` has no field
able to hold those values. Tests assert nothing survives parsing or persistence.
**Alternatives.** Strip fields after reading them.
**Why.** "We remember to strip it" fails eventually. A boundary that cannot be crossed does not.
**Status.** Active.

---

## Phase: architecture — 2026-07-26

### D19 — ~~CLI-first, SQLite, plain Python, no ORM~~
**Context.** First architecture proposal, optimized for research iteration speed.
**Decision.** Flat modules, SQLite, hand-written SQL, CLI output; web layer much later.
**Why (at the time).** Ten tables and a few thousand films do not need Postgres; a UI built before the
model is validated locks in the wrong shape.
**Status.** **SUPERSEDED by D20.**

### D20 — Full stack: Next.js + FastAPI + Postgres + Redis
**Context.** D19 was challenged directly: the project needs to read as a serious system to people who
might hire you.
**Decision.** Monorepo with a framework-free `core/`, FastAPI, RQ worker, Postgres, Redis, Next.js
frontend. Postgres from day one.
**Why — three separate reasons, one of them purely technical:**
1. **`pg_trgm`.** The resolver's fuzzy fallback is a trigram similarity search. In Postgres that is a GIN
   index and a `similarity()` query; in SQLite it is hand-rolled string distance over a full scan. A real
   capability difference in the hardest part of the pipeline — not ceremony.
2. **The stated objective.** D19 under-weighted the hiring signal. A CLI demonstrates none of API
   design, data modelling, job orchestration, or frontend work — the actual skills being advertised.
3. **The frontend is a research instrument.** D14 needs n≥15. Twenty-five friends will not clone a repo,
   install Python, and get a TMDB key — and the ones who won't are exactly the divergent, older-cinema
   accounts most needed. Upload-a-zip is the only realistic path to that sample size. Under D19, Stage 2
   was quietly gated behind friction that would have killed it.

Also: SQLite→Postgres migration later is genuinely annoying (types, upserts, sequences), and Postgres
was always the endpoint.
**Status.** Active. **Supersedes D19.**

### D21 — Reverse the no-ORM call: SQLAlchemy 2.0 + Alembic
**Why.** Follows from D20. Alembic needs SQLAlchemy for autogenerate, and typed models pay for
themselves across an API boundary.
**Status.** Active. **Supersedes** part of D19.

### D22 — Frontend held to Phase 3, behind two gates
**Context.** D20 brings the frontend forward in importance; the risk is building it too early.
**Decision.** No frontend until the director-only go/no-go is answered *and* the Stage 1 writeup exists.
**Why.** Not because the project is simple — risk ordering. If crew does not beat director-only, the
product changes shape and any UI is wasted. The writeup gate is the insurance policy.
**Status.** Active.

### D23 — No accounts; token-keyed persisted uploads
**Decision.** No login, password, or email. Results live at an unguessable token URL. Uploads *are*
persisted, keyed by that token.
**Alternatives.** Full accounts; or no persistence at all.
**Why.** Smallest attack surface consistent with Stage 2's needs. Persistence matters because Phase 4
re-analysis after Track 3 pooling would otherwise mean re-recruiting all 25 participants.
**Status.** Active. Removed the `users` table (D25).

### D24 — `docs/` untracked, `research/` tracked, `CLAUDE.md` at root
**Decision.** Design docs and working instructions stay local; `research/` is committed.
**Why.** `research/` holds shareable output and the git-timestamped pre-registration, where the timestamp
*is* the evidence. `CLAUDE.md` sits at the root because Claude Code only auto-loads it from there.
**Known cost.** The design docs therefore have **no git history** — overwriting a section destroys the
prior reasoning permanently. This log exists to compensate, which is why it lives in `research/`.
**Status.** Active.

---

## Phase: implementation decisions — 2026-07-26

### D25 — Single `ariadne/` package; no `users` table
**Decision.** `core`, `db`, `cli`, `api`, `worker` live inside one installable package rather than as
sibling top-level directories. No `users` table — the upload token is the unit of identity.
**Why.** One distribution, no top-level name collisions. The missing table follows from D23.
**Status.** Active. Both recorded as spec deviations in `Ariadne.MD` §9.

### D26 — Fetch and store all credits, every department
**Decision.** Role scope is a config value, not a schema commitment.
**Why.** One API call returns the whole credit list, so filtering at fetch time saves nothing and would
mean refetching 1,345 films after any change of mind.
**Status.** Active. Resolved the open "role scope" question architecturally.

### D27 — Add a `likes` table (spec gap)
**Context.** The spec listed `likes/films.csv` as an input but its data model had no table for it.
**Decision.** Add the table and persist likes now, before anything reads them.
**Why.** Parsing and discarding would leave the Phase 1.7 ablation with nothing to run on — and unlike
the author's own export, a participant's cannot be re-requested later.
**Status.** Active.

### D28 — Declare non-ORM indexes in the models too
**Context.** The trigram index was first created by raw SQL in the migration only. `alembic check`
revealed the next autogenerate would emit a `DROP INDEX` for it.
**Decision.** Declare it in `Film.__table_args__` with `postgresql_using="gin"`; keep `alembic check` in
CI.
**Why.** The failure would have been silent and delayed — fuzzy matching degrading to a full scan months
later with no error.
**Status.** Active.

### D29 — Match export members on exact relative paths
**Context.** Exports contain `deleted/diary.csv` and `orphaned/diary.csv` beside the real `diary.csv`.
**Decision.** Match exact export-relative paths after stripping an optional single top-level folder.
**Why.** Filename matching would have silently ingested diary entries the user deleted.
**Status.** Active.

---

## Phase: TMDB findings — 2026-07-26

### D30 — ~~Year is a hard equality constraint in matching~~
**Decision.** Require the candidate's year to equal the Letterboxd year exactly.
**Why (at the time).** It is the only thing separating `Whiplash` 2013 from 2014.
**Status.** **SUPERSEDED by D31.**

### D31 — The resolver ranks candidates; year tolerance is conditional
**Context.** F12 — `Salò` is 1975 on Letterboxd and 1976 on TMDB. D30 would have **rejected the correct
film**. But plain ±1 tolerance breaks `Whiplash`, two distinct films one year apart with identical
titles.
**Decision.**
1. Exact year match on a candidate with an exactly matching folded title wins outright.
2. Otherwise ±1 year is acceptable **only when exactly one candidate falls in that window**. Ambiguity
   inside the window is a resolution failure, not a coin flip.
3. A gap of 2+ years is a rejection.
**Why.** Satisfies both cases, and makes `match_method` and `confidence` carry real information.
**Status.** Active. **Supersedes D30.**

### D32 — Demote unicode normalization; require dash folding
**Context.** F8 listed 22 non-ASCII titles as resolution risks.
**Decision.** Keep NFKC as a cheap safeguard; add en/em-dash → hyphen folding. Do not build
transliteration.
**Why.** F10 — TMDB stores the same codepoints Letterboxd exports. `Alien³`, `Bāhubali`, `Léon`,
`WALL·E`, `Salò` all match exactly. F11 — the en dash is the one genuine mismatch
(`Gangs of Wasseypur – Part 1` vs TMDB's hyphen). The predicted hard problem was mostly not real.
**Status.** Active. Partially **supersedes** the framing in F8.

### D33 — Title similarity alone can never accept a match
**Context.** F13 — `Obi-Wan Kenobi` is a miniseries. TMDB movie search returns no film by that name; its
top hit is `Obi-Wan Kenobi: A Jedi's Return`, a documentary.
**Decision.** A candidate must clear both a title threshold *and* the D31 year rule. TV detection and
exclusion is a correctness requirement with a reported count.
**Why.** Resolving to a plausible wrong answer is worse than failing: the documentary's crew would enter
the taste profile and nothing downstream would look wrong.
**Status.** Active.

### D34 — Rate limiter owns its own clock
**Context.** One injected `sleep` was shared by the rate limiter and the retry backoff.
**Decision.** Separate them; the limiter keeps its own clock.
**Why.** Sharing conflated two unrelated concerns and made neither testable in isolation — caught by
tests that could not express what they meant.
**Status.** Active.

### D35 — DB upsert deferred from 1.2 to 1.3
**Decision.** Phase 1.2 is the client and fixtures only; persistence lands with the normalizer in 1.3.
**Why.** `films.normalized_title` is `NOT NULL` and cannot be computed without the normalizer.
Inventing a placeholder would mean maintaining two normalizers.
**Status.** Active.

---

## Phase: process — 2026-07-26

### D36 — No commits without explicit instruction
**Decision.** Work is finished, verified, and left in the tree for review. "Phase complete" is not
permission to commit.
**Status.** Active. Recorded in `CLAUDE.md` §7.

### D37 — This log exists and is append-only
**Context.** Decisions were being recorded across four files, superseded reasoning was being overwritten,
and `docs/` has no git history (D24).
**Decision.** Maintain `research/DECISIONS.md` chronologically and append-only. Never edit or delete an
entry; mark it superseded and add a new one.
**Why.** The chain is the artifact. A log showing a decision reversed by evidence is a stronger signal
than one where everything was right the first time.
**Status.** Active.
