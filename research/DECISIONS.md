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

---

## Phase: building the resolver — 2026-07-26

### D38 — Title exactness outranks the year
**Context.** D31 specified a year rule but left the priority between title and year implicit. The
first working implementation checked exact-year first.
**Consequence observed.** `Salò, or the 120 Days of Sodom` resolved to
**`Backstage on the Set of Salò, or the 120 Days of Sodom`**, because that documentary carries
Letterboxd's stated 1975 while the film itself is dated 1976. The wrong crew would have entered the
profile with nothing downstream looking wrong.
**Decision.** Candidates with an exactly matching title are considered first; the year discriminates
*within* that set. Only if no exact title exists anywhere in the year window is a fuzzy match
considered.
**Why.** F16. Refusing is cheap and visible; a plausible wrong match is neither.
**Status.** Active. Refines D31.

### D39 — A non-exact title must clear 0.75 similarity
**Context.** The first threshold, 0.30, accepted `Obi-Wan Kenobi` → `Obi-Wan Kenobi: A Jedi's Return`
(a documentary) at 0.45.
**Decision.** `ACCEPT_SIMILARITY = 0.75`.
**Why.** Calibrated against measured cases rather than chosen: `Mission: Impossible – Dead Reckoning`
→ `... Part One` scores 0.79 and must be accepted (TMDB renamed the film); the Salò backstage
documentary scores 0.65 and the Obi-Wan documentary 0.45, both of which must be rejected. Comfortable
margin on both sides.
**Status.** Active. Delivers the F13/D33 requirement.

### D40 — Vote dominance resolves duplicate title-and-year entries
**Context.** F15 — three collisions have multiple TMDB entries sharing title *and* year, so the year
rule cannot separate them. Refusing all of them would cost 3 of 36 regression cases.
**Decision.** Accept the leader only if it has ≥50 votes and ≥10× the runner-up. Otherwise refuse.
Recorded as a distinct `MatchMethod.DOMINANT` so these are countable and auditable separately.
**Alternatives.** Refuse always (costs ~8% of resolution rate); take the most popular unconditionally
(guessing).
**Why.** The observed margins are overwhelming — Aladdin 12,120 vs 68, Frozen 2,029 vs 2, Beauty and
the Beast 16,079 vs 0. The duplicates are bootlegs, shorts and a museum filming. Crucially this uses
popularity to disambiguate **identity**, not to predict **taste**, so it does not reintroduce the
prestige confound of D6.
**Status.** Active.

### D41 — Reimplement pg_trgm in Python rather than use difflib
**Context.** Resolution scores API candidates in memory but will score our own catalog in Postgres.
**Decision.** Implement pg_trgm's exact definition in Python; assert parity against Postgres in an
integration test.
**Why.** A threshold calibrated on one path would be wrong on the other if the measures differed.
Verified: 0 mismatches over 20 pairs, identical trigram sets for ASCII, and multibyte values agreeing
to 3.8e-08.
**Status.** Active.

### D42 — Correction: pg_trgm's value is the local catalog, not API candidates
**Context.** D20 justified Postgres partly on `pg_trgm` powering "the resolver's fuzzy fallback."
**Correction.** That reason was wrong. Candidates arrive from the API, number at most ~20, and are
scored in memory — no index helps. `pg_trgm` earns its place on the *local catalog* fast path, which
avoids an API call once films are cached. That is precisely the Stage 2 economics of resolving the
15th recruited account.
**Status.** The conclusion of D20 stands; this entry records that one of its three stated reasons was
imprecise. See F17.

### D43 — Add a `reason` column to resolutions
**Decision.** Store the human-readable reason the resolver decided as it did.
**Why.** The 1.4 hand audit needs to see *which rule fired*, not just an id. "Exact title, year off by
1" versus "votes dominate" is the difference between an audit and staring at numbers.
**Status.** Active.

### D44 — Match Postgres on `similarity('', '')`
**Context.** Our implementation returned 1.0 for two empty strings; Postgres returns 0.0.
**Decision.** Return 0.0.
**Why.** Agreement is this module's entire purpose (D41), and it is the safer answer — an empty title
should match nothing.
**Status.** Active.

### D45 — Parity tests must respect float4 and trigram hashing
**Context.** An initial parity test demanded 1e-9 agreement and "failed" on 8 pairs that were
correct.
**Decision.** Assert `rel=1e-6`; compare trigram *sets* only for ASCII input and *values* for
multibyte.
**Why.** F18 — `similarity()` returns `real` (~1.2e-07 relative precision), and `show_trgm()` hashes
multibyte trigrams to hex. The tolerance was unsatisfiable, not the code wrong.
**Status.** Active.

### D46 — NFKC is load-bearing (revises D32)
**Context.** D32 demoted NFKC to "a cheap safeguard" after F10 found TMDB stores the same codepoints
Letterboxd exports.
**Correction.** Measuring the whole export found **U+00A0, a no-break space**, in two Star Wars
titles. It is invisible on screen, and NFKC is the only thing that folds it. Without NFKC both films
fail to resolve and nobody inspecting the titles could see why.
**Decision.** Keep NFKC, applied before dash folding. Store the character as an explicit `\u00a0`
escape in the fixtures so no edit can silently lose it.
**Why.** F19.
**Status.** Active. **Supersedes** the framing in D32; dash folding is still the highest-volume fix
(12 titles) but NFKC is not optional.

### D47 — Ratings carry the title and year; resolutions hold only resolver output
**Context.** `ratings` stored only the Letterboxd URI, so nothing downstream knew *what* to resolve.
The first fix seeded a `resolutions` row per rating at ingest, carrying name and year with method
`UNRESOLVED`.
**Consequence observed.** Those seed rows were indistinguishable from genuine cache hits, so the
resolver read them back as "already attempted" and resolved nothing at all — 0/200 on the first run.
**Decision.** Put `name` and `year` on `ratings`. Drop the seeding entirely. A row in `resolutions`
now unambiguously means "attempted".
**Alternatives.** Add a `PENDING` enum value to represent the seeded state.
**Why.** Removing the ambiguity beats adding a state to describe it. Fewer states, no third enum
migration, and the two tables get cleaner responsibilities: `ratings` records what the user's export
said, `resolutions` records what the resolver concluded.
**Status.** Active.

### D48 — Television is recorded on resolutions, never in films
**Context.** `films.media_type` existed to hold TV so it could be "excluded with a reported count".
**Correction.** That cannot work. **TMDB movie ids and TV ids are separate namespaces** and can collide
on the same integer, while `films.tmdb_id` is the primary key — storing a series could silently
overwrite or be mistaken for an unrelated film.
**Decision.** Drop `films.media_type`. Add `MatchMethod.TELEVISION`, so a Letterboxd TV entry is a
resolution outcome with a NULL `tmdb_id`. The exclusion count becomes a one-line query.
**Why.** Keeps the count reportable as F13 requires, with no possibility of a wrong join. A test
asserts a TV id never enters `films`.
**Status.** Active. Supersedes the `media_type` design.

### D49 — Refuse `Split` rather than let popularity cross a year gap
**Context.** Letterboxd dates Shyamalan's *Split* 2016; TMDB dates it 2017 (id 381288, 18,447 votes).
Four obscure films genuinely titled *Split* do carry 2016 (53, 25, 8 and 3 votes). The resolver
confined itself to exact-year candidates, found no dominant one, and refused.
**Decision.** Leave it refused. Do **not** extend vote dominance across a year gap.
**Why.** The tempting fix — take the overwhelmingly popular candidate even when a year off — is
exactly the rule that would resolve a `Whiplash` 2013 query to the 2014 feature (16,828 votes against
the short's 419). Nothing distinguishes the two situations structurally: both are "obscure film with
the stated year versus famous film one year off". Under D9 and D33 a visible refusal beats a silent
wrong match.
**Mitigation.** The refusal reason now names the strong near-year candidate, so the 1.4 hand audit can
resolve it in one glance. `MatchMethod.MANUAL` exists for exactly this.
**Status.** Active. Accepted cost: a small, measured, visible dent in resolution rate.

### D50 — `set_config()` rather than `SET LOCAL`
**Context.** Setting `pg_trgm.similarity_threshold` per transaction failed: `SET` does not accept bind
parameters.
**Decision.** `SELECT set_config('pg_trgm.similarity_threshold', :value, true)`.
**Why.** Keeps the value parameterized instead of interpolating it into SQL.
**Status.** Active.

### D51 — The `%` operator, not `similarity() >= x`
**Context.** The local-catalog query needs the GIN trigram index to be used.
**Decision.** Filter with `normalized_title % :query` and order by `similarity(...) DESC`.
**Why.** Only the `%` operator is index-backed; a `similarity() >= x` predicate would force a full
scan, which is precisely what F17/D42 identified as the reason Postgres is here at all.
**Status.** Active.

---

## Phase: the data-quality gate — 2026-07-26

### D52 — A numeral guard, because trigram similarity cannot see sequels
**Context.** The first full-library run produced **8 wrong matches**, all of them a film resolved to
its own sibling: *Gangs of Wasseypur Part 2* → *Part 1*, *Kill Bill Vol. 1* → *Vol. 2*,
*Nymphomaniac Vol. II* → *Vol. I*, *Harry Potter … Part 1* → *Part 2*, *Back to the Future Part III* →
*Part II*, *Jatt & Juliet 2* → *Jatt & Juliet*, *Justice League Dark* → *Justice League*, and
*Whiplash* (2013 short) → *Whiplash* (2014 feature).
**Cause.** Trigram similarity is nearly blind to numerals. *Back to the Future Part III* against
*Part II* scores **0.963**; *Harry Potter … Part 1* against *Part 2* scores 0.907. The numeral carries
almost all of the meaning and almost none of the trigram weight.
**Decision.** Compare sequence markers separately from similarity. Roman numerals and number words
fold to digits, so "Part One", "Part I" and "Part 1" are one marker. A candidate is rejected outright
when our title carries a numeral it lacks, or when both carry different ones.
**Asymmetry, deliberately.** TMDB routinely holds a *longer* title for the same film — *Glass Onion: A
Knives Out Mystery*, *… Dead Reckoning Part One* — so extra words or numerals on TMDB's side prove
nothing. Extra words on *our* side mean we would be discarding what identifies the film, so
*Justice League Dark* cannot match *Justice League*.
**Status.** Active. All 8 cases are now regression tests.

### D53 — The local catalog answers only on an exact year
**Context.** *Whiplash* (2013) resolved to the 2014 feature. The 2014 film had already been cached, and
the local-catalog lookup offered it as a near-year candidate — one TMDB's own year-filtered search
would never have returned.
**Decision.** `local_candidates` requires `year == year`, not the ±1 window the API path uses. Year
disagreements go to the API.
**Why.** The local path is an optimisation and should be the conservative one. TMDB's search already
constrains candidates by year, so relaxing the constraint locally created a failure mode that existed
nowhere else.
**Status.** Active.

### D54 — Test the path the data takes, not the path that is easy to fixture
**Context.** The 36-case regression set passes and contains *Whiplash* 2013 vs 2014 explicitly. The
resolver still got it wrong in production.
**Cause.** The regression set is built from recorded API search fixtures, and TMDB's year filter means
the 2013 query returns only the short. The bug lived entirely in the local-catalog path, which had no
equivalent test.
**Decision.** Integration tests now cover local-catalog resolution directly, including the exact-year
restriction. Treat "the regression set passes" as evidence about one path only.
**Why.** This is the more uncomfortable lesson of 1.4: **a green suite gave false confidence about the
single case it was written to protect.** Coverage of code is not coverage of paths the data actually
takes.
**Status.** Active.

### D55 — A TMDB-id collision check is a first-class integrity metric
**Context.** The 8 wrong matches were found by noticing 1,296 resolutions mapped to only 1,288 distinct
films, then listing the duplicates. The audit would have found them eventually; this found them in one
query.
**Decision.** Report distinct-films-per-resolution and list every TMDB id claimed by more than one
Letterboxd entry as part of the Level 1 metrics.
**Why.** Two different films resolving to one id is almost always an error and is cheap to detect. It is
the highest-yield integrity check discovered so far.
**Status.** Active.

### D56 — Precision is measured over matches; refusals are reported separately
**Context.** The audit covers five strata, two of which are refusals (television, unresolved).
**Decision.** Precision counts only `exact`, `trigram` and `dominant`. Television and unresolved get
their own accuracy figure.
**Why.** A refusal is not a wrong answer. Folding refusals into precision would report recall as though
it were precision, and would have turned a 100% precision result into 96.6% by mixing in three known
false negatives — a number that means nothing.
**Status.** Active.

### D57 — Audit the risky strata exhaustively, sample only the safe one
**Context.** A uniform 100-case sample of 1,297 resolutions would have been roughly 94 `exact` matches
and a handful of everything else.
**Decision.** Audit every non-exact outcome (125 cases) and sample the `exact` stratum (50 of 1,220,
fixed seed).
**Why.** It measures the paths that can actually be wrong with no sampling error, and spends the
sampling budget on the stratum where a mistake is least likely. A uniform sample would have measured
the safe path precisely and the risky paths barely at all.
**Status.** Active.

### D58 — Leave the three subtitle false negatives unfixed
**Context.** F31 — `Glass Onion` and `Wake Up Dead Man` are refused because TMDB appends
": A Knives Out Mystery", scoring 0.38 and 0.46.
**Decision.** Accept the loss for now.
**Why.** Accepting prefix matches would also accept `Obi-Wan Kenobi: A Jedi's Return` for the
miniseries, converting a correct television call into a wrong film match. Three films out of 1,297 is
0.23%; trading audited 100% precision for that is a bad exchange. Precision protects every downstream
metric, recall costs three rows.
**Status.** Active, revisit at 1.5.

---

## Phase: credits — 2026-07-26

### D59 — One request per film, using `append_to_response=credits`
**Context.** Credits and detail were going to be two calls per film, 2,594 in total.
**Decision.** `/movie/{id}?append_to_response=credits`.
**Why.** Halves the request count, and detail is needed regardless because search results omit
`origin_country` — the field F29 identified as blocking the region coverage metric. One call now fills
in country, refreshes vote counts, and returns the full crew.
**Status.** Active.

### D60 — Role whitelists are exact job strings, guarded by department
**Context.** TMDB's Editing department on *The Godfather* holds two `Editor`, five `Assistant Editor`
and two `Additional Editor`. Its Camera department holds one `Director of Photography` alongside
`Camera Operator`, `Still Photographer` and `Assistant Camera`.
**Decision.** Each modelled role whitelists exact job strings and requires a matching department.
**Alternatives.** Substring matching on "editor", "photography", "music".
**Why.** Loose matching would put five assistant editors into the taste model alongside the one person
whose choices shaped the cut, diluting the very effect being measured. The department guard stops a
`Music` credit filed under Production from reading as a composer.
**Status.** Active.

### D61 — Source-material credits are excluded from `writer`
**Decision.** `Screenplay`, `Writer` and `Screenwriter` count. `Novel`, `Book`, `Story` and
`Story Editor` do not.
**Why.** A novelist credited on an adaptation did not work on the film. Ariadne is about
collaborators, and treating Mario Puzo's novel credit as a below-the-line contribution would be a
category error.
**Status.** Active. Worth revisiting only if the writer role turns out to be uselessly sparse.

### D62 — Credits ingestion is resumable by default
**Decision.** Films already holding credits are skipped unless `--refresh` is passed.
**Why.** 1,297 sequential requests will be interrupted sooner or later, and re-fetching what is
already stored spends rate budget for nothing. Same reasoning as `--retry-failed` on resolution.
**Status.** Active.

### D63 — Count films where a role has more than one credited person
**Context.** *The Godfather* credits two editors.
**Decision.** Coverage reports, per role, how many films credit more than one person in it.
**Why.** The model has to decide whether to split the effect, attribute to both, or take the first,
and that decision needs to be informed by how often the case arises rather than assumed away.
**Status.** Active. Feeds the 1.6/1.7 design.

### D64 — "Top-10 per role" is downgraded to "top-N, and N may be very small"
**Context.** F35 — measured sparsity. At a threshold of ≥8 films the estimable population is 11–46
people per role, matching the §6 projection. At ≥12 it collapses to **1–4 people for four of the six
roles** (cinematographer 1, production designer 1, director 2, editor 3, writer 4). Composer is the
sole exception at 29.
**Decision.** The product promises a top-N per role where N varies by role and may be a single name or
none. The insufficient-data bucket is a primary surface, not a footnote.
**Why.** Which regime applies is decided by the 1.7 detection floor, which is not yet measured — and F4
makes a floor above 8 plausible, since 71.9% of ratings are whole stars and 222 films sit unordered at
5.0. Promising a top-10 before knowing the floor would be committing to a product the data may not
support.
**Status.** Active. Revisit once 1.7 measures the floor.

### D65 — Estimation and traversal separation is now load-bearing
**Context.** F36 — people with ≥8 films account for 6–14% of credits per role (composer 37%).
**Decision.** Effects are estimated for the well-sampled few; recommendations traverse the whole graph
of 77,037 people.
**Why.** This was recorded as a cheap-now, annoying-later precaution in §6. The measurement shows it is
the difference between a recommender that can reach ~10% of the library and one that can reach all of
it.
**Status.** Active, and promoted from precaution to requirement.

### D66 — Drop the era-coverage caveat for four roles; keep it for production design
**Context.** F5 predicted coverage would be worst pre-1980. F33 measured the opposite: the 1970s show
100% editor and 100% cinematographer coverage.
**Decision.** Stop caveating editor, cinematographer, composer and writer on era. Keep the caveat for
production design, which really does degrade with age (17% in the 1940s).
**Why.** The prediction was wrong and carrying a false caveat costs credibility.
**Note.** D16 — scoping claims to post-2000 cinema — is unaffected: it rests on the rating
distribution, not on coverage.
**Status.** Active.

### D67 — Report production design's coverage next to its results
**Context.** F34 — 79.6% overall but **49% for the 308 Indian films**, a quarter of this library.
**Decision.** Keep the role, and never present its results without its coverage figure and the regional
shortfall.
**Why.** A 49%-covered role sitting beside a 96%-covered one, unmarked, implies a confidence that does
not exist.
**Status.** Active.

---

## Phase: the evaluation harness — 2026-07-26

### D68 — Two metrics: a gate metric and a product metric
**Context.** F39 — `director_only` already reaches P@20 = 0.950 at threshold 4.0, one film from the
ceiling, with each film worth 0.05.
**Decision.** Gate metric is **Precision@100 at ≥4.5** (31 films of headroom, 0.01 granularity, base
rate 0.300). Product metric stays **Precision@20 at ≥4.0** and is reported beside it.
**Why.** A saturated metric cannot adjudicate the go/no-go, and the number a user experiences is not
necessarily the number that can measure a model.
**Crucially: chosen before the crew model existed.** Picking k after seeing crew results would be
indistinguishable from picking the k that flattered them. `--grid` prints the full table so the choice
can be checked.
**Status.** Active. Supersedes the single-metric decision in D11, which stands in spirit — rank
metrics over MAE — but was underspecified on k and threshold.

### D69 — The go/no-go is reported against both director-only and the best baseline
**Context.** F40 — `genre_only` beats `director_only` on the gate metric on both splits.
**Decision.** Report two comparisons: crew versus `director_only` (the thesis) and crew versus the best
baseline, currently `genre_only` (usefulness).
**Why.** The spec treated rung 4 as the bar; it is not the highest rung for this account. Beating the
director while losing to genre would be a hollow result, and publishing the first without the second
would mislead.
**Status.** Active. Refines the gate in `Ariadne.MD` §7.

### D70 — Baselines get the strongest version of themselves
**Decision.** The popularity baseline fits its own least-squares mapping from TMDB's 0–10 scale rather
than assuming half. Genre and director baselines use the same shrinkage the real model will.
**Why.** A baseline built as a strawman proves nothing when beaten. The ladder is only evidence if each
rung is the best form of its idea.
**Status.** Active.

### D71 — Never compare a computed aggregate to zero exactly
**Context.** F42 — `np.full(100, 3.342).std()` is 1.3e-15, so a `std() == 0.0` guard never fired and
NaN reached Postgres, which rejected the run.
**Decision.** Test constancy with `min == max`. Pass every float through a finiteness check before it
enters a JSONB payload.
**Why.** JSON has no NaN, so one non-finite value makes an entire run unwritable. Losing a measurement
to a formatting detail is not acceptable.
**Status.** Active, regression-tested.
