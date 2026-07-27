# Can you find your favourite editor from your Letterboxd ratings?

**Stage 1 result, single account, 1,345 rated films. 2026-07-27.**

---

## The short version

I could not name my favourite editor, cinematographer, composer or production designer. Almost nobody
can. So I tried to compute them from my own Letterboxd export, and to measure how much data the
question actually needs.

**What I found:**

1. **There is real signal in below-the-line crew.** Seven of eleven roles produce effects larger than
   noise generates. Exactly **one survives correction for testing eleven roles at once**: Aarti Bajaj,
   editor, p = 0.002.
2. **That same finding is mostly her director's.** With directors competing for the same variance,
   Bajaj's effect keeps only **24%** of its size. She edits Anurag Kashyap's films. "Favourite editor"
   was largely "favourite director, seen through his editor."
3. **Whether crew beats directors is unresolved, and one library cannot resolve it.** Crew leads
   director-only by +0.030 on the honest split, 95% CI **[−0.020, +0.100]**, and *trails* by 0.020 on
   the other. Inconclusive, not positive and not negative. **Four explanations I offered for that were
   each measured and each rejected** — including the sharpest number here: a gradient booster given no
   crew or cast features at all predicts as well as the crew model, and *worse* once you give it the
   crew features.
4. **Asked instead how much of a rating each kind of information explains, crew is the only one that
   clears zero** — and the answer is dominated by something else entirely. What the rest of the world
   thought of a film explains **28.3%** of the variation in my ratings on its own. Era, country, genre,
   director, cast and crew together add **3.3 points** on top of that. Caveat attached below: this
   metric was not pre-registered, and I chose it after the gate metric came out null.
5. **A half-star crew preference is undetectable at this sample size** — at any film count this library
   contains. That number, as far as I can tell, has not been published before.
6. **My own strongest signal is a dislike**, not a favourite: every one of my five largest composer
   effects is a Hollywood blockbuster scorer, all negative.

The honest headline is not "crew beats directors." It is: **here is exactly how much data it takes to
know your favourite editor, and 1,345 films is not enough.**

The uncomfortable one, which I did not expect to be writing: measured this way, my library looks mostly
like a well-calibrated version of general consensus with a thin personal layer on top. Roughly
three-quarters of the variation in my ratings is not explained by anything I can compute from film
metadata at all.

---

## Why this is harder than a `GROUP BY`

Averaging your ratings per crew member fails in three ways, and each one changed a result here.

**Consensus carries no information about you.** A 5-star rating for *The Godfather* is what nearly
everyone gives it. A 5-star rating for something the world is lukewarm on is dense with signal. So
nothing is fitted on raw ratings — everything is fitted on the **residual** against what the film's
context predicts you would give it. My 5.0 for *Gangs of Wasseypur — Part 1* is +2.014 above
expectation; my 5.0 for *The Godfather* is worth almost nothing.

**The expectation model has to be good, or its failures become crew effects.** My first version used
TMDB's average rating alone. It left a **+0.744-star mean residual on Japanese films** — larger than any
crew effect I went on to report. Every crew member working predominantly in one national cinema was
being credited with the model's inability to predict that cinema. Adding vote count, country, decade
and genre brought Japan to +0.102 and **shrank every headline effect by 18–75%.**

**Crews are sticky, and some collaborators are mathematically inseparable.** Christopher Nolan wrote 13
films in my library and directed all 13. No arithmetic separates his writing from his direction, so the
model says exactly that instead of picking one.

---

## Method

Plain tools, deliberately. Two flavours of linear regression and one hand-written estimator.

| step | what |
|---|---|
| Expectation | Ridge over vote_average, log(vote_count), country, decade, genre |
| Track 1 | Empirical Bayes shrinkage of each person's mean residual, `n/(n+k)`, with `k` from a one-way random-effects estimator |
| Track 2 | Ridge over a sparse 1,297 × ~2,000 film-by-person matrix, fitting everyone jointly |
| Significance | Permutation null over the **maximum** effect per shuffle, 400 permutations |
| Comparison | Paired bootstrap, 2,000 resamples, on the difference between predictors |
| Decomposition | The same paired bootstrap, 1,000 resamples, on the difference in variance explained when one kind of information is added to a consensus base. Each block is centred separately, as the split specification requires — without it a model is charged for a mean shift between regimes that it could not have known about |

No neural network. With ~1,300 rows and almost entirely sparse binary features, a network would have
more parameters than films. The binding constraint is information, not model capacity: at 12 films per
person the standard error is ±0.23 stars regardless of architecture.

**Data:** 1,345 rated films → 1,297 resolved to TMDB (99.0% of the 1,309 that are actually films; 36
entries are television). 157,451 crew credits, 77,037 distinct people. Resolution precision was
hand-audited at **127/127** over a stratified sample, with every non-exact match audited exhaustively.

---

## Result 1: the gate is inconclusive

Six predictors, temporal split (train on films logged before 2024, test after), scored on Precision@100
at ≥4.5 stars:

| predictor | gate | Spearman |
|---|---|---|
| global mean | 0.300 | 0.000 |
| popularity | 0.700 | 0.655 |
| context (popularity + country + decade + genre) | 0.660 | 0.632 |
| **director only** | **0.680** | 0.634 |
| **crew, Track 1** | **0.710** | **0.651** |
| crew, Track 2 (ridge) | 0.680 | 0.644 |

Paired bootstrap on the differences:

| comparison | diff | 95% CI | verdict |
|---|---|---|---|
| crew vs director-only | +0.030 | [−0.020, +0.100] | within noise |
| crew vs context | +0.050 | [−0.010, +0.090] | within noise |
| crew vs director-only (random split) | **−0.020** | [−0.090, +0.050] | within noise |

**Every interval includes zero.** Crew has the best rank correlation on both splits and leads the
director on one, but a five-film difference at k=100 on a 527-film test set is about one standard error.

I am reporting this as inconclusive rather than picking the flattering reading. The metric configuration
was fixed **before the crew model existed**, from a measured grid, because the obvious metric was
already saturated: director-only reaches Precision@20 = 0.950, one film from the ceiling, which cannot
resolve an improvement.

**What would settle it:** the interval half-width scales as 1/√n, so a bigger single test set is the
wrong instrument. Twenty accounts each producing this same weak directional result would settle it by
sign test — **15 of 20 favouring crew gives p ≈ 0.02.** Twenty individually inconclusive results are
collectively conclusive.

### Four explanations I offered for that, and measured

Each of these was my own account of why crew prediction was weak. All four were tested and all four were
rejected, in this order:

| # | explanation | verdict |
|---|---|---|
| 1 | too few roles — five is not enough | rejected: twelve roles score no better than five |
| 2 | cast is missing, and actors recur far more than editors | rejected: actor effects are the same size, and adding them *lowered* the gate 0.710 → 0.680 |
| 3 | the arithmetic dilutes — averaging across roles pulls every adjustment toward zero | rejected: twelve combination strategies span 0.660–0.710 and the two splits disagree about which is best |
| 4 | the model is too rigid — a linear model cannot express the pattern | rejected: gradient boosting is within noise on every gate comparison |

The sharpest number in this project sits inside that last null. **A gradient booster given no crew or cast
features at all scores 0.720, against 0.710 for the crew model — and adding the crew features drops it to
0.650.** A flexible model handed columns describing who made the film declines to use them, and is worse
for having been offered them.

That is a stronger statement than "the effects are small." It says that for this library, who worked on a
film carries almost no *predictive* information about how I will rate it, measured four independent ways.
Which is what makes Result 4 worth stating carefully rather than triumphantly: the one place who-made-it
shows up as more than nothing is a +0.040 effect on a metric I chose after the fact.

---

## Result 2: the detection floor

There is no ground truth for "favourite editor," so I manufactured one: a synthetic viewer whose ratings
are consensus expectation plus a known bonus whenever a chosen person is credited, plus noise matched to
the real residual spread and **quantised to half-stars**, because a floor measured on a smooth scale
would be dishonest about a scale that is 72% whole stars.

Films needed to recover a planted effect into a role's top 3, in ≥80% of trials:

| effect | editor | composer | cinematographer | writer |
|---|---|---|---|---|
| +0.25★ | never | never | never | never |
| **+0.50★** | **never** | **never** | **never** | **never** |
| +0.75★ | 12 | never | never | never |
| +1.00★ | 12 | 12 | 8 | 12 |
| +1.50★ | 8 | 8 | 5 | 5 |

**A half-star preference cannot be recovered at any film count this library contains.** Only 3 editors,
1 cinematographer and 4 writers reach 12 films; composer alone reaches 28.

This is the number I would most want someone else to reuse. It says the honest reporting threshold is 12
films, which leaves most roles with one to four names and the insufficient-data bucket as the majority of
every list.

**Significance and power are different questions and I report both.** The permutation null says Bajaj's
+0.390 is beyond what noise produces (p = 0.002). The floor says a procedure looking for an effect that
size would find it roughly half the time. Both are true: **the effect is real, and the method that found
it would have missed it on a different sample of my films.**

---

## Result 3: what it actually found

Eleven roles, 400-permutation null each. With eleven tests the Bonferroni threshold is 0.0045:

| role | person | films | effect | p | clears 0.0045 |
|---|---|---|---|---|---|
| **editor** | **Aarti Bajaj** | 12 | **+0.390** | **0.002** | **yes** |
| playback singer | Shilpa Rao | 12 | −0.463 | 0.005 | borderline |
| writer | Sajid Samji | 12 | −0.332 | 0.007 | no |
| cinematographer | Roger Deakins | 13 | +0.168 | 0.022 | no |
| composer | Brian Tyler | 16 | −0.297 | 0.027 | no |
| supervising sound editor | John A. Larsen | 12 | −0.297 | 0.027 | no |
| sound designer | Randy Thom | 24 | −0.269 | 0.035 | no |
| casting | Francine Maisler | 39 | +0.198 | 0.080 | no |
| production design | Stuart Craig | 12 | −0.028 | 0.249 | no |
| costume design | Manish Malhotra | 20 | 0.000 | 0.441 | no |
| choreography | Raju Khan | 13 | 0.000 | 0.444 | no |

Costume design and choreography return **exactly zero** despite large raw means — the estimator finds no
between-person variance and collapses everything, which is the machinery working rather than failing.

### Then the attribution, which undercuts the headline

Refitting with directors as competing features asks whether an effect is the person's own:

| person | role | Track 1 | with directors | kept | |
|---|---|---|---|---|---|
| **Aarti Bajaj** | editor | +0.390 | +0.093 | **0.24** | **absorbed** |
| Roger Deakins | cinematographer | +0.168 | +0.058 | 0.35 | absorbed |
| Randy Thom | sound designer | −0.269 | −0.157 | 0.58 | **survives** |
| Pritam Chakraborty | composer | −0.136 | −0.019 | 0.23 | absorbed |

**15 of 80 effects survive.** The single most robust crew finding in my library is also the one most
attributable to somebody else. That tension is the honest summary of the whole exercise, and about half
the time the answer to "who is your favourite editor" is "your favourite director, via their editor."

### My strongest signal is a dislike

All five of my largest composer effects are negative: Brian Tyler, Danny Elfman, Henry Jackman, John
Powell, John Williams. Every one a prolific Hollywood blockbuster scorer.

I rate big studio films below consensus, and it surfaces through composers because blockbuster composers
are the best-sampled crew in any library. A feature called "your favourite editor" would bury its own
best-evidenced finding for being unflattering.

---

## Result 4: what actually explains a rating

The gate metric asks whether a model would recommend well. A different and more basic question is how
much of a rating each kind of information accounts for at all. So: six kinds of information, each added
**on its own** to the same starting point — what the rest of the world thought of the film, meaning
TMDB's average and its vote count — with a 1,000-resample paired bootstrap on every difference.

Added one at a time rather than stacked, because stacking makes each number depend on the order the
layers happen to be listed in, and that order is a design choice, not a result.

**Temporal split, 527 held-out films. Consensus alone explains 28.3% of the variation.**

| what it knows | explains | 95% CI |
|---|---|---|
| when it was made | +0.020 | [+0.004, +0.037] |
| where it comes from | −0.022 | [−0.042, −0.002] |
| what kind of film it is | +0.031 | [+0.007, +0.056] |
| who directed it | +0.013 | [−0.031, +0.055] |
| who acted in it | +0.022 | [−0.012, +0.058] |
| **who else made it (below-the-line crew)** | **+0.040** | **[+0.003, +0.077]** |
| all six together | +0.033 | [−0.003, +0.071] |

Four things in that table.

**Consensus is the story.** 28.3% from what everyone else thought, against at most 4 points from anything
else. The single most predictive fact about how I will rate a film is how the rest of the world rated it.
That is not the finding a taste-analysis project hopes for, and it is the finding.

**Neither the director nor the cast clears zero, and crew does.** Era and genre also clear zero, at
+0.020 and +0.031, so crew is not the only surviving layer — it is the only *person* layer that survives,
which is the Stage 1 thesis arriving through a different door after the gate metric refused to settle it.

**All six together (+0.033) is no better than crew alone (+0.040).** Adding the director, the cast and
every context feature to the crew model buys nothing measurable.

**The layers overlap heavily.** Measured one at a time they sum to +0.104; delivered together they are
+0.033. So **68% of the marginal contributions are the same information counted twice** — a director
carries their era, a genre carries its decade, a cast carries its director. This is why these are bars
against a common base and never segments of a whole. "Your taste is 30% director" would overstate the
total threefold.

### The disclosure that belongs with this result

**This metric was not pre-registered, and I selected it after the gate metric came out null.** No gate
interval for any layer clears zero on either split. Had I reported only the pre-registered metric, Result
4 would read "nothing is distinguishable from noise."

That is metric-shopping unless the reason is measured and stated, so here is the reason. Across all eight
subsets of the context features, variance explained rises with features while the gate wanders with no
relationship to it:

| features | gate | explained |
|---|---|---|
| vote_average only | 0.700 | +0.179 |
| consensus (+ vote_count) | 0.680 | +0.216 |
| + genre | 0.690 | +0.240 |
| + decade + genre | **0.640** | **+0.250** |
| + all three | 0.660 | +0.246 |

The best-explaining subset is the second-worst ranker. At 527 test films each film moves the gate by
0.010, and six small differences all sit inside that noise band — the gate was chosen to resolve one large
difference for the go/no-go, and it cannot resolve six small ones.

I still would not call Result 4 a confirmation of the thesis. It is a +0.040 effect with a lower bound of
+0.003, in one library, on a metric chosen after seeing that the pre-registered one was silent. Treat it
as the most promising thing to pre-register for Stage 2, not as a settled answer.

### The same analysis on the leaky split, as a warning

| layer | temporal | random | ratio |
|---|---|---|---|
| who directed it | +0.022 | **+0.094** | 4.3x |
| who acted in it | +0.029 | +0.071 | 2.4x |
| who else made it | +0.040 | +0.088 | 2.2x |
| when it was made | +0.014 | +0.031 | 2.2x |

Every layer involving a person more than doubles. The director layer quadruples and moves from an interval
containing zero to **[+0.049, +0.136]** — a confident director effect, on the same data, split the
convenient way. Five of six layers "clear zero" on the random split; one does on the temporal split.

This is the clearest argument I have for why the temporal split is the headline everywhere in this
project. The same instrument, given data split the easy way, reports a result that does not survive being
asked to predict forward.

---

## Pre-registration, including what I got wrong

Predictions were committed to git **before any crew effect was computed**.

| # | prediction | outcome |
|---|---|---|
| baseline | could not name my editor, cinematographer, composer or production designer | **the premise, and the thing the tool is measured against** |
| 2a | Kurosawa, Kore-eda, Kubrick top the director list | Kurosawa 11 films, Kubrick 9, Kore-eda 5 — below the floor, untestable |
| 2c | top crew names will be strangers | **correct** — Bajaj, Tyler, Samji, Rao |
| **2d** | **95% confident crew beats director *clearly* (>0.74)** | **wrong.** 0.710, and within noise |
| **2e** | **90% confident crew beats the best baseline** | **wrong** on the original comparison |
| 2f | cinematographer or composer strongest | **correct** — composer largest by magnitude |

A second pre-registration, for the expanded role list, predicted attribution survival would **rise**
above 42% because casting and sound roles move between directors more freely than editors. **It fell to
19%.** The reasoning was backwards: Indian playback singers and sound designers work within recurring
composer-director teams, making them *more* director-concentrated than editors, not less.

Two confident predictions missed, and one piece of reasoning was inverted. That is the point of writing
them down first.

---

## Errors caught, because they are part of the result

Five that would have changed conclusions:

1. **The shuffle test caught a shrinkage bug.** Effects fitted on *randomly permuted* ratings were
   almost as large as real ones — 1.111 against 1.147, with 37 of 44 findings reproducible from noise.
   The shrinkage constant was ~3 instead of ~56 under the null. **This would have invalidated every crew
   finding**, and only the negative control could have caught it.
2. **My own pass criterion was too weak to notice**, since it only asked whether shuffled was below real.
3. **Eight wrong film matches**, all sequels resolved to their siblings — *Back to the Future Part III*
   to *Part II* scores 0.963 on trigram similarity. Found by a structural check (two entries resolving to
   one film) rather than by the precision threshold, which those eight passed at 99.38%.
4. **The regression suite gave false confidence about the exact case it protected.** It contains
   *Whiplash* 2013 versus 2014 as its headline case and passed throughout, while production resolved the
   short to the feature — because the bug lived on a code path the fixtures could not express.
5. **A feature I added leaked split membership.** Diary coverage is 16% of train and 69% of test, so it
   was close to a "this row is in the test set" indicator, and it moved the headline gate by 0.020.

And one that changed no published number but says something about how defects survive. A baseline rung
called `genre_only` was, I eventually measured, **exactly identical to its neighbour** — 0.0000 maximum
difference across all 527 held-out films. Fitting genre effects on residuals from a model that already
used genre leaves nothing for the effects to explain. I had noticed this three days earlier and written it
down in a design note, accurately, as a passing remark — and then left the rung in the ladder. A
degeneracy documented and not fixed is a bug with paperwork.

Five of the six were caught by a check that exists to fail. The sixth was caught by finally computing a
number I had already described in prose.

---

## Limitations

- **n = 1.** Nothing here generalises to anyone else. It is one person's taste, measured carefully.
- **About three-quarters of the variation in my ratings is unexplained** by anything computable from film
  metadata. Consensus reaches 28.3% and everything else adds 3.3 points. That residual is not evidence of
  a deeper unmeasured taste, and it is not noise either — it is simply outside what these features can see.
- **The "who else made it" bar bundles twelve roles while "who directed it" is one**, so the two are not
  comparable person-for-person. Crew gets twelve chances to find something. That asymmetry *is* the thesis
  comparison, deliberately, but it is not a fair fight between two individuals.
- **Result 4's metric was chosen after seeing the pre-registered one come out null.** The reason is
  measured and stated, but the ordering is the ordering.
- **Claims are scoped to post-2000 cinema.** 83% of the library postdates 2000; 5.9% predates 1980.
- **24% Indian cinema**, which shapes several findings — production design coverage is 49% for those
  films against 92% for US ones.
- **Some effects are permanently unidentifiable, not merely under-sampled.** If Bajaj never edited for
  another director, no quantity of data separates her from Kashyap. The information does not exist.
- **The rating scale is an information ceiling.** Half-stars, 72% whole, 222 films tied at 5.0. Rewatch
  counts help — the largest single improvement anything produced — but only on the 38% of films the diary
  covers, and only on the split that cannot judge it fairly.
- **Recommendations still smuggle directors in.** Most reasons are writers, and writers who direct their
  own films make "because of X (writer)" a director recommendation in disguise. Flagged in the output,
  counted correctly, not yet fixed.

---

## What would settle it

**Fifteen to twenty-five libraries, recruited for divergence rather than agreement.** Not because more
data would rescue the finding, but because it is the only instrument that can answer the question:

- The thesis needs **replication**, not a bigger single test. 15 of 20 accounts pointing the same way is
  p ≈ 0.02.
- Per-person depth needs **pooling**. A hierarchical model lets someone with 4 films by a composer borrow
  strength from someone with 12, which is the one genuine modelling upgrade available.
- The era and region skew needs **different people**, specifically ones deep in pre-1970 and
  non-Anglophone cinema, where my data is weakest.

And I would want to be clear-eyed that more data buys **certainty, not a positive answer.** If the true
effect is as small as this correction suggests, twenty libraries will measure "slightly, sometimes, for
some people" precisely. That is still a result, and it was decided in advance that it would be published
either way.

---

## Reproducing this

Everything is in the repository. `research/` holds the pre-registrations, this writeup, and the
hand-audit record; `docs/DATA_FINDINGS.MD` holds 79 numbered findings, each paired with the decision it
forced; `research/DECISIONS.md` holds 111 decisions in chronological, append-only order, including every
reversal with its original reasoning intact.

The commit history is the honest version: it contains the run where the headline result got *worse* after
a bias was fixed, and the four separate hypotheses I offered for why crew prediction was weak, each
measured and each rejected.
