# Pre-registration — Stage 1 crew effects

**Status: FILLED IN AND COMMITTED, 2026-07-26 — before any crew effect was computed.**

Read as committed:

- **2f** — "cinematographer or composer or camera crew" is read as *cinematographer or composer*. Only
  six roles are modelled, and camera crew collapses into cinematographer.
- **2b** — the named films are `Vaastav` (1999), `3 Idiots` (2009), `Haider` (2014), `Ghajini` (2008),
  and the `John Wick` series (2014/2017/2019). All present and resolved, so the prediction is testable.
- **Testability of 2a** — Kurosawa has 11 films, Kubrick 9, Kore-eda **5**. If the detection floor lands
  at 8, Kore-eda will not be estimable and that third of the check becomes untestable. Noted here so it
  is not later read as a miss.
- **A confounding case the author named unprompted:** Anurag Kashyap appears as director on 9 films and
  as writer on the same 9. Those two effects are perfectly inseparable, which is exactly the case D9
  says to refuse rather than guess.

---

## Why this file exists, and what it is not

An earlier draft of this file asked which five crew people would top the list. That was the wrong
question. The author's honest answer was *"I have never thought about crew except directors"* — which is
not a failure to answer. **It is the premise of the project.** If a user could already name their
favourite editor, Ariadne would have nothing to tell them.

So this file records two things instead:

1. **The baseline of self-knowledge** the tool claims to improve on. "I cannot name my favourite editor"
   is a measurement, and it is the one the result will be judged against.
2. **Predictions that can actually be wrong** — about directors, about films, about the model's
   behaviour — so the run has something it can fail.

**The git commit timestamp is the proof.** Recorded afterwards, any reaction to the output is
unfalsifiable; recorded first, the same reaction is evidence.

---

## What was already known when this was written

Honest disclosure, because a pre-registration that overstates its own blindness is worthless.

**Already seen** — reported in the Phase 1.5 summary before this file existed, which was an error on the
assistant's part:

- Raw film counts: most-credited editors **Rameshwar S. Bhagat (14)** and **Aarti Bajaj (12)**;
  cinematographer **Roger Deakins (13)**; composers **Hans Zimmer (38)**, **Pritam Chakraborty (33)**,
  **John Williams (26)**.
- Coverage and sparsity (F33, F35), and the baseline ladder (F40): `genre_only` 0.740,
  `director_only` 0.690.

**Not computed by anyone yet:** any crew effect at all, which people survive shrinkage, which are
inseparable from a collaborator, the detection floor.

**Why the counts do not spoil this.** A film count is not an effect. Effects are fitted on residuals
against consensus expectation and then shrunk, so a prolific person with unremarkable residuals scores
near zero. Zimmer's 38 films say nothing about whether his involvement predicts *this user's* rating
above what the film would have earned anyway. The names are still contamination and the writeup must
say so.

---

## Part 1 — The baseline: what you know before the model tells you

Answer honestly. **"No idea" is the expected answer for most of these and is the point.**

| Role | Can you name yours, unaided? | If yes, who? |
|---|---|---|
| Director | `yes / no` | `yes, Akira Kurosawa___` |
| Editor | `yes / no` | `_No idea__` |
| Cinematographer | `yes / no` | `_No idea__` |
| Composer | `yes / no` | `__No Idea_` |
| Production designer | `yes / no` | `_No Idea__` |
| Writer | `yes / no` | `_Yes mostly in case of directors that write their own creenplays Anurag Kashyap__` |

**How many below-the-line crew members could you name at all, from any film?** `__Probably the director, producer and writer not mch other than that_`

**Have you ever chosen what to watch because of someone other than the director or cast?**
`_Maybe the writers__`

---

## Part 2 — Predictions that can be wrong

These are the falsifiable ones. They do not need crew knowledge.

### 2a. Directors you expect to top the director list

You do know these, which makes this the **sanity check the model can fail**. If it reports a favourite
director whose films you rated 2.0, something is broken upstream and no crew result should be trusted.

| # | Director | Why |
|---|---|---|
| 1 | `Akira Kurosawa___` | `_Love his style__` |
| 2 | `Hirokazu Koreeda___` | `___Love his stories` |
| 3 | `__Stanley Kubrick_` | `His films are unorthodox___` |

### 2b. Films you expect to drive the strongest effects

Films you rated highly *relative to how the world rates them* — the model learns from disagreement with
consensus, so these matter more than your favourites in the abstract.

`Probably Foreign Films mostly like Vaastav,3 Idiots, Hiader Ghajini, otherwise _maybe action films like john wick__`

### 2c. Will the top below-the-line names be people you recognise?

- [ ] Mostly yes — I'll know the names even if I couldn't have ranked them
- [ ] About half
- [1] Mostly no — they'll be strangers
- [ ] No idea

### 2d. Will below-the-line crew beat the director-only baseline on the gate metric?

The project's thesis. `director_only` scores **0.690**.

- [ 1] Yes, clearly (> 0.74)
- [ ] Yes, narrowly (0.70–0.74)
- [ ] No, about the same (0.66–0.70)
- [ ] No, worse

Confidence, 0–100%: `__95_`

### 2e. Will it beat `genre_only` at 0.740, the best baseline?

- [ 1] Yes    - [ ] No

Confidence, 0–100%: `_90__`

### 2f. Which role will show the strongest effect?

Not which role matters most in filmmaking — which one *this model on this data* will find. Composer has
far more statistical power than the rest (F37), which is a fact about sample size, not importance.

`_Cinematrographer or composer or camera crew__`

### 2g. Anything you expect to be surprised by

`_no idea yet__`

---

## Rules

1. **Do not edit anything above after the run.** Corrections go in a dated note at the bottom.
2. Append-only after the first commit, on the same principle as `DECISIONS.md`.
3. The Stage 1 writeup reports every prediction against its outcome — **especially the wrong ones.**

---

## Outcome

*Blank until after the first fit.*

| Prediction | Outcome | Note |
|---|---|---|
| | | |
