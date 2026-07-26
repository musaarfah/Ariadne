# research/

**This directory is tracked in git. `docs/` is not.**

`docs/` holds local working documents — the spec, the phase plan, the data findings, the status board.
`research/` holds what gets shared, and what has to be verifiable by someone who was not here.

## What lives here

| File | When it appears | Why |
|---|---|---|
| `DECISIONS.md` | From the start, appended to continuously | Chronological, append-only record of every decision and reversal. Lives here rather than `docs/` because `docs/` is untracked and has no git history, so overwriting a section there destroys the prior reasoning permanently. |
| `preregistration.md` | **Before the first real model run** (Phase 1.7) | Five crew people expected to top the list, committed with a git timestamp. The timestamp is the proof — it only counts if it precedes the run. |
| `metrics/` | From Phase 1.6 | Persisted evaluation runs, exported from `analysis_runs.metrics`. Never overwritten, so the writeup can show how numbers moved rather than only where they landed. |
| `writeup-stage1.md` | Phase 1.9 | **Hard gate.** Phase 2 does not start until this exists. Method, the full baseline ladder, negative controls, detection floor, results, limitations, claims scoped to post-2000 cinema. |
| `plots/` | Phase 1.6 onward | Calibration curves, effect distributions, detection-floor sweeps. |

## Rules

- Nothing here gets rewritten to look better in hindsight. Corrections are additions.
- Every reported metric carries the baseline ladder alongside it. An absolute number on its own is
  incomplete.
- A negative result gets published. That was decided in advance, before any data was fit.
