# sparse-rater-audit

Audit-first screening of rater effects in **sparse, multi-dimensional human evaluations** of model outputs.

In a sparse rating campaign most outputs are rated by one person, so an unusually low or high rater
mean can reflect either the rater (severity, leniency) or the outputs that rater happened to receive.
This code reports **flags for human review**. It never changes a rating. Score correction is computed
only as a counterfactual sensitivity analysis.

This is a minimal implementation of the procedure described in:

> P. Kliuchnikova, S. V. Kovalchuk. *A Bias-Auditing Procedure for Sparse, Multi-Dimensional Human
> Evaluations of Code-Generation Model Outputs: A Pilot Study.* Manuscript submitted to *Algorithms* (MDPI), 2026.

The human ratings analysed in the paper are confidential and are **not** included. The repository
ships a simulator that produces data in the same format.

## What it computes

| Step | What it does | Output |
|---|---|---|
| **M1: severity** | Rater mean over all ratings, standardized across raters (\|z\| > 1.5). A session-count-matched permutation test per rater, with Benjamini–Hochberg (BH) adjustment. The flag count is compared with its permutation distribution. | `z`, `p_perm`, `q_bh`, `m1_screen` |
| **M2: rating profiles** | Share of rated outputs matching P1 (C ≤ −1 and U ≥ +1) or P2 (C ≤ −1, U ≤ −1 and S ≥ +1). Threshold 0.10. | `profile_score`, `m2_review` |
| **M3: position** | Exact two-sided binomial test of left vs. right preferences. Ties at 0 are excluded. | `p_binom`, `q_binom`, `m3_screen` |
| **Design connectivity** | Connected components of the bipartite rater–output graph. Rater effects are comparable only within a component. | `design` |
| **Counterfactual** | Mean-centering of M1-flagged raters with ≥ 3 sessions, for λ ∈ {0, 0.25, 0.5, 0.75, 1}. Shows how many above/below-threshold verdicts would change. | `counterfactual_lambda.csv`, `changed_verdicts.csv` |

The candidate score is Q = mean over ratings of Σ w_d (m_d + 2)/4, with weights C = 0.40, U = 0.35 and S = 0.25. The verdict threshold is Q > 0.5.

All permutation tests are **conditional on exchangeable assignment** of sessions to raters. If raters
received systematically different outputs, a flag does not identify severity. That is why flags are
meant to prompt review, not automatic correction.

## Install

```bash
git clone https://github.com/kliuchnikovaps/sparse-rater-audit.git
cd sparse-rater-audit
pip install -r requirements.txt
```

Python ≥ 3.9.

## Quick start

```bash
python examples/simulate_sessions.py --out examples/synthetic_sessions.csv   # synthetic data; rater R01 is shifted by -1
python -m rater_audit examples/synthetic_sessions.csv --out results/
```

This writes `results/rater_flags.csv`, `results/summary.json`, `results/counterfactual_lambda.csv` and
`results/changed_verdicts.csv`. On this synthetic file (default seed), the per-rater test flags the shifted
rater R01 after BH adjustment, while the z-screen does not. The two M1 statistics are centred
differently (rater-weighted vs. session-weighted), so they can disagree.

From Python:

```python
from rater_audit import load_sessions, run_audit

df = load_sessions("examples/synthetic_sessions.csv")
res = run_audit(df, seed=0)
print(res["flags"].query("q_bh < 0.05"))
print(res["design"]["components"], res["correction"].table)
```

## Input format

One row per session: one rater, one question, two outputs shown side by side.

| Column | Meaning |
|---|---|
| `rater_id` | Pseudonymous rater identifier |
| `question_id` | Question (prompt) identifier |
| `C_L`, `U_L`, `S_L`, `C_R`, `U_R`, `S_R` | Ratings of the left and right output, integers in [−2, 2]: correct, useful, consistent |
| `pref` | Preference slider in [−2, 2]: negative = left, positive = right, 0 = no preference |
| `answer_L`, `answer_R` (optional) | Output texts. Identical texts within a question are treated as one candidate. |

If a rater rated the same question more than once, the last submission is kept.

## Tests

```bash
pip install pytest
python -m pytest -q
```

## Repository layout

```
rater_audit/
  audit.py      screens M1-M3, design connectivity, candidate scores, counterfactual correction
  __main__.py   command-line interface
examples/
  simulate_sessions.py     synthetic sparse campaign in the expected format
tests/
  test_audit.py
```

## License

MIT (see `LICENSE`).
