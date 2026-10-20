"""
Core of the audit-first procedure.

Input: one row per session. In a session, a rater sees one question and two outputs (left and right),
rates each output on three five-point scales from -2 to +2 (C = correct, U = useful, S = consistent),
and states a preference (slider from -2 = strongly left to +2 = strongly right).

Screens (flags prompt human review; they never change a rating):
  M1  severity: z-screen of rater means (|z| > 1.5) and a session-count-matched permutation test with
      Benjamini-Hochberg adjustment across raters;
  M2  cross-dimensional rating profiles (P1: C <= -1 and U >= +1; P2: C <= -1, U <= -1 and S >= +1);
  M3  positional preference: exact two-sided binomial test on the directional preferences.
Diagnostics: connected components of the rater-candidate graph.
Counterfactual (sensitivity analysis only): mean-centering of M1-flagged raters with enough sessions.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest

DIMS = ("C", "U", "S")
SIDES = ("L", "R")
WEIGHTS = {"C": 0.40, "U": 0.35, "S": 0.25}
RATING_COLS = [f"{d}_{s}" for d in DIMS for s in SIDES]
REQUIRED = ["rater_id", "question_id", *RATING_COLS, "pref"]


# --------------------------------------------------------------------------- input
def load_sessions(path_or_df, drop_repeats: bool = True) -> pd.DataFrame:
    """Read and validate the session table.

    Required columns: rater_id, question_id, C_L, U_L, S_L, C_R, U_R, S_R, pref (slider in [-2, 2]).
    Optional: answer_L / answer_R (output text; identical texts of a question are one candidate).
    If a rater rated the same question more than once, the last submission is kept.
    """
    df = pd.read_csv(path_or_df) if isinstance(path_or_df, str) else path_or_df.copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df["rater_id"] = df["rater_id"].astype(str)
    df["question_id"] = df["question_id"].astype(str)
    if not df[RATING_COLS].isin(range(-2, 3)).all().all():
        raise ValueError("ratings must be integers in [-2, 2]")
    if drop_repeats:
        df = df.drop_duplicates(["rater_id", "question_id"], keep="last")
    return df.reset_index(drop=True)


def _candidate_ids(df: pd.DataFrame, side: str) -> pd.Series:
    if f"answer_{side}" in df.columns:
        tag = df[f"answer_{side}"].map(lambda t: hashlib.sha1(str(t).strip().encode()).hexdigest()[:12])
    else:
        tag = pd.Series(side, index=df.index)
    return df["question_id"] + "|" + tag


def _session_means(df: pd.DataFrame) -> np.ndarray:
    return df[RATING_COLS].to_numpy(float).mean(axis=1)


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    """BH-adjusted q-values, returned in the order of the input."""
    order = p.sort_values()
    m = len(order)
    q = (order * m / np.arange(1, m + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    return q.reindex(p.index)


# --------------------------------------------------------------------------- M1
def severity_screen(df: pd.DataFrame, z_threshold: float = 1.5) -> pd.DataFrame:
    """Per-rater mean over all six ratings, standardized across raters (equal weight per rater)."""
    sm = pd.Series(_session_means(df)).groupby(df["rater_id"].to_numpy())
    out = pd.DataFrame({"n_sessions": sm.size(), "mean": sm.mean()})
    out["z"] = (out["mean"] - out["mean"].mean()) / out["mean"].std(ddof=0)
    out["delta"] = out["mean"] - df[RATING_COLS].to_numpy(float).mean()   # offset from the all-ratings mean
    out["m1_screen"] = out["z"].abs() > z_threshold
    return out


def severity_permutation_test(df: pd.DataFrame, n_draws: int = 5000, n_perm: int = 1000,
                              z_threshold: float = 1.5, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Conditional permutation tests under exchangeable rater labels.

    Per rater: n_i sessions are drawn without replacement from all sessions; the statistic is
    |mean - grand mean|; p = (b + 1) / (B + 1); BH across raters.
    Corpus level: rater labels are permuted across sessions and the number of screening flags recounted.
    """
    rng = np.random.default_rng(seed)
    x = _session_means(df)
    grand = df[RATING_COLS].to_numpy(float).mean()
    stats = severity_screen(df, z_threshold)
    N = len(x)
    p = {}
    for r, row in stats.iterrows():
        n = int(row.n_sessions)
        idx = np.argsort(rng.random((n_draws, N)), axis=1)[:, :n]
        null = np.abs(x[idx].mean(axis=1) - grand)
        p[r] = (1 + np.sum(null >= abs(row["mean"] - grand) - 1e-12)) / (n_draws + 1)
    stats["p_perm"] = pd.Series(p)
    stats["q_bh"] = benjamini_hochberg(stats["p_perm"])
    labels = df["rater_id"].to_numpy()
    counts = []
    for _ in range(n_perm):
        m = pd.Series(x).groupby(rng.permutation(labels)).mean()
        counts.append(int((((m - m.mean()) / m.std(ddof=0)).abs() > z_threshold).sum()))
    counts = np.array(counts)
    observed = int(stats["m1_screen"].sum())
    summary = {"observed_flags": observed, "perm_mean": float(counts.mean()),
               "perm_95": [float(np.percentile(counts, 2.5)), float(np.percentile(counts, 97.5))],
               "p_flag_count": float((1 + np.sum(counts >= observed)) / (n_perm + 1))}
    return stats, summary


# --------------------------------------------------------------------------- M2
def profile_screen(df: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """Share of rated candidates matching the profiles P1 and P2 (score = profiles matched / 2)."""
    scores = []
    for s in SIDES:
        C, U, S = (df[f"{d}_{s}"] for d in DIMS)
        p1 = (C <= -1) & (U >= 1)
        p2 = (C <= -1) & (U <= -1) & (S >= 1)
        scores.append(pd.DataFrame({"rater_id": df["rater_id"], "s": (p1.astype(int) + p2.astype(int)) / 2}))
    out = pd.concat(scores).groupby("rater_id")["s"].mean().to_frame("profile_score")
    out["m2_review"] = out["profile_score"] > threshold
    return out


# --------------------------------------------------------------------------- M3
def positional_screen(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Exact two-sided binomial test of left vs. right preferences (sessions at 0 are excluded)."""
    rows = {}
    for r, g in df.groupby("rater_id"):
        n_left, n_right = int((g["pref"] < 0).sum()), int((g["pref"] > 0).sum())
        p = binomtest(n_left, n_left + n_right, 0.5).pvalue if n_left + n_right else np.nan
        rows[r] = {"n_left": n_left, "n_right": n_right, "p_binom": p}
    out = pd.DataFrame(rows).T.astype({"n_left": int, "n_right": int, "p_binom": float})
    tested = out["p_binom"].notna()
    out["q_binom"] = np.nan
    out.loc[tested, "q_binom"] = benjamini_hochberg(out.loc[tested, "p_binom"])
    out["m3_screen"] = out["p_binom"] < alpha
    return out


# --------------------------------------------------------------------------- design
def design_connectivity(df: pd.DataFrame) -> dict:
    """Connected components of the bipartite rater-candidate graph (union-find)."""
    parent: dict = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s in SIDES:
        for r, c in zip(df["rater_id"], _candidate_ids(df, s)):
            parent[find("r:" + r)] = find("c:" + c)
    comp = pd.Series({node: find(node) for node in parent})
    raters = comp[comp.index.str.startswith("r:")]
    sizes = raters.value_counts()
    largest = sizes.index[0]
    in_largest = {k[2:] for k in raters[raters == largest].index}
    alone = int((sizes == 1).sum())
    return {"components": int(comp.nunique()), "largest_raters": int(sizes.iloc[0]),
            "largest_candidates": int(((comp == largest) & comp.index.str.startswith("c:")).sum()),
            "largest_share_of_sessions": float(df["rater_id"].isin(in_largest).mean()),
            "raters_sharing_no_candidate": alone, "raters_in_largest": sorted(in_largest)}


# --------------------------------------------------------------------------- scores and counterfactual
def candidate_scores(df: pd.DataFrame, offsets: pd.Series | None = None, lam: float = 0.0) -> pd.Series:
    """Q_c: mean over rating events of sum_d w_d (m_d + 2) / 4, after subtracting lam * offset of the rater."""
    shift = lam * df["rater_id"].map(offsets if offsets is not None else {}).fillna(0.0).to_numpy()
    parts = []
    for s in SIDES:
        q = sum(WEIGHTS[d] * (df[f"{d}_{s}"].to_numpy(float) - shift + 2) / 4 for d in DIMS)
        parts.append(pd.DataFrame({"candidate": _candidate_ids(df, s).to_numpy(), "q": q}))
    return pd.concat(parts).groupby("candidate")["q"].mean()


@dataclass
class CorrectionResult:
    corrected_raters: list
    table: pd.DataFrame          # one row per lambda
    changed: pd.DataFrame        # candidates whose verdict changes at lambda = 1


def counterfactual_correction(df: pd.DataFrame, flags: pd.DataFrame, min_sessions: int = 3,
                              lambdas=(0, 0.25, 0.5, 0.75, 1.0), threshold: float = 0.5) -> CorrectionResult:
    """Mean-centering of M1-screen-flagged raters with >= min_sessions sessions.

    This is a counterfactual: reported scores always come from the original ratings.
    """
    sel = flags[flags["m1_screen"] & (flags["n_sessions"] >= min_sessions)]
    offsets = sel["delta"]
    base = candidate_scores(df)
    rows = []
    for lam in lambdas:
        q = candidate_scores(df, offsets, lam)
        rows.append({"lambda": lam, "mean_q": q.mean(), "above_threshold": int((q > threshold).sum()),
                     "changed": int(((q > threshold) != (base > threshold)).sum())})
    q1 = candidate_scores(df, offsets, 1.0)
    changed = pd.DataFrame({"q_original": base, "q_corrected": q1})
    changed = changed[(q1 > threshold) != (base > threshold)]
    return CorrectionResult(list(sel.index), pd.DataFrame(rows), changed)


# --------------------------------------------------------------------------- one call
def run_audit(df: pd.DataFrame, seed: int = 0, n_draws: int = 5000, n_perm: int = 1000) -> dict:
    m1, m1_summary = severity_permutation_test(df, n_draws=n_draws, n_perm=n_perm, seed=seed)
    flags = m1.join(profile_screen(df)).join(positional_screen(df))
    corr = counterfactual_correction(df, flags)
    return {"flags": flags, "m1_flag_count": m1_summary, "design": design_connectivity(df),
            "correction": corr}
