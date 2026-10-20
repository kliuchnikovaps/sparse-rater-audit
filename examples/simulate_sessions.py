"""Generate a synthetic session table in the expected format (no real ratings are distributed).

The design mimics a sparse campaign: skewed workloads (many raters with one session), few questions
rated by more than one rater, and one rater whose ratings are shifted down by `--shift` scale points.

    python examples/simulate_sessions.py --out examples/synthetic_sessions.csv
"""
import argparse

import numpy as np
import pandas as pd


def simulate(n_raters=40, n_questions=600, shift=-1.0, seed=7):
    rng = np.random.default_rng(seed)
    workload = np.maximum(1, np.round(rng.pareto(1.2, n_raters) * 3)).astype(int)
    workload[0] = max(workload.max(), 40)                       # R01: the heaviest rater, shifted
    raters = [f"R{i + 1:02d}" for i in range(n_raters)]
    quality = rng.normal(-0.4, 1.0, (n_questions, 2))           # latent quality of the two outputs
    rows = []
    for r, (rid, n) in enumerate(zip(raters, workload)):
        for q in rng.choice(n_questions, size=min(n, n_questions), replace=False):
            delta = shift if r == 0 else 0.0
            row = {"rater_id": rid, "question_id": f"q{q:03d}",
                   "answer_L": f"answer {q}-a", "answer_R": f"answer {q}-b"}
            for side, mu in zip("LR", quality[q]):
                for d in "CUS":
                    row[f"{d}_{side}"] = int(np.clip(np.round(mu + delta + rng.normal(0, 0.8)), -2, 2))
            diff = (row["C_R"] + row["U_R"]) - (row["C_L"] + row["U_L"]) + rng.normal(0, 1.0)
            row["pref"] = int(np.clip(np.round(diff / 2), -2, 2))
            rows.append(row)
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "session_id", np.arange(1, len(df) + 1))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/synthetic_sessions.csv")
    ap.add_argument("--shift", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    df = simulate(shift=a.shift, seed=a.seed)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} sessions, {df.rater_id.nunique()} raters, {df.question_id.nunique()} questions -> {a.out}")
