"""Command line: python -m rater_audit sessions.csv --out results/"""
import argparse
import json
import os

from .audit import load_sessions, run_audit


def main():
    ap = argparse.ArgumentParser(description="Audit-first screening of rater effects.")
    ap.add_argument("csv", help="session table (see README for the format)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-draws", type=int, default=5000, help="null draws per rater (M1 per-rater test)")
    ap.add_argument("--n-perm", type=int, default=1000, help="label permutations (M1 flag count)")
    a = ap.parse_args()

    df = load_sessions(a.csv)
    res = run_audit(df, seed=a.seed, n_draws=a.n_draws, n_perm=a.n_perm)
    os.makedirs(a.out, exist_ok=True)
    res["flags"].to_csv(os.path.join(a.out, "rater_flags.csv"))
    res["correction"].table.to_csv(os.path.join(a.out, "counterfactual_lambda.csv"), index=False)
    res["correction"].changed.to_csv(os.path.join(a.out, "changed_verdicts.csv"))
    design = {k: v for k, v in res["design"].items() if k != "raters_in_largest"}
    summary = {"sessions": len(df), "raters": int(df.rater_id.nunique()), "design": design,
               "m1_flag_count": res["m1_flag_count"],
               "m1_bh_flags": res["flags"].index[res["flags"].q_bh < 0.05].tolist(),
               "m2_review": res["flags"].index[res["flags"].m2_review].tolist(),
               "m3_flags": res["flags"].index[res["flags"].m3_screen].tolist(),
               "corrected_raters (counterfactual)": res["correction"].corrected_raters,
               "changed_verdicts_at_lambda_1": len(res["correction"].changed)}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
