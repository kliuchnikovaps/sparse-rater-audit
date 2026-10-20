import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from rater_audit import (benjamini_hochberg, candidate_scores, load_sessions, positional_screen,
                         profile_screen, run_audit)
from simulate_sessions import simulate


def test_bh_matches_definition():
    p = pd.Series([0.01, 0.04, 0.03, 0.2], index=list("abcd"))
    q = benjamini_hochberg(p)
    assert np.allclose(q.to_numpy(), [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.2])


def test_score_of_neutral_ratings_is_half():
    df = load_sessions(pd.DataFrame([{"rater_id": "A", "question_id": "q1", "C_L": 0, "U_L": 0, "S_L": 0,
                                      "C_R": 2, "U_R": 2, "S_R": 2, "pref": 1}]))
    q = candidate_scores(df)
    assert sorted(q.round(6)) == [0.5, 1.0]


def test_profiles_and_positions():
    df = load_sessions(pd.DataFrame([{"rater_id": "A", "question_id": f"q{i}", "C_L": -2, "U_L": 2, "S_L": 0,
                                      "C_R": 0, "U_R": 0, "S_R": 0, "pref": -1} for i in range(8)]))
    assert profile_screen(df).loc["A", "profile_score"] == 0.25   # P1 on one of two candidates, (1/2)/2
    pos = positional_screen(df)
    assert pos.loc["A", "n_left"] == 8 and pos.loc["A", "p_binom"] < 0.01


def test_shifted_rater_is_found_on_synthetic_data():
    df = load_sessions(simulate(shift=-1.5, seed=3))
    res = run_audit(df, seed=0, n_draws=2000, n_perm=200)
    assert res["flags"].loc["R01", "q_bh"] < 0.05
    assert res["design"]["components"] >= 1
