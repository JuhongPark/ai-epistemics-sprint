"""Pre-registered robustness checks (pre-registration, section 4), reported regardless of outcome.

H1 (external teams; FB baseline team excluded, which is itself a robustness check):
  - dataset questions only / market questions only
  - entries with >= 500 scored rows in the team-round
  - excluding the external team with the most team-rounds
H3:
  - entries with >= 500 and >= 1000 scored questions (status quo vs team-mean counts)
  - excluding the largest external team (by entries)
Raw vs difficulty-adjusted Brier: within a team-round all entries answer the same questions,
so H1/H2 contrasts are identical under both (question effects cancel); H3 is defined only on
the adjusted scale, because raw Brier is not comparable across rounds.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from common import DERIVED, BENCHMARK, load_rows, preprocess, brier_index, record

HERE = Path(__file__).parent
RNG = np.random.default_rng(20261118)
SUPER = float(pd.read_csv(DERIVED / "h3_threshold.csv", index_col=0).iloc[0, 0])  # replicated, from 04


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h1 = load("02_h1_inflation")


def h1_on(df, label, key):
    rows = []
    for org, due, entries, M in h1.team_round_matrices(df):
        infl, a, b = h1.split_half_inflation(M, RNG)
        rows.append(dict(org=org, inflation=infl, adv_a=a))
    d = pd.DataFrame(rows)
    if d.empty:
        print(f"  {label}: no team-rounds"); return
    boot = h1.cluster_boot_mean(d["inflation"], RNG)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    record(f"rob.h1.{key}", n=len(d), share=d["inflation"].mean() / d["adv_a"].mean(), lo=lo, hi=hi)
    print(f"  {label}: n={len(d)} | inflation {d['inflation'].mean():.5f} [{lo:.5f}, {hi:.5f}] "
          f"p={(boot <= 0).mean():.4f} | luck share {d['inflation'].mean() / d['adv_a'].mean():.2f}")


def main():
    cols = ["file", "model", "morg", "org", "due", "id", "source", "direction", "res_date",
            "resolved_to", "resolved", "forecast", "imputed"]
    df = preprocess(load_rows(cols))
    ext = df[df["org"] != BENCHMARK]

    print("H1 robustness (external teams)")
    h1_on(ext[ext["qtype"] == "dataset"], "dataset questions only", "dataset")
    h1_on(ext[ext["qtype"] == "market"], "market questions only", "market")
    n_rows = ext.groupby(["org", "due", "model_pk"])["brier"].transform("size")
    h1_on(ext[n_rows >= 500], "entries with >= 500 scored rows", "min500")
    tr = pd.read_csv(DERIVED / "h1_team_rounds.csv")
    biggest = tr[tr["external"]].groupby("org").size().idxmax()
    h1_on(ext[ext["org"] != biggest], f"excluding team with most team-rounds ({(tr['org'] == biggest).sum()} rounds)", "drop_top")

    print("\nH3 robustness")
    full = pd.read_csv(DERIVED / "h3_entry_full.csv")
    rep = pd.read_csv(DERIVED / "replicated_entry_scores.csv")[["model_pk", "n"]]
    full = full.merge(rep, on="model_pk", how="left")
    largest = full.groupby("org").size().idxmax()
    for label, d in [("all entries", full), ("entries with n >= 500", full[full["n"] >= 500]),
                     ("entries with n >= 1000", full[full["n"] >= 1000]),
                     (f"excluding largest team ({(full['org'] == largest).sum()} entries)", full[full["org"] != largest])]:
        sq = d[d["bi"] > SUPER]["org"].nunique()
        tm = (brier_index(d.groupby("org")["overall"].mean()) > SUPER).sum()
        record("rob.h3." + label.split(" (")[0].replace(" ", "_").replace(">=", "ge"), sq=sq, tm=tm)
        print(f"  {label}: teams above {SUPER:.2f}: status quo {sq} | team mean {tm}")


if __name__ == "__main__":
    main()
