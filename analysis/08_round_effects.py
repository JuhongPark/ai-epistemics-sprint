"""Exploratory context checks requested in review (not pre-registered).

1. Round effects: mean replicated Brier Index of external single-round entries by round, and the
   maximum dataset horizon scored in each round (partly resolved rounds score short horizons only).
2. Recency: first round of each external entry vs its score.
3. Team identity: distinct file-name submitter prefixes within the largest named team.
"""

import numpy as np
import pandas as pd

from common import DATA, DERIVED, BENCHMARK, load_rows, preprocess, load_qfe, brier_index, record


def main():
    idx = pd.read_csv(DATA / "processed" / "index.csv")
    ext_idx = idx[idx["organization"] != BENCHMARK]
    rounds_per_entry = ext_idx.groupby(["organization", "model"])["due"].agg(["nunique", "min"])

    rep = pd.read_csv(DERIVED / "replicated_entry_scores.csv")
    rep = rep[rep["org"] != BENCHMARK].merge(
        rounds_per_entry.reset_index().rename(columns={"organization": "org", "nunique": "n_rounds", "min": "first_round"}),
        on=["org", "model"], how="left")
    single = rep[rep["n_rounds"] == 1]
    by_round = single.groupby("first_round")["bi_overall"].agg(["mean", "max", "count"]).round(2)
    print("External single-round entries: replicated Brier Index by round")
    print(by_round.tail(16).to_string())

    df = preprocess(load_rows(["file", "model", "morg", "org", "due", "id", "source", "direction",
                               "res_date", "resolved_to", "resolved", "forecast", "imputed"]))
    ds = df[(df["qtype"] == "dataset")]
    hz = ds.groupby("due")["horizon"].agg(["min", "max"])
    print("\nDataset horizons scored per round (days), selected rounds:")
    print(hz.loc[[d for d in ["2024-07-21", "2026-06-07", "2026-08-30", "2026-10-25"] if d in hz.index]].to_string())

    rep["first_month"] = rep["first_round"].str[:7]
    rec = rep.groupby("first_month")["bi_overall"].agg(["max", "mean", "count"]).round(2)
    print("\nBest and mean replicated Brier Index by first-submission month (external entries)")
    print(rec.to_string())
    lt = rep[rep["n"] < 1000]
    print(f"\nEntries with < 1000 scored questions: earliest first round {lt['first_round'].min()} | "
          f"entries with >= 1000: first rounds {rep[rep['n'] >= 1000]['first_round'].min()} to {rep[rep['n'] >= 1000]['first_round'].max()}")

    # Data and board facts quoted in the report
    per_tr = ext_idx.groupby(["organization", "due"]).size()
    lb = pd.read_csv(DATA / "leaderboards" / "lb_tournament.csv")
    lbe = lb[lb["Team"] != BENCHMARK].copy()
    cnt = lbe.groupby("Team").size()
    lbe["big"] = lbe["Team"].map(cnt) >= 6
    sup_pub = float(lb[lb["Model"].str.startswith("Superforecaster")]["Overall"].iloc[0])
    top20 = lbe.sort_values("Overall", ascending=False).head(20)
    above = lbe[lbe["Overall"] > sup_pub]
    record("data", files=len(idx), rounds=idx["due"].nunique(), ext_files=len(ext_idx),
           ext_orgs=ext_idx["organization"].nunique(), ext_rounds=ext_idx["due"].nunique(),
           ext_team_rounds=len(per_tr), ext_team_rounds_le3=(per_tr <= 3).sum())
    record("board", rows=len(lb), teams=lb["Team"].nunique(), ext_entries=len(lbe), ext_teams=len(cnt),
           big_teams=(cnt >= 6).sum(), big_entries=lbe["big"].sum(), big_share=lbe["big"].mean(),
           top20_big=top20["big"].sum(), above=len(above), above_big=above["big"].sum(),
           above_teams=above["Team"].nunique(), above_min_entries=cnt[above["Team"].unique()].min(),
           above_max_entries=cnt[above["Team"].unique()].max(), above_N_min=above["N"].min(),
           above_N_max=above["N"].max(), n_ge1000=(lbe["N"] >= 1000).sum(), sup_pub=sup_pub)
    others = by_round.drop(index=["2026-06-07", "2026-08-30"])["mean"]
    others = others[others.index >= "2026-02-01"]
    record("rounds", jun07=by_round.loc["2026-06-07", "mean"], aug30=by_round.loc["2026-08-30", "mean"],
           others_min=others.min(), others_max=others.max(), hz_jun07=hz.loc["2026-06-07", "max"],
           hz_aug30=hz.loc["2026-08-30", "max"], hz_super=hz.loc["2024-07-21", "max"],
           first_lt1000=lt["first_round"].min(), best_2025_10=rec.loc["2025-10", "max"],
           best_2026_08=rec.loc["2026-08", "max"], best_ge1000=rep[rep["n"] >= 1000]["bi_overall"].max())

    named = ext_idx[~ext_idx["organization"].astype(str).str.startswith("Anonymous")]["organization"]
    gdm = ext_idx[ext_idx["organization"] == named.value_counts().idxmax()]["file"]
    prefixes = gdm.str.extract(r"\d{4}-\d{2}-\d{2}\.external\.([^.]+)\.")[0].dropna().unique()
    print(f"\nLargest named team: {len(gdm)} files; distinct file-name submitter tokens: {len(prefixes)} -> {list(prefixes)[:8]}")


if __name__ == "__main__":
    main()
