"""Step 1 (checkpoint A): replicate ForecastBench tournament leaderboard scores.

Score per entry (model_pk) = mean over its questions of (Brier - published question FE),
computed separately for dataset and market questions, averaged, rescaled so that
"Always 0.5" scores 0.25, then mapped to the Brier Index (1 - sqrt(s)) * 100.
Validation: Pearson r between replicated and published Overall for matched entries.
"""

import numpy as np
import pandas as pd

from common import DERIVED, DATA, BENCHMARK, load_rows, preprocess, load_qfe, brier_index, record

COLS = ["file", "model", "morg", "org", "due", "id", "source", "direction", "res_date",
        "resolved_to", "resolved", "forecast", "imputed"]


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    df = preprocess(load_rows(COLS))
    qfe = load_qfe()
    df["qfe"] = df["question_pk"].map(qfe)
    print(f"rows after preprocessing: {len(df):,} | rows without published FE: {df['qfe'].isna().sum():,}")
    df = df[df["qfe"].notna()]
    df["adj"] = df["brier"] - df["qfe"]

    per = (df.groupby(["model_pk", "org", "morg", "model", "qtype"])["adj"]
             .agg(["mean", "count"]).unstack("qtype"))
    per.columns = [f"{a}_{b}" for a, b in per.columns]
    per = per.reset_index()
    per["adj_overall"] = (per["mean_dataset"] + per["mean_market"]) / 2

    a05 = per[(per["org"] == BENCHMARK) & (per["model"] == "Always 0.5")]
    assert len(a05) == 1, "Always 0.5 not found exactly once"
    for col in ["mean_dataset", "mean_market", "adj_overall"]:
        per[col] = per[col] + 0.25 - a05[col].iloc[0]
    per["bi_dataset"] = brier_index(per["mean_dataset"])
    per["bi_market"] = brier_index(per["mean_market"])
    per["bi_overall"] = brier_index(per["adj_overall"])
    per["n"] = per["count_dataset"].fillna(0) + per["count_market"].fillna(0)
    per.to_csv(DERIVED / "replicated_entry_scores.csv", index=False)

    # Validate against the published tournament leaderboard (external teams + references)
    lb = pd.read_csv(DATA / "leaderboards" / "lb_tournament.csv")
    lb["key"] = lb["Model"].str.strip().str.lower()
    per["key"] = per["model"].astype(str).str.strip().str.lower()
    ext = per[per["org"] != BENCHMARK]
    m = ext.merge(lb, on="key", how="inner", suffixes=("", "_lb"))
    # keep one-to-one matches only (model names unique on both sides)
    m = m[m.groupby("key")["key"].transform("size") == 1]
    print(f"external entries replicated: {len(ext):,} | matched 1:1 to published rows: {len(m):,}")
    r_overall = np.corrcoef(m["bi_overall"], m["Overall"])[0, 1]
    mad = (m["bi_overall"] - m["Overall"]).abs()
    n_match = (m["n"] == m["N"]).mean()
    print(f"Pearson r (Overall): {r_overall:.4f} | mean |diff|: {mad.mean():.3f} | max |diff|: {mad.max():.3f}")
    print(f"share with identical N: {n_match:.3f}")
    record("rep", r=r_overall, mad=mad.mean(), maxdiff=mad.max(), share_sameN=n_match,
           n_ext=len(ext), n_matched=len(m))
    for qt, col in [("dataset", "Dataset"), ("market", "Market")]:
        ok = m[f"bi_{qt}"].notna() & m[col].notna()
        r_qt = np.corrcoef(m.loc[ok, f'bi_{qt}'], m.loc[ok, col])[0, 1]
        record("rep", **{f"r_{qt}": r_qt})
        print(f"  r ({qt}): {r_qt:.4f}")
    worst = m.assign(diff=m["bi_overall"] - m["Overall"]).sort_values("diff", key=abs, ascending=False)
    print(worst[["Team", "Model", "Overall", "bi_overall", "N", "n"]].head(8).to_string(index=False))
    m.to_csv(DERIVED / "replication_validation.csv", index=False)
    print("CHECKPOINT A:", "PASS" if r_overall > 0.95 else "FAIL", f"(r = {r_overall:.4f})")


if __name__ == "__main__":
    main()
