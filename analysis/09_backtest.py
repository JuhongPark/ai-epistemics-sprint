"""Exploratory real-data backtest of the 'held-out next round' policy (added after registration).

For every external team-round, score each variant on that round's questions only (difficulty-
adjusted Brier Index, on the leaderboard's global scale). A round-level
'claim' is a team whose best variant in round t is above the superforecaster score (68.19,
replicated). The claim 'survives' if the same variant (same model name) is still above it in the
team's next round in which that variant appears. Single-round scores are noisy, so this is
descriptive.
"""

import numpy as np
import pandas as pd

from common import DERIVED, round_scores, record

def main():
    sup = float(pd.read_csv(DERIVED / "h3_threshold.csv", index_col=0).iloc[0, 0])
    g = round_scores()
    g.to_csv(DERIVED / "round_level_scores.csv", index=False)
    claims = []
    for org, d in g.groupby("org"):
        rounds = sorted(d["due"].unique())
        for i, t in enumerate(rounds):
            dt = d[d["due"] == t]
            best = dt.loc[dt["bi"].idxmax()]
            if best["bi"] <= sup:
                continue
            later = d[(d["model"] == best["model"]) & (d["due"] > t)].sort_values("due")
            nxt = later.iloc[0] if len(later) else None
            claims.append(dict(org=org, round=t, model=best["model"], k=len(dt), bi_t=best["bi"],
                               next_round=None if nxt is None else nxt["due"],
                               bi_next=np.nan if nxt is None else nxt["bi"]))
    c = pd.DataFrame(claims)
    c.to_csv(DERIVED / "backtest_claims.csv", index=False)
    tested = c.dropna(subset=["bi_next"])
    surv = (tested["bi_next"] > sup)
    record("backtest", claims=len(c), teams=c["org"].nunique() if len(c) else 0, tested=len(tested),
           survived=int(surv.sum()), mean_bi_t=tested["bi_t"].mean() if len(tested) else np.nan,
           mean_bi_next=tested["bi_next"].mean() if len(tested) else np.nan,
           multi_entry_claims=int((c["k"] >= 2).sum()) if len(c) else 0)
    print(f"Round-level claims above {sup:.2f}: {len(c)} from {c['org'].nunique()} teams; "
          f"testable in a later round: {len(tested)}; survived: {int(surv.sum())}")
    if len(tested):
        print(f"Mean score: {tested['bi_t'].mean():.2f} in the claim round -> {tested['bi_next'].mean():.2f} next time")
        print(tested[["org", "round", "k", "bi_t", "next_round", "bi_next"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
