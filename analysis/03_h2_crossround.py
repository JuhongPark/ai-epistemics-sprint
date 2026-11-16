"""H2 (pre-registered): cross-round regression to the mean.

For each team, take pairs of consecutive rounds (in that team's own submission history) in
which >= 2 variants (same model name) appear in both rounds. In round t, the best variant is
the one with the lowest mean Brier on the round's common questions; its advantage is the
mean of the overlapping variants minus its score. Shrinkage = advantage in t - advantage of
the same variant in t+1. H2: mean shrinkage > 0 (cluster bootstrap by team, one-sided).
Also reported: mean Spearman correlation of within-team variant ranks across the two rounds.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import DERIVED, BENCHMARK, load_rows, preprocess, record

RNG = np.random.default_rng(20261115)
N_BOOT = 10_000
COLS = ["file", "model", "morg", "org", "due", "id", "source", "direction", "res_date",
        "resolved_to", "resolved", "forecast", "imputed"]


def round_scores(g):
    """Per-variant mean Brier on question ids answered by all variants in this round."""
    g = g.assign(qid=g["source"].astype(str) + "_" + g["id"].astype(str))
    wide = g.groupby(["model", "qid"])["brier"].mean().unstack("qid").dropna(axis=1)
    return wide.mean(1) if wide.shape[1] >= 20 else None


def main():
    df = preprocess(load_rows(COLS))
    df = df[~((df["org"] == BENCHMARK) & (df["morg"] == BENCHMARK))]

    pairs = []
    for org, g in df.groupby("org"):
        rounds = sorted(g["due"].unique())
        scores = {r: round_scores(g[g["due"] == r]) for r in rounds}
        rounds = [r for r in rounds if scores[r] is not None]
        for t, t1 in zip(rounds[:-1], rounds[1:]):
            common = scores[t].index.intersection(scores[t1].index)
            if len(common) < 2:
                continue
            s_t, s_t1 = scores[t][common], scores[t1][common]
            best = s_t.idxmin()
            adv_t = s_t.mean() - s_t[best]
            adv_t1 = s_t1.mean() - s_t1[best]
            rho = spearmanr(s_t.values, s_t1.values).correlation if len(common) >= 3 else np.nan
            pairs.append(dict(org=org, round_t=t, round_t1=t1, k=len(common), adv_t=adv_t,
                              adv_t1=adv_t1, shrinkage=adv_t - adv_t1,
                              best_still_best=s_t1.idxmin() == best, spearman=rho,
                              external=org != BENCHMARK))
    pr = pd.DataFrame(pairs)
    pr.to_csv(DERIVED / "h2_round_pairs.csv", index=False)

    def report(name, d, key):
        teams = d["org"].unique()
        by_team = {o: d[d["org"] == o] for o in teams}
        boot = []
        for _ in range(N_BOOT):
            pick = RNG.choice(teams, len(teams), replace=True)
            boot.append(pd.concat([by_team[o] for o in pick])["shrinkage"].mean())
        boot = np.array(boot)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ret_boot = []  # added after registration; separate RNG keeps registered results unchanged
        rng_r = np.random.default_rng(hash(key) % 2**32 if False else (20261120 if key == "all" else 20261121))
        for _ in range(2000):
            pick = pd.concat([by_team[o] for o in rng_r.choice(teams, len(teams), replace=True)])
            ret_boot.append(pick["adv_t1"].mean() / pick["adv_t"].mean())
        record(f"h2.{key}", retained_lo=np.percentile(ret_boot, 2.5), retained_hi=np.percentile(ret_boot, 97.5))
        # Added after registration: smallest shrinkage the registered test detects with 80% power
        # (one-sided alpha 0.05), from the bootstrap standard error: (1.645 + 0.842) x SE.
        se = float(np.std(boot, ddof=1))
        record(f"h2.{key}", mde=2.487 * se, mde_share=2.487 * se / d["adv_t"].mean())
        record(f"h2.{key}", pairs=len(d), teams=len(teams), adv_t=d["adv_t"].mean(), adv_t1=d["adv_t1"].mean(),
               shrink=d["shrinkage"].mean(), lo=lo, hi=hi, p=(boot <= 0).mean(),
               retained=d["adv_t1"].mean() / d["adv_t"].mean(), still_best=d["best_still_best"].mean(),
               chance=(1 / d["k"]).mean(), spearman=d["spearman"].mean(), n_spearman=d["spearman"].notna().sum())
        print(f"\nH2 [{name}] round pairs = {len(d)} | teams = {len(teams)} | k range {d['k'].min()}-{d['k'].max()}")
        print(f"  mean advantage of round-t best, round t:   {d['adv_t'].mean():.5f} Brier")
        print(f"  same variant's advantage in round t+1:     {d['adv_t1'].mean():.5f} Brier")
        print(f"  mean shrinkage: {d['shrinkage'].mean():.5f}  95% CI [{lo:.5f}, {hi:.5f}]  "
              f"one-sided p = {(boot <= 0).mean():.4f}")
        print(f"  retained share of advantage: {d['adv_t1'].mean() / d['adv_t'].mean():.2f}")
        print(f"  best variant still best in t+1: {d['best_still_best'].mean():.2f} "
              f"(chance level = mean 1/k = {(1 / d['k']).mean():.2f})")
        print(f"  mean Spearman of variant ranks (k>=3 pairs, n={d['spearman'].notna().sum()}): "
              f"{d['spearman'].mean():.2f}")

    report("primary: all teams incl. ForecastBench", pr, "all")
    report("robustness: external teams only", pr[pr["external"]], "ext")

    # Added after review (not pre-registered): exact sign-flip tests, which do not rely on a
    # bootstrap with few clusters.
    import itertools
    for name, d in [("external", pr[pr["external"]]), ("all teams", pr)]:
        tm = d.groupby("org")["shrinkage"].mean().to_numpy()
        n = len(tm)
        if n <= 20:
            signs = np.array(list(itertools.product([1, -1], repeat=n)))
            null = (signs * tm).mean(1)
        else:
            null = (RNG.choice([1, -1], (200_000, n)) * tm).mean(1)
        p_team = (null >= tm.mean() - 1e-15).mean()
        x = d["shrinkage"].to_numpy()
        nullp = (RNG.choice([1, -1], (200_000, len(x))) * x).mean(1)
        p_pair = (nullp >= x.mean() - 1e-15).mean()
        record(f"h2.{'ext' if name == 'external' else 'all'}", signflip_team=p_team, signflip_pair=p_pair)
        print(f"Sign-flip one-sided p [{name}]: team-level (n={n}) {p_team:.4f} | pair-level (n={len(x)}) {p_pair:.4f}")


if __name__ == "__main__":
    main()
