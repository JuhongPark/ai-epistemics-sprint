"""Checkpoint B + H1 (pre-registered, pre-registration, section 4).

Team-round = (organization, forecast due date). Entries = model_pk within a team-round.
Within a team-round all entries answer the same question set, so question fixed effects
cancel and raw Brier differences equal difficulty-adjusted differences.

H1 split-half selection inflation, per team-round with k >= 2 entries:
  split question ids (source+id; all horizons together) into halves A/B (200 random splits);
  pick the best entry on A; advantage = team-mean Brier - best-entry Brier;
  inflation = advantage on A - advantage on B.
Pooled mean inflation > 0, one-sided cluster bootstrap over team-rounds (alpha = 0.05).

Reference null (descriptive): best-minus-mean gap when entry labels are permuted within
team x question (all entries equal skill).
"""

import numpy as np
import pandas as pd

from common import DERIVED, BENCHMARK, load_rows, preprocess, record

RNG = np.random.default_rng(20261114)
N_SPLITS = 200
N_BOOT = 10_000
N_PERM = 200
COLS = ["file", "model", "morg", "org", "due", "id", "source", "direction", "res_date",
        "resolved_to", "resolved", "forecast", "imputed"]


def team_round_matrices(df):
    """Yield (org, due, entries, M) with M = k x m matrix of per-question-id mean Brier,
    restricted to question ids answered by every entry in the team-round."""
    df = df.assign(qid=df["source"].astype(str) + "_" + df["id"].astype(str))
    per = df.groupby(["org", "due", "model_pk", "qid"])["brier"].mean()
    for (org, due), g in per.groupby(level=["org", "due"]):
        wide = g.droplevel(["org", "due"]).unstack("qid")
        if len(wide) < 2:
            continue
        wide = wide.dropna(axis=1)
        if wide.shape[1] < 20:
            continue
        yield org, due, list(wide.index), wide.to_numpy()


def split_half_inflation(M, rng, frac=0.5):
    k, m = M.shape
    infl, adv_a, adv_b = [], [], []
    for _ in range(N_SPLITS):
        mask = np.zeros(m, bool)
        n_sel = m // 2 if frac == 0.5 else max(1, min(m - 1, int(round(m * frac))))  # 0.5 = registered
        mask[rng.choice(m, n_sel, replace=False)] = True
        a, b = M[:, mask].mean(1), M[:, ~mask].mean(1)
        best = np.argmin(a)
        ad_a = a.mean() - a[best]
        ad_b = b.mean() - b[best]
        infl.append(ad_a - ad_b); adv_a.append(ad_a); adv_b.append(ad_b)
    return np.mean(infl), np.mean(adv_a), np.mean(adv_b)


def permutation_gap(M, rng):
    """Expected best-minus-mean gap if entries were exchangeable (shuffle within question)."""
    k, m = M.shape
    gaps = []
    for _ in range(N_PERM):
        P = np.take_along_axis(M, rng.permuted(np.tile(np.arange(k)[:, None], (1, m)), axis=0), axis=0)
        s = P.mean(1)
        gaps.append(s.mean() - s.min())
    return np.mean(gaps)


def cluster_boot_mean(x, rng, n=N_BOOT):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), (n, len(x)))
    return x[idx].mean(1)


def main():
    df = preprocess(load_rows(COLS))
    dummy = (df["org"] == BENCHMARK) & (df["morg"] == BENCHMARK)
    df = df[~dummy]

    rows = []
    for org, due, entries, M in team_round_matrices(df):
        s = M.mean(1)
        infl, adv_a, adv_b = split_half_inflation(M, RNG)
        rows.append(dict(org=org, due=due, k=len(entries), m=M.shape[1],
                         observed_gap=s.mean() - s.min(),
                         null_gap=permutation_gap(M, RNG),
                         inflation=infl, adv_selected_half=adv_a, adv_heldout_half=adv_b,
                         external=org != BENCHMARK))
    tr = pd.DataFrame(rows)
    tr.to_csv(DERIVED / "h1_team_rounds.csv", index=False)

    ext = tr[tr["external"]]
    print("CHECKPOINT B")
    print(f"  team-rounds with k>=2 (all): {len(tr)} | external: {len(ext)} | "
          f"external teams: {ext['org'].nunique()} | k range: {ext['k'].min()}-{ext['k'].max()} | "
          f"median common question ids: {ext['m'].median():.0f}")
    print("  ->", "PASS" if len(ext) >= 20 else "FAIL", "(target >= 20 external team-rounds)")
    record("h1", n_all=len(tr), n_ext=len(ext), teams_ext=ext["org"].nunique(), median_m=ext["m"].median())

    def report(name, d, key):
        boot = cluster_boot_mean(d["inflation"], RNG)
        p = (boot <= 0).mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        share = d["inflation"].mean() / d["adv_selected_half"].mean()
        record(f"h1.{key}", adv_a=d["adv_selected_half"].mean(), adv_b=d["adv_heldout_half"].mean(),
               inflation=d["inflation"].mean(), lo=lo, hi=hi, p=p, share=share,
               ratio=d["observed_gap"].mean() / d["null_gap"].mean())
        print(f"\nH1 [{name}] n team-rounds = {len(d)}")
        print(f"  mean advantage on selection half: {d['adv_selected_half'].mean():.5f} Brier")
        print(f"  mean advantage on held-out half:  {d['adv_heldout_half'].mean():.5f} Brier")
        print(f"  mean inflation: {d['inflation'].mean():.5f}  95% CI [{lo:.5f}, {hi:.5f}]  "
              f"one-sided p = {p:.4f}")
        print(f"  luck share (inflation / in-sample advantage): {share:.2f}")
        print(f"  full-sample best-minus-mean gap: {d['observed_gap'].mean():.5f} | "
              f"equal-skill null gap: {d['null_gap'].mean():.5f} | "
              f"ratio observed/null: {d['observed_gap'].mean() / d['null_gap'].mean():.2f}")

    report("primary: all teams incl. ForecastBench", tr, "all")
    report("robustness: external teams only", ext, "ext")

    # Added after review (not pre-registered): bootstrap clustered by team rather than team-round
    def team_cluster(d, name, key):
        teams = d["org"].unique()
        g = {o: d[d["org"] == o] for o in teams}
        infl, share = [], []
        for _ in range(2000):
            pick = pd.concat([g[o] for o in RNG.choice(teams, len(teams))])
            infl.append(pick["inflation"].mean()); share.append(pick["inflation"].mean() / pick["adv_selected_half"].mean())
        lo, hi = np.percentile(infl, [2.5, 97.5]); slo, shi = np.percentile(share, [2.5, 97.5])
        record(f"h1.{key}.team", lo=lo, hi=hi, share_lo=slo, share_hi=shi, n_teams=len(teams))
        print(f"\nH1 team-clustered bootstrap [{name}] teams = {len(teams)}: inflation CI [{lo:.5f}, {hi:.5f}] | "
              f"luck share CI [{slo:.2f}, {shi:.2f}] | one-sided p = {(np.array(infl) <= 0).mean():.4f}")
    team_cluster(ext, "external", "ext")
    team_cluster(tr, "all teams", "all")

    # Added after review: luck share as a function of the share of questions used for selection
    rows = []
    mats = [(org, M) for org, due, entries, M in team_round_matrices(df) if org != BENCHMARK]
    for frac in [0.3, 0.5, 0.7, 0.8, 0.9]:
        res = [split_half_inflation(M, RNG, frac) for _, M in mats]
        infl = np.array([r[0] for r in res]); adv = np.array([r[1] for r in res])
        boot = RNG.integers(0, len(infl), (2000, len(infl)))
        shares = infl[boot].mean(1) / adv[boot].mean(1)
        rows.append(dict(frac=frac, luck_share=infl.mean() / adv.mean(), infl=infl.mean(), adv=adv.mean(),
                         lo=np.percentile(shares, 2.5), hi=np.percentile(shares, 97.5)))
    sf = pd.DataFrame(rows)
    sf.to_csv(DERIVED / "h1_selection_fraction.csv", index=False)
    print("\nLuck share by fraction of questions used for selection (external teams):")
    print(sf.round(3).to_string(index=False))

    # Added after registration (not pre-registered): positive control. Shuffling entry
    # labels within each question makes all entries equally skilled by construction, so a valid
    # estimator must return a held-out advantage near 0 and a luck share near 100%. Uses its own
    # RNG so the registered numbers above are unchanged.
    rng_pc = np.random.default_rng(20261119)
    pc = []
    for _, M in mats:
        k, m = M.shape
        perm = rng_pc.permuted(np.tile(np.arange(k)[:, None], (1, m)), axis=0)
        pc.append(split_half_inflation(np.take_along_axis(M, perm, axis=0), rng_pc))
    pc_infl = np.array([r[0] for r in pc]); pc_a = np.array([r[1] for r in pc]); pc_b = np.array([r[2] for r in pc])
    boot = rng_pc.integers(0, len(pc_b), (2000, len(pc_b)))
    b_lo, b_hi = np.percentile(pc_b[boot].mean(1), [2.5, 97.5])
    record("h1.posctl", adv_a=pc_a.mean(), adv_b=pc_b.mean(), adv_b_lo=b_lo, adv_b_hi=b_hi,
           share=pc_infl.mean() / pc_a.mean(), n=len(pc))
    print(f"\nPositive control (labels shuffled within question, equal skill by construction): "
          f"selection-half advantage {pc_a.mean():.5f}, held-out {pc_b.mean():.5f} [{b_lo:.5f}, {b_hi:.5f}], "
          f"luck share {pc_infl.mean() / pc_a.mean():.2f}")


if __name__ == "__main__":
    main()
