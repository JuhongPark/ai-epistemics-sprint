"""Exploratory (added after registration): selection across the whole board, with null arms.

Reviewers asked whether the held-out drop of the board's top entries is selection or an artefact
(held-out questions differing in difficulty), what it is when the board selects on all questions,
and whether the Figure 1 pattern only reflects stronger teams submitting more entries.

1. For selection fractions f = 0.5 ... 0.9 (200 random question splits each, all horizons of a
   question kept together), pick the ten best external entries on the selection set and re-score
   them on the held-out set. Null arms on the same splits: ten random entries (any systematic
   difference between the two sets) and the bottom ten (regression to the mean should lift them).
   Also: entries above the superforecaster score on the selection set.
2. Extrapolation to selection on all questions: a log-log fit of the gain against 1/f over
   f = 0.5 ... 0.9, evaluated at f = 1 (revised: the earlier 1/f-to-1/sqrt(f) bracket
   is not a bound, and it overshot the observed 90% values). The same is done for H1's within-team
   gain (h1_selection_fraction.csv).
3. Entry count and team strength: Spearman correlation, over external teams, of the number of
   entries with the best entry's score and with the team's median entry score.
4. Replication error for the top 20 external entries.
"""

import importlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from common import DERIVED, BENCHMARK, record

warnings.filterwarnings("ignore", "invalid value encountered in sqrt")  # entries absent from a split
h3 = importlib.import_module("04_h3_consequence")
RNG = np.random.default_rng(20261127)
FRACS = (0.5, 0.6, 0.7, 0.8, 0.9)
N_SPLITS = 200
N_BOOT, N_INNER = 200, 20
BI = 156.25  # index points per unit Brier near the top of the board


def loglog_extrapolate(f, y):
    """Selection gain falls as the selection share f rises. Its local elasticity in 1/f is not fixed
    (it exceeds 1 when true differences are large relative to noise), so we fit log y = a + b log(1/f)
    across the observed fractions and evaluate at f = 1; the observed value at the largest f is an
    upper bound. Returns (prediction at f = 1, fitted elasticity b, observed value at the largest f)."""
    f, y = np.asarray(f, float), np.asarray(y, float)
    b, a = np.polyfit(np.log(1 / f), np.log(y), 1)
    return float(np.exp(a)), float(b), float(y[np.argmax(f)])


def main():
    ent, qids, mats = h3.build()
    sup = float(pd.read_csv(DERIVED / "h3_threshold.csv", index_col=0).iloc[0, 0])
    ext = (ent["org"] != BENCHMARK).to_numpy()
    rows, curve = [], []
    for f in FRACS:
        for _ in range(N_SPLITS):
            mask = RNG.random(len(qids)) < f
            sa, sb = h3.score(ent, mats, mask.astype(float)), h3.score(ent, mats, (~mask).astype(float))
            d = pd.DataFrame({"org": ent["org"], "a": sa["bi"], "b": sb["bi"]})[ext].dropna()
            top, bot = d.nlargest(10, "a"), d.nsmallest(10, "a")
            # Added after registration: one entry per team (each team's best on the selection set), so the
            # top ten are not clustered within a few teams
            tt = d.loc[d.groupby("org")["a"].idxmax()].nlargest(10, "a")
            t1, t3 = d.nlargest(1, "a"), d.nlargest(3, "a")  # added after registration: drop by rank
            rnd = d.iloc[RNG.choice(len(d), 10, replace=False)]
            abv = d[d["a"] > sup]
            rows.append(dict(f=f, top=(top["a"] - top["b"]).mean(), rand=(rnd["a"] - rnd["b"]).mean(),
                             bottom=(bot["a"] - bot["b"]).mean(),
                             above=(abv["a"] - abv["b"]).mean() if len(abv) else np.nan, n_above=len(abv),
                             above_held=(abv["b"] > sup).sum() if len(abv) else 0,
                             top_teams=(tt["a"] - tt["b"]).mean(), top1=(t1["a"] - t1["b"]).mean(),
                             top3=(t3["a"] - t3["b"]).mean()))
    r = pd.DataFrame(rows)
    r["net"] = r["top"] - r["rand"]
    for f, g in r.groupby("f"):
        curve.append(dict(f=f, top=g["top"].mean(), top_lo=g["top"].quantile(0.025), top_hi=g["top"].quantile(0.975),
                          rand=g["rand"].mean(), rand_lo=g["rand"].quantile(0.025), rand_hi=g["rand"].quantile(0.975),
                          bottom=g["bottom"].mean(), net=g["net"].mean(), net_lo=g["net"].quantile(0.025),
                          net_hi=g["net"].quantile(0.975), above=g["above"].mean(), n_above=g["n_above"].mean(),
                          above_held=g["above_held"].mean(), top_teams=g["top_teams"].mean(),
                          teams_net=(g["top_teams"] - g["rand"]).mean(), top1_net=(g["top1"] - g["rand"]).mean(),
                          top3_net=(g["top3"] - g["rand"]).mean()))
    c = pd.DataFrame(curve)
    c.to_csv(DERIVED / "board_selection_curve.csv", index=False)
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200):
        print("Held-out change of the board's top ten, ten random entries and the bottom ten (index points; "
              "positive = loses on held-out questions); 95% range across splits:")
        print(c.to_string(index=False))

    kb, eb, kb90 = loglog_extrapolate(c["f"], c["net"])
    h1 = pd.read_csv(DERIVED / "h1_selection_fraction.csv")
    h1 = h1[h1["frac"] >= 0.5]
    jb, ej, jb90 = loglog_extrapolate(h1["frac"], h1["infl"] * BI)
    adv1 = float(h1["adv"].iloc[-1] * BI)  # advantage on the selection set at the largest fraction
    print(f"Extrapolated to selection on all questions (log-log fit): board top-ten selection gain {kb:.2f} points "
          f"(elasticity {eb:.2f}; observed at 90%: {kb90:.2f}); within-team gain {jb:.3f} points (elasticity {ej:.2f}; "
          f"at 90%: {jb90:.3f}), luck share {jb / adv1:.0%} (at 90%: {jb90 / adv1:.0%})")
    rec = {f"f{int(f * 100)}_{k}": c.set_index("f").loc[f, k] for f in (0.5, 0.8)
           for k in ("top", "top_lo", "top_hi", "rand", "rand_lo", "rand_hi", "bottom", "net", "net_lo", "net_hi",
                     "above", "n_above", "above_held", "top_teams", "teams_net", "top1_net", "top3_net")}
    record("bsel", full=kb, full_elast=eb, f90_net_obs=kb90, h1_full=jb, h1_elast=ej, h1_f90=jb90,
           h1_full_luck=jb / adv1, h1_f90_luck=jb90 / adv1, **rec)

    # Question-bootstrap interval for the null-corrected drop (top ten minus ten random entries):
    # resample questions with replacement, then average over random splits of the resample.
    for f in (0.5, 0.8):
        boot, inner_var = [], []
        for _ in range(N_BOOT):
            w = np.bincount(RNG.integers(0, len(qids), len(qids)), minlength=len(qids)).astype(float)
            vals = []
            for _ in range(N_INNER):
                mask = RNG.random(len(qids)) < f
                sa, sb = h3.score(ent, mats, w * mask), h3.score(ent, mats, w * ~mask)
                d = pd.DataFrame({"a": sa["bi"], "b": sb["bi"]})[ext].dropna()
                top, rnd = d.nlargest(10, "a"), d.iloc[RNG.choice(len(d), 10, replace=False)]
                vals.append((top["a"] - top["b"]).mean() - (rnd["a"] - rnd["b"]).mean())
            boot.append(np.mean(vals)); inner_var.append(np.var(vals, ddof=1))
        # Resampling with replacement duplicates questions, which adds noise and inflates any
        # selection gain, so percentile intervals are biased upward. We use the bootstrap standard
        # error around the point estimate instead.
        est = c.set_index("f").loc[f, "net"]
        # Each resample's mean over N_INNER random splits carries Monte Carlo noise; remove it.
        se = float(np.sqrt(max(np.var(boot, ddof=1) - np.mean(inner_var) / N_INNER, 0.0)))
        p_lo, p_hi = np.percentile(boot, [2.5, 97.5])
        print(f"f = {f}: null-corrected top-ten drop {est:.2f}, question-bootstrap SE {se:.2f} -> 95% interval "
              f"[{est - 1.96 * se:.2f}, {est + 1.96 * se:.2f}] (percentile interval [{p_lo:.2f}, {p_hi:.2f}] is "
              f"biased upward; {N_BOOT} x {N_INNER} splits)")
        record("bsel", **{f"f{int(f * 100)}_se": se, f"f{int(f * 100)}_ci_lo": est - 1.96 * se,
                          f"f{int(f * 100)}_ci_hi": est + 1.96 * se})

    # 3. Entry count vs team strength (Figure 1 confound)
    full = pd.read_csv(DERIVED / "h3_entry_full.csv")
    t = full.groupby("org")["bi"].agg(k="size", best="max", median="median")
    rb, pb = stats.spearmanr(t["k"], t["best"])
    rm, pm = stats.spearmanr(t["k"], t["median"])
    print(f"Teams {len(t)}: Spearman(entries, best entry) = {rb:.2f} (p = {pb:.3f}); "
          f"Spearman(entries, median entry) = {rm:.2f} (p = {pm:.3f})")
    record("kcorr", n=len(t), best=rb, best_p=pb, median=rm, median_p=pm)

    # 4. Replication error for the top 20 external entries (published order)
    v = pd.read_csv(DERIVED / "replication_validation.csv")
    v = v[v["Team"] != BENCHMARK].sort_values("Overall", ascending=False).head(20)
    err = (v["bi_overall"] - v["Overall"]).abs()
    same_n = (v["n"] == v["N"]).mean()
    print(f"Top 20 external entries: mean |replicated - published| = {err.mean():.2f}, max {err.max():.2f}; "
          f"identical question counts {same_n:.0%}")
    record("rep.top20", mae=err.mean(), max=err.max(), same_n=same_n)


if __name__ == "__main__":
    main()
