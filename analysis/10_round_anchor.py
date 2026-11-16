"""Exploratory (added after registration; rewritten after a code audit): were the two
strong rounds (7 June and 30 August 2026) easy, or were their entries better?

What a fixed reference can and cannot measure. ForecastBench's dataset-question difficulty effects are
fitted on its own model runs (leaderboard main.py, two_way_fixed_effects: external submissions and the
Always-x, Random Uniform and Imputed dummies are dropped). With a free effect per question, the fitted
runs' adjusted scores average to their model effects on every question, so the pool's average is flat
across rounds by construction and cannot reveal an easy round. On the tournament board, market questions
are adjusted by the market's own Brier score, so a shift there measures skill against the crowd, not
round difficulty. (An earlier version averaged the fixed LLM runs and read their lower scores as evidence
against easy rounds; that inference was invalid and is withdrawn.)

What we report, per strong round and question type, relative to the other rounds of the same block
(ForecastBench renamed its runs after 21 June 2026, so blocks are February to 21 June and 5 July to
13 September):
  - external entries: entry-weighted mean of all external round-level scores (every entry of every team
    in that round), with a bootstrap over entries;
  - the Naive Forecaster, a fixed persistence-style baseline that ForecastBench runs every round;
  - ForecastBench's fixed LLM runs (pinned on dataset questions; skill against the crowd on markets),
    shown for transparency only.
Scores are in index points of the leaderboard's scale; positive = higher (better) in the strong round.
"""

import numpy as np
import pandas as pd

from common import DERIVED, BENCHMARK, round_scores, brier_index, record

BLOCKS = {"2026-06-07": ("2026-02-01", "2026-06-21"), "2026-08-30": ("2026-07-05", "2026-09-13")}
STRONG = list(BLOCKS)
DUMMIES = {"Always 0", "Always 1", "Always 0.5", "Random Uniform", "Imputed Forecaster", "Naive Forecaster"}
RNG = np.random.default_rng(20261126)
N_BOOT = 2000


def main():
    g = round_scores(include_benchmark=True)
    g = g[g["due"] >= "2026-02-01"].copy()
    # Per-type index: one global shift per type (as the leaderboard does for the overall score), so
    # always-0.5 averages 0.25 over these rounds; a per-round shift would undo the difficulty adjustment.
    a05 = g[g["model"] == "Always 0.5"]
    for qt in ("dataset", "market"):
        g[f"bi_{qt}"] = brier_index(g[qt] - a05[qt].mean() + 0.25)
    g["bi_overall"] = g["bi"]
    fb = g[(g["org"] == BENCHMARK) & ~g["model"].isin(DUMMIES) & ~g["model"].str.startswith("Superforecaster")
           & ~g["model"].str.startswith("Public")]
    naive = g[(g["org"] == BENCHMARK) & (g["model"] == "Naive Forecaster")].set_index("due")
    ext = g[g["org"] != BENCHMARK]

    rows = []
    for strong, (lo, hi) in BLOCKS.items():
        blk = lambda d: d[(d["due"] >= lo) & (d["due"] <= hi)]
        e = blk(ext)
        for qt in ("dataset", "market", "overall"):
            c = f"bi_{qt}"
            es, eo = e[e["due"] == strong][c].dropna().to_numpy(), e[e["due"] != strong][c].dropna().to_numpy()
            shift = es.mean() - eo.mean()
            boot = [RNG.choice(es, len(es)).mean() - RNG.choice(eo, len(eo)).mean() for _ in range(N_BOOT)]
            nv = naive.loc[(naive.index >= lo) & (naive.index <= hi), c]
            n_shift = nv[strong] - nv.drop(strong).mean()
            dev = []
            for m, d in blk(fb).groupby("model"):
                d = d.set_index("due")[c].dropna()
                if strong in d.index and len(d) >= 3:
                    dev.append(d[strong] - d.drop(strong).mean())
            rows.append(dict(round=strong, qtype=qt, external=shift, ext_lo=np.percentile(boot, 2.5),
                             ext_hi=np.percentile(boot, 97.5), n_ext=len(es), naive=n_shift,
                             ext_minus_naive=shift - n_shift, fb_llm=np.mean(dev), n_fb=len(dev)))
    r = pd.DataFrame(rows)
    rows_by = {(x["round"], x["qtype"]): x for x in rows}
    r.to_csv(DERIVED / "round_anchor.csv", index=False)
    with pd.option_context("display.float_format", "{:+.2f}".format, "display.width", 200):
        print("Strong round minus other rounds of its block (index points):")
        print(r.to_string(index=False))
    for _, x in r.iterrows():
        record(f"anchor.{x['round'].replace('-', '')}.{x['qtype']}",
               **{k: x[k] for k in ("external", "ext_lo", "ext_hi", "n_ext", "naive", "ext_minus_naive", "fb_llm", "n_fb")})

    # Added after registration. (a) How unusual are the strong rounds for the Naive Forecaster? Its
    # round-to-round spread over the 2026 rounds (also used as an alternative round-noise scale in the
    # simulation). (b) How much of a round shift for the Naive Forecaster carries over to external
    # entries? Slope of the external entry-weighted round mean on the Naive Forecaster's round score,
    # across rounds with at least three external entries, with a bootstrap over rounds.
    nv_all = naive["bi_overall"]
    nv26 = nv_all[nv_all.index >= "2026-02-01"]
    # Revised (third audit): express each strong round's shift (measured within its
    # block, as above) in units of the Naive Forecaster's round-to-round SD over the other 2026 rounds,
    # so the shift and its z-score share one reference.
    sd_ref = float(nv26.drop(STRONG).std(ddof=1))
    shift = {r: float(rows_by[(r, "overall")]["naive"]) for r in STRONG}
    z = {r: shift[r] / sd_ref for r in STRONG}
    g_all = round_scores(include_benchmark=True)
    nv_full = g_all[(g_all["org"] == BENCHMARK) & (g_all["model"] == "Naive Forecaster")].set_index("due")["bi"]
    ext_rows = g_all[g_all["org"] != BENCHMARK]
    ext_all = ext_rows.groupby("due")["bi"].agg(["mean", "size"])
    rng_s = np.random.default_rng(20261136)

    def slope(dd):
        x, y = dd["naive"] - dd["naive"].mean(), dd["ext"] - dd["ext"].mean()
        return float((x * y).sum() / (x * x).sum())

    # Revised: the carry-over slope is estimated WITHOUT the two strong rounds it is
    # used to judge (2026 rounds), with a bootstrap over rounds; and within teams (team fixed effects:
    # each team's round mean, demeaned by team, on the Naive Forecaster's score, demeaned the same way).
    d = pd.DataFrame({"ext": ext_all[ext_all["size"] >= 3]["mean"], "naive": nv_full}).dropna()
    d26 = d[(d.index >= "2026-02-01") & ~d.index.isin(STRONG)]
    b = slope(d26)
    bs = [slope(d26.iloc[rng_s.integers(0, len(d26), len(d26))]) for _ in range(2000)]
    tr = ext_rows.groupby(["org", "due"])["bi"].mean().reset_index()
    tr = tr[(tr["due"] >= "2026-02-01") & ~tr["due"].isin(STRONG)]
    tr["naive"] = tr["due"].map(nv_full)
    tr = tr.dropna()
    tr = tr[tr.groupby("org")["due"].transform("nunique") >= 2]
    xd = tr["naive"] - tr.groupby("org")["naive"].transform("mean")
    yd = tr["bi"] - tr.groupby("org")["bi"].transform("mean")
    b_fe = float((xd * yd).sum() / (xd * xd).sum())
    print(f"Naive Forecaster: SD over other 2026 rounds {sd_ref:.2f}; strong-round shifts "
          f"{shift[STRONG[0]]:+.2f} and {shift[STRONG[1]]:+.2f} = {z[STRONG[0]]:.2f} and {z[STRONG[1]]:.2f} SD")
    print(f"Carry-over slope without the strong rounds (2026, {len(d26)} rounds): {b:.2f} "
          f"[{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]; within teams ({len(tr)} team-rounds): {b_fe:.2f}")
    record("anchor.naive", sd_2026=sd_ref, n_2026=int(len(nv26) - 2), z_jun07=z[STRONG[0]], z_aug30=z[STRONG[1]])
    record("anchor.slope", loo=b, loo_lo=float(np.percentile(bs, 2.5)), loo_hi=float(np.percentile(bs, 97.5)),
           n_loo=len(d26), fe=b_fe, n_fe=len(tr))


if __name__ == "__main__":
    main()
