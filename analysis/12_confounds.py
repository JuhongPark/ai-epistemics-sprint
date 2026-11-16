"""Exploratory (added after registration): did the AI entries face an easier test, and how large a
lead could the board detect?

1. Horizon matching. The seven entries above the superforecaster median come from rounds scored only
   on dataset horizons up to 90 and 30 days; the 2024 superforecaster round runs to 365 days. We
   re-score the superforecaster median and those entries on dataset questions with horizon <= 90
   (and <= 30) days plus all market questions, with the leaderboard's pipeline and scale.
2. (Round anchors by question type moved to 10_round_anchor.py.)
3. Detectability. The smallest true lead of a single-round entry over the superforecaster score that
   a two-sided 5% test would detect half the time (1.96 x combined SE) or 80% of the time (2.80 x),
   from the median published CI of single-round external entries and the question-bootstrap SE of
   the superforecaster score; and the 95% interval of the top entry's edge.
4. Named teams only: H1 luck share and the entry-count correlation without anonymous teams.
"""

import json
import re

import numpy as np
import pandas as pd
from scipy import stats

from common import DATA, DERIVED, BENCHMARK, ROUND_COLS, load_rows, preprocess, load_qfe, brier_index, \
    round_scores, record

SUP = "Superforecaster median forecast"
BI = 156.25


def horizon_matched(df, top_pk, sup_rep):
    out = {}
    keep = set(top_pk) | set(df[df["model"].isin([SUP, "Always 0.5"])]["model_pk"])
    d0 = df[df["model_pk"].isin(keep)]
    for h in (None, 90, 30):
        d = d0 if h is None else d0[(d0["qtype"] == "market") | (d0["horizon"] <= h)]
        g = d.groupby(["model_pk", "qtype"])["adj"].mean().unstack()
        g["overall"] = (g["dataset"] + g["market"]) / 2
        a05 = g.loc[[k for k in g.index if k.endswith("_Always 0.5")], "overall"].iloc[0]
        g["bi"] = brier_index(g["overall"] + 0.25 - a05)
        s = g.loc[[k for k in g.index if k.endswith(SUP)], "bi"].iloc[0]
        gaps = g.loc[g.index.isin(top_pk), "bi"] - s
        tag = "all" if h is None else f"h{h}"
        out[tag] = dict(sup=s, gap_mean=gaps.mean(), gap_max=gaps.max(), gap_min=gaps.min(),
                        n_sup_dataset=int((d["model"] == SUP)[d["qtype"] == "dataset"].sum()))
        print(f"horizons {'all' if h is None else f'<= {h} days'}: superforecasters {s:.2f}; "
              f"top entries' gaps mean {gaps.mean():+.2f} (range {gaps.min():+.2f} to {gaps.max():+.2f})")
    for tag in ("h90", "h30"):
        out[tag]["gap_change"] = out[tag]["gap_mean"] - out["all"]["gap_mean"]
    return out


def main():
    df = preprocess(load_rows(ROUND_COLS))
    df["qfe"] = df["question_pk"].map(load_qfe())
    df = df[df["qfe"].notna()]
    df["adj"] = df["brier"] - df["qfe"]
    sup_rep = float(pd.read_csv(DERIVED / "h3_threshold.csv", index_col=0).iloc[0, 0])
    full = pd.read_csv(DERIVED / "h3_entry_full.csv")
    top = full[full["bi"] > sup_rep]
    hm = horizon_matched(df, set(top["model_pk"]), sup_rep)
    for tag, v in hm.items():
        record(f"conf.{tag}", **v)

    # 3. Detectability, on the consistent noise basis of 04b_noise_bootstrap.py (revised)
    nb = json.load(open(DERIVED / "noise_bootstrap.json"))
    se_e, se_s = nb["sd_entry_med"], nb["sd_super"]
    se = float(np.hypot(se_e, se_s))
    ents = pd.read_csv(DERIVED / "noise_bootstrap_entries.csv")
    ents = ents[ents["bi"] > sup_rep].sort_values("bi", ascending=False)
    top_e = ents.iloc[0]
    gap_top = float(top_e["bi"]) - sup_rep
    half = 1.96 * float(np.hypot(top_e["sd"], se_s))
    print(f"Detectable true lead of a single-round entry: {1.96 * se:.2f} points (50% power), {2.80 * se:.2f} (80%); "
          f"top entry's edge {gap_top:+.2f} (95% interval {gap_top - half:+.2f} to {gap_top + half:+.2f})")
    record("detect", se_entry=se_e, se_sup=se_s, mde50=1.96 * se, mde80=2.80 * se, top_gap=gap_top,
           top_lo=gap_top - half, top_hi=gap_top + half)

    # 3b. Each above-median entry's gap net of its round's shift for the Naive Forecaster (overall
    # score, as the gaps are), from 10_round_anchor.py. Interval: the entry's and the superforecaster
    # score's bootstrap SDs; the Naive Forecaster's shift is treated as fixed.
    ra = pd.read_csv(DERIVED / "round_anchor.csv")
    shift = ra[ra["qtype"] == "overall"].set_index("round")["naive"]
    idx = pd.read_csv(DATA / "processed" / "index.csv")
    rnd = idx[idx["organization"] != BENCHMARK].groupby(["organization", "model"])["due"].min()
    ents["round"] = [rnd.get((o, m)) for o, m in zip(ents["org"], ents["model"])]
    ents["gap"] = ents["bi"] - sup_rep
    ents["net"] = ents["gap"] - ents["round"].map(shift)
    ents["net_hi"] = ents["net"] + 1.96 * np.hypot(ents["sd"], se_s)
    ents[["org", "model", "round", "bi", "gap", "net", "net_hi"]].to_csv(DERIVED / "net_of_naive.csv", index=False)
    print(ents[["org", "round", "gap", "net", "net_hi"]].round(2).to_string(index=False))
    record("net", n=len(ents), n_rounds_known=int(ents["round"].isin(shift.index).sum()), n_above=int((ents["net"] > 0).sum()),
           net_max=ents["net"].max(), net_min=ents["net"].min(), net_hi_max=ents["net_hi"].max())

    # 5. Net gaps with the estimated carry-over (added after registration): the external round mean moves by
    # b points per point of the Naive Forecaster's round score (10_round_anchor.py), so subtract b x shift.
    met = json.load(open(DERIVED / "metrics.json"))
    # Revised: slopes estimated without the two strong rounds (and within teams)
    b, b_lo, b_hi, b_fe = met["anchor.slope.loo"], met["anchor.slope.loo_lo"], met["anchor.slope.loo_hi"], met["anchor.slope.fe"]
    sh = ents["round"].map(shift)
    for tag, bb in (("b", b), ("b_lo", b_lo), ("b_hi", b_hi), ("b_fe", b_fe)):
        ents[f"net_{tag}"] = ents["gap"] - bb * sh
    top_e = ents.sort_values("gap", ascending=False).iloc[0]
    record("net", top_b=top_e["net_b"], top_b_lo=top_e["net_b_hi"], top_b_hi=top_e["net_b_lo"], top_fe=top_e["net_b_fe"],
           top_full=top_e["net"], n_above_b=int((ents["net_b"] > 0).sum()))

    # 6. The real board under each display policy (added after registration). Teams above the replicated
    # superforecaster median on the 14 November board when each policy is applied to the actual scores.
    full = pd.read_csv(DERIVED / "h3_entry_full.csv")
    lbk = pd.read_csv(DATA / "leaderboards" / "lb_tournament.csv")
    k_team = lbk[lbk["Team"] != BENCHMARK].groupby("Team").size()
    best = full.groupby("org")["bi"].max()
    tmean = pd.Series(brier_index(full.groupby("org")["overall"].mean()))
    cont = pd.read_csv(DERIVED / "h3_continuous.csv")
    held80 = cont[cont["frac"] == 0.8].set_index("org")["gap_to_super"]
    gain = lambda k: met[f"cal.gain_het_k{max(kk for kk in (1, 2, 3, 6, 9, 22) if kk <= k)}"]
    corrected = pd.Series({o: best[o] - gain(int(k_team.get(o, 1))) for o in best.index})
    def margins(s):  # teams above the median and their margins
        v = sorted((s[s > 0]).to_numpy(), reverse=True)
        return ", ".join(f"+{x:.2f}" for x in v) if v else "none"
    rows = [("Status quo: any entry above the median", int((full["bi"] > sup_rep).sum()), int((best > sup_rep).sum()), margins(best - sup_rep)),
            ("One entry per team (the team's best)", int((best > sup_rep).sum()), int((best > sup_rep).sum()), margins(best - sup_rep)),
            (f"Best entry minus the model's expected selection gain for the team's k ({met['cal.gain_het_k9']:.2f} at k = 9)", int((corrected > sup_rep).sum()), int((corrected > sup_rep).sum()), margins(corrected - sup_rep)),
            ("Best entry of each team above, re-scored on held-out questions (20%)", int((held80 > 0).sum()), int((held80 > 0).sum()), margins(held80)),
            ("Team mean", int((tmean > sup_rep).sum()), int((tmean > sup_rep).sum()), margins(tmean - sup_rep))]
    pol = pd.DataFrame(rows, columns=["policy", "entries", "teams", "margins"])
    pol.to_csv(DERIVED / "real_board_policies.csv", index=False)
    print(pol.to_string(index=False))
    print("corrected best:", corrected[corrected > sup_rep - 1].round(2).to_dict())
    second = max(float(np.sort((corrected - sup_rep).to_numpy())[-2]), float(np.sort(held80.to_numpy())[-2]))
    record("realb", sq_entries=rows[0][1], sq_teams=rows[0][2], one_teams=rows[1][2], corr_teams=rows[2][2],
           held_teams=rows[3][2], mean_teams=rows[4][2], corr_top=float(corrected.max() - sup_rep), second_margin=second)

    # 4. Named teams only
    tr = pd.read_csv(DERIVED / "h1_team_rounds.csv")
    tr = tr[tr["external"]]
    named = tr[~tr["org"].str.startswith("Anonymous")]
    t = full.groupby("org")["bi"].agg(k="size", best="max", median="median")
    tn = t[~t.index.str.startswith("Anonymous")]
    rm = stats.spearmanr(tn["k"], tn["median"])[0]
    anon_entries = int(full["org"].str.startswith("Anonymous").sum())
    print(f"Named teams only: H1 luck share {named['inflation'].mean() / named['adv_selected_half'].mean():.0%} "
          f"({len(named)} team-rounds, {named['org'].nunique()} teams); Spearman(entries, median entry) {rm:.2f} "
          f"({len(tn)} teams); anonymous entries {anon_entries} of {len(full)}")
    lbn = pd.read_csv(DATA / "leaderboards" / "lb_tournament.csv")
    board_teams = set(lbn[lbn["Team"] != BENCHMARK]["Team"])
    record("named", h1_board_overlap=len(set(tr["org"]) & board_teams), h1_share=named["inflation"].mean() / named["adv_selected_half"].mean(), n_tr=len(named),
           n_teams=named["org"].nunique(), kcorr_median=rm, n_teams_board=len(tn), anon_entries=anon_entries,
           n_entries=len(full))


if __name__ == "__main__":
    main()
