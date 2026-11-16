"""RQ4 (pre-registered policy evaluation): simulation of "beats superforecasters" claims.

A team has true Brier Index = SUPER + delta for its best system. It submits k entries, at most 3
per round (ForecastBench's rule), so its entries span ceil(k/3) rounds. Each entry's observed
score = truth + round noise (shared by the team's entries in that round) + entry noise.

Calibration (computed here from the data; revised after review, which found that the first
draft hard-coded a value computed for entries with N between 550 and 650):
  - total sd   = median published 95% CI half-width of single-round external entries / 1.96
  - entry sd   = mean over external team-rounds of (equal-skill permutation gap / c_k), where
                 c_k = E[mean - min] of k iid N(0,1); converted from Brier to Brier Index with
                 dBI/dBrier = -50 / sqrt(s) at BI 68 (s = 0.1024), i.e. 156.25
  - round sd   = sqrt(total^2 - entry^2)
  - skill sd   (heterogeneous scenario) = half-normal spread that reproduces the observed mean
                 full-sample best-minus-mean gap (set by simulation; revised,
                 replacing a normal-theory formula that did not match the half-normal draw)
Scenarios:
  - equal: all of a team's entries have the same true skill
  - heterogeneous: entry 1 (the pre-declared primary) is the team's best system with truth delta;
    other entries are truly weaker by |N(0, skill sd)|
Policies: status quo (best entry), pre-declared primary entry, team mean, best entry re-scored
on a fresh round. Under equal skill the primary and held-out rows equal k = 1 by construction.
Sensitivity: total sd from entries with < 1000 scored questions.
"""

import json

import numpy as np
import pandas as pd

from common import DATA, DERIVED, BENCHMARK, record

RNG = np.random.default_rng(20261117)
N_SIM = 200_000
KS = [1, 2, 3, 6, 9, 22]
DELTAS = [0.5, 0.0, -0.5, -1.0]
DBI_DBRIER = 50 / 0.32


def c_k(k, n=400_000):
    x = RNG.standard_normal((n, k))
    return float((x.mean(1) - x.min(1)).mean())


def model_gap(s, sd_e, kw, n=200_000):
    """Mean best-minus-mean gap of a team-round under the heterogeneous model: entry 1 at 0, others
    lower by s * |N(0, 1)|, plus entry noise sd_e (common random numbers across s)."""
    r = np.random.default_rng(20261130)
    g = 0.0
    for k, w in kw.items():
        skill = np.concatenate([np.zeros((n, 1)), -s * np.abs(r.normal(0, 1, (n, k - 1)))], 1)
        x = skill + sd_e * r.normal(0, 1, (n, k))
        g += w * (x.max(1) - x.mean(1)).mean()
    return g


def calib_skill(sd_e, target, kw):
    lo, hi = 0.0, 6.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if model_gap(mid, sd_e, kw) < target else (lo, mid)
    return (lo + hi) / 2


def board_weighted_gaps():
    """Revised (code audit). H1 scores an entry by the mean Brier over its question
    ids (about a quarter are market ids), while the board and its CIs weight dataset and market
    questions 50/50. For the policy simulation, recompute each external team-round's observed
    best-minus-mean gap and its shuffled-label (equal-skill) gap with the board's weighting.
    Same team-rounds and question ids as H1 (02_h1_inflation.py: k >= 2, >= 20 shared ids)."""
    from common import load_rows, preprocess
    cols = ["file", "model", "morg", "org", "due", "id", "source", "direction", "res_date", "resolved_to",
            "resolved", "forecast", "imputed"]
    df = preprocess(load_rows(cols))
    df = df[df["org"] != BENCHMARK]
    df = df.assign(qid=df["source"].astype(str) + "_" + df["id"].astype(str))
    per = df.groupby(["org", "due", "model_pk", "qid", "qtype"])["brier"].mean()
    rng = np.random.default_rng(20261131)
    rows = []
    for (org, due), g in per.groupby(level=["org", "due"]):
        wide = g.droplevel(["org", "due"]).unstack(["qid", "qtype"])
        if len(wide) < 2:
            continue
        wide = wide.dropna(axis=1)
        if wide.shape[1] < 20:
            continue
        M = wide.to_numpy()
        types = np.asarray(wide.columns.get_level_values("qtype"))
        present = [q for q in ("dataset", "market") if (types == q).any()]
        w = np.zeros(M.shape[1])
        for q in present:
            w[types == q] = 1 / (types == q).sum() / len(present)
        s = M @ w
        k, m = M.shape
        gaps = []
        for _ in range(200):
            P = np.take_along_axis(M, rng.permuted(np.tile(np.arange(k)[:, None], (1, m)), axis=0), axis=0)
            sp = P @ w
            gaps.append(sp.mean() - sp.min())
        rows.append(dict(org=org, due=due, k=k, observed_gap=s.mean() - s.min(), null_gap=np.mean(gaps)))
    return pd.DataFrame(rows)


def calibrate():
    idx = pd.read_csv(DATA / "processed" / "index.csv")
    ext_idx = idx[idx["organization"] != BENCHMARK]
    n_rounds = ext_idx.groupby(["organization", "model"])["due"].nunique()
    lb = pd.read_csv(DATA / "leaderboards" / "lb_tournament.csv")
    e = lb[lb["Team"] != BENCHMARK].copy()
    ci = e["Overall 95% CI"].str.extract(r"\[([\d.]+),\s*([\d.]+)\]").astype(float)
    e["hw"] = (ci[1] - ci[0]) / 2
    e["rounds"] = [n_rounds.get((t, m), np.nan) for t, m in zip(e["Team"], e["Model"])]
    single = e[e["rounds"] == 1]
    hw = single["hw"].median()
    hw_alt = e[e["N"] < 1000]["hw"].median()
    # Revised: total entry noise and baseline noise come from one bootstrap
    # (04b_noise_bootstrap.py: questions within round, stratified by round x source). The published
    # CI half-width (hw) is kept for reference.
    nb = json.load(open(DERIVED / "noise_bootstrap.json"))
    sd_total = nb["sd_entry_med"]

    # H1's scale (per question id): used for the out-of-sample checks, which are measured on it
    tr = pd.read_csv(DERIVED / "h1_team_rounds.csv")
    tr = tr[tr["external"]]
    ck = {k: c_k(k) for k in sorted(tr["k"].unique())}
    kw = tr["k"].value_counts(normalize=True)
    sd_entry_h1 = (tr["null_gap"] / tr["k"].map(ck)).mean() * DBI_DBRIER
    tot_b = (tr["observed_gap"] / tr["k"].map(ck)).mean()
    sd_skill_normal = float(np.sqrt(max((tot_b * DBI_DBRIER) ** 2 - sd_entry_h1**2, 0)))
    obs_gap_h1 = float(tr["observed_gap"].mean() * DBI_DBRIER)
    # Revised: the normal-theory spread above does not match the half-normal draw
    # (it reproduced a 0.53-point gap against 0.74 observed), so the spread is set by simulation.
    sd_skill_h1 = calib_skill(sd_entry_h1, obs_gap_h1, kw)

    # Board's scale (dataset and market weighted 50/50): used for the policy simulation
    bw = board_weighted_gaps()
    bw.to_csv(DERIVED / "rq4_board_weighted_gaps.csv", index=False)
    sd_entry = float((bw["null_gap"] / bw["k"].map(ck)).mean() * DBI_DBRIER)
    obs_gap = float(bw["observed_gap"].mean() * DBI_DBRIER)
    sd_round = float(np.sqrt(max(sd_total**2 - sd_entry**2, 0)))
    sd_skill = calib_skill(sd_entry, obs_gap, bw["k"].value_counts(normalize=True))
    cal = dict(n_single_round=len(single), median_hw_single=hw, sd_total=sd_total,
               sd_entry=sd_entry, sd_round=sd_round, sd_skill=sd_skill, obs_gap=obs_gap, n_bw=len(bw),
               sd_entry_h1=sd_entry_h1, sd_skill_h1=sd_skill_h1, obs_gap_h1=obs_gap_h1,
               sd_skill_normal=sd_skill_normal, model_gap_old=model_gap(sd_skill_normal, sd_entry_h1, kw),
               median_hw_lt1000=hw_alt, sd_total_alt=hw_alt / 1.96,
               sd_round_alt=float(np.sqrt(max((hw_alt / 1.96) ** 2 - sd_entry**2, 0))))
    return cal


def simulate(k, delta, sd_round, sd_entry, sd_skill=0.0):
    rounds = int(np.ceil(k / 3))
    round_of = np.repeat(np.arange(rounds), 3)[:k]
    truth = np.full((N_SIM, k), delta)
    if sd_skill > 0 and k > 1:
        truth[:, 1:] -= np.abs(RNG.normal(0, sd_skill, (N_SIM, k - 1)))
    obs = truth + RNG.normal(0, sd_round, (N_SIM, rounds))[:, round_of] + RNG.normal(0, sd_entry, (N_SIM, k))
    best_ix = obs.argmax(1)
    best = obs[np.arange(N_SIM), best_ix]
    fresh = truth[np.arange(N_SIM), best_ix] + RNG.normal(0, sd_round, N_SIM) + RNG.normal(0, sd_entry, N_SIM)
    # Added after review: a realistic primary, chosen as the best variant on a previous round's
    # (independently noisy) scores, then scored on this round.
    prior = truth + RNG.normal(0, sd_round, (N_SIM, 1)) + RNG.normal(0, sd_entry, (N_SIM, k))
    prim_ix = prior.argmax(1)
    return {
        "status quo (best entry)": (best > 0).mean(),
        "pre-declared primary entry": (obs[:, 0] > 0).mean(),
        "primary chosen on prior round": (obs[np.arange(N_SIM), prim_ix] > 0).mean(),
        "team mean": (obs.mean(1) > 0).mean(),
        "held-out re-score of best": (fresh > 0).mean(),
        "expected inflation of best (BI)": (best - truth[np.arange(N_SIM), best_ix]).mean(),
    }


def main():
    cal = calibrate()
    pd.Series(cal).to_csv(DERIVED / "rq4_calibration.csv")
    record("cal", **cal)
    print("Calibration:", {k: round(v, 3) if isinstance(v, float) else v for k, v in cal.items()})
    rows = []
    for scen, sr, se, ss in [("equal", cal["sd_round"], cal["sd_entry"], 0.0),
                             ("heterogeneous", cal["sd_round"], cal["sd_entry"], cal["sd_skill"]),
                             ("equal, alt noise", cal["sd_round_alt"], cal["sd_entry"], 0.0)]:
        for delta in DELTAS:
            for k in KS:
                rows.append(dict(scenario=scen, delta=delta, k=k, **simulate(k, delta, sr, se, ss)))
    out = pd.DataFrame(rows)
    out.to_csv(DERIVED / "rq4_policy_sim.csv", index=False)

    # Added after registration (separate RNG; existing scenarios unchanged): (a) uncertainty in the
    # superforecaster baseline as a noise term shared by every comparison, sd from the question
    # bootstrap of the superforecaster score; (b) a winner's-curse-corrected policy that subtracts the
    # expected best-of-k inflation under equal skill before comparing.
    rng_e = np.random.default_rng(20261124)
    sd_base = float(json.load(open(DERIVED / "noise_bootstrap.json"))["sd_super"])
    infl_eq = out[(out["scenario"] == "equal") & (out["delta"] == 0.0)].set_index("k")["expected inflation of best (BI)"]
    # Revised (code audit): the correction subtracts the expected selection gain of
    # the best entry over the team's best system under the calibrated (heterogeneous) model; the
    # equal-skill gain is larger and over-corrects, so it is kept only as a worst-case column.
    rng_g = np.random.default_rng(20261132)
    def het_gain(k, sd_round, sd_entry, sd_skill, n=N_SIM):
        rounds = int(np.ceil(k / 3)); round_of = np.repeat(np.arange(rounds), 3)[:k]
        truth = np.zeros((n, k))
        if k > 1:
            truth[:, 1:] -= sd_skill * np.abs(rng_g.normal(0, 1, (n, k - 1)))
        obs = truth + rng_g.normal(0, sd_round, (n, rounds))[:, round_of] + rng_g.normal(0, sd_entry, (n, k))
        return float(obs.max(1).mean())
    infl_het = {k: het_gain(k, cal["sd_round"], cal["sd_entry"], cal["sd_skill"]) for k in KS}
    record("cal", gain_eq_k9=float(infl_eq[9]), **{f"gain_het_k{k}": v for k, v in infl_het.items()})
    extra = []
    for delta in DELTAS:
        for k in KS:
            rounds = int(np.ceil(k / 3)); round_of = np.repeat(np.arange(rounds), 3)[:k]
            truth = np.full((N_SIM, k), delta)
            if k > 1:
                truth[:, 1:] -= np.abs(rng_e.normal(0, cal["sd_skill"], (N_SIM, k - 1)))
            obs = truth + rng_e.normal(0, cal["sd_round"], (N_SIM, rounds))[:, round_of] + rng_e.normal(0, cal["sd_entry"], (N_SIM, k))
            base = rng_e.normal(0, sd_base, N_SIM)  # error in the measured superforecaster score
            prior = truth + rng_e.normal(0, cal["sd_round"], (N_SIM, 1)) + rng_e.normal(0, cal["sd_entry"], (N_SIM, k))
            bi = obs.argmax(1); best = obs[np.arange(N_SIM), bi]
            fresh = truth[np.arange(N_SIM), bi] + rng_e.normal(0, cal["sd_round"], N_SIM) + rng_e.normal(0, cal["sd_entry"], N_SIM)
            extra.append({"scenario": "hetero + baseline noise", "delta": delta, "k": k,
                          "status quo (best entry)": (best > base).mean(),
                          "winner's-curse-corrected best": (best - infl_het[k] > base).mean(),
                          "equal-skill-corrected best": (best - infl_eq[k] > base).mean(),
                          "primary chosen on prior round": (obs[np.arange(N_SIM), prior.argmax(1)] > base).mean(),
                          "held-out re-score of best": (fresh > base).mean(),
                          "team mean": (obs.mean(1) > base).mean()})
    ex = pd.DataFrame(extra)
    # Added after registration: total error at k = 9 with equal costs for a false claim (team 0.5 worse
    # appears above) and a missed lead (team 0.5 better does not appear above)
    def total_err(col, k):
        lo, hi = ex[(ex["delta"] == -0.5) & (ex["k"] == k)][col].iloc[0], ex[(ex["delta"] == 0.5) & (ex["k"] == k)][col].iloc[0]
        return lo + (1 - hi)
    cols = ["status quo (best entry)", "winner's-curse-corrected best", "primary chosen on prior round",
            "held-out re-score of best", "team mean"]
    errs = {cc: total_err(cc, 9) for cc in cols}
    record("loss", single=total_err("status quo (best entry)", 1), k9_min=min(errs.values()), k9_max=max(errs.values()),
           sq_k9=errs["status quo (best entry)"], mean_k9=errs["team mean"])
    # Added after registration: the cost ratio (false claim : missed lead) above which each policy beats the
    # status quo at k = 9, and a raised-threshold status quo matched to the primary entry's false-claim rate.
    fp = lambda col: ex[(ex["delta"] == -0.5) & (ex["k"] == 9)][col].iloc[0]
    det = lambda col: ex[(ex["delta"] == 0.5) & (ex["k"] == 9)][col].iloc[0]
    sqc = "status quo (best entry)"
    cstar = {cc: (det(sqc) - det(cc)) / (fp(sqc) - fp(cc)) for cc in cols[1:4]}
    rng_r = np.random.default_rng(20261138)
    def best_minus_base(delta, n=N_SIM, k=9):
        rounds = 3; round_of = np.repeat(np.arange(rounds), 3)[:k]
        truth = np.full((n, k), float(delta)); truth[:, 1:] -= cal["sd_skill"] * np.abs(rng_r.normal(0, 1, (n, k - 1)))
        o = truth + rng_r.normal(0, cal["sd_round"], (n, rounds))[:, round_of] + rng_r.normal(0, cal["sd_entry"], (n, k))
        return o.max(1) - rng_r.normal(0, sd_base, n)
    xw, xb = best_minus_base(-0.5), best_minus_base(0.5)
    t_match = float(np.quantile(xw, 1 - fp("primary chosen on prior round")))
    record("roc", t=t_match, det_sq=float((xb > t_match).mean()), det_prim=det("primary chosen on prior round"),
           det_held=det("held-out re-score of best"), fp_prim=fp("primary chosen on prior round"))
    record("loss", c_corr=cstar[cols[1]], c_prim=cstar[cols[2]], c_held=cstar[cols[3]])
    print(f"Break-even cost ratios vs status quo at k = 9: { {kk: round(v, 2) for kk, v in cstar.items()} }; "
          f"status quo with threshold +{t_match:.2f} matches the primary entry's false claims and detects {(xb > t_match).mean():.2f}")
    ex.to_csv(DERIVED / "rq4_policy_sim_extra.csv", index=False)
    record("cal", sd_base=sd_base)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(f"\n[hetero + baseline noise] sd_base = {sd_base:.2f}")
        print(ex.drop(columns="scenario").to_string(index=False))

    # Validation that can fail (added after registration; revised). Calibration uses H1's
    # full-round gaps; the luck share by selection fraction (Figure A1a) is a different function of the
    # same parameters, so predicting it tests the model's form. Run on H1's own scale (per question id).
    trk = pd.read_csv(DERIVED / "h1_team_rounds.csv")
    trk = trk[trk["external"]]["k"].value_counts(normalize=True)
    sf = pd.read_csv(DERIVED / "h1_selection_fraction.csv").set_index("frac")
    def luck_curve(s, sd_e, n=100_000, seed=20261129):
        r = np.random.default_rng(seed)
        out = []
        for f in sf.index:
            num = den = 0.0
            for k, w in trk.items():
                skill = np.zeros((n, k)); skill[:, 1:] = -s * np.abs(r.normal(0, 1, (n, k - 1)))
                a = skill + r.normal(0, sd_e / np.sqrt(f), (n, k))
                b = skill + r.normal(0, sd_e / np.sqrt(1 - f), (n, k))
                ix = a.argmax(1)
                adv_a = a[np.arange(n), ix] - a.mean(1); adv_b = b[np.arange(n), ix] - b.mean(1)
                num += w * (adv_a - adv_b).mean(); den += w * adv_a.mean()
            out.append(num / den)
        return np.array(out)
    pred = luck_curve(cal["sd_skill_h1"], cal["sd_entry_h1"], n=200_000)
    obs = sf["luck_share"].to_numpy()
    grid = np.round(np.arange(0.8, 3.001, 0.05), 2)
    sse = [((luck_curve(s, cal["sd_entry_h1"]) - obs) ** 2).sum() for s in grid]
    s_fit = float(grid[int(np.argmin(sse))])
    fit = luck_curve(s_fit, cal["sd_entry_h1"], n=200_000)
    pv = pd.DataFrame({"predicted": pred, "fitted_spread": fit, "observed": obs, "obs_lo": sf["lo"].to_numpy(),
                       "obs_hi": sf["hi"].to_numpy()}, index=sf.index)
    pv["inside"] = (pv["predicted"] >= pv["obs_lo"]) & (pv["predicted"] <= pv["obs_hi"])
    pv.to_csv(DERIVED / "rq4_validation_fraction.csv")
    print("\n[validation: luck share by selection fraction, predicted vs observed]")
    print(pv.round(3).to_string())
    over = pv["predicted"] - pv["observed"]
    mult = s_fit / cal["sd_skill_h1"]
    record("val", **{f"pred{int(f * 100)}": v for f, v in zip(sf.index, pred)}, n_inside=int(pv["inside"].sum()),
           n=len(pv), over_min=over.min(), over_max=over.max(), s_fit=s_fit, mult=mult,
           fit_inside=int(((pv["fitted_spread"] >= pv["obs_lo"]) & (pv["fitted_spread"] <= pv["obs_hi"])).sum()))

    # Sensitivity of the k = 9 results (added after registration; revised: the correction
    # uses each variant's own heterogeneous gain; entry-noise variants keep the total and recalibrate
    # the skill spread to the observed gap; one variant uses the spread fitted to the luck-share curve).
    rng_s = np.random.default_rng(20261128)
    naive_sd = float(json.load(open(DERIVED / "metrics.json"))["anchor.naive.sd_2026"])  # from 10_round_anchor.py
    # Revised (third audit): the Naive Forecaster's round-to-round SD is a total; the
    # round-noise component that matches sd_round is sqrt(total^2 - entry^2). An upper bound, since part
    # of that spread is the Naive Forecaster's own skill varying by round.
    rn_ub = float(np.sqrt(max(naive_sd**2 - cal["sd_entry"] ** 2, 0)))
    record("cal", sd_round_ub=rn_ub)
    kw_b = pd.read_csv(DERIVED / "rq4_board_weighted_gaps.csv")["k"].value_counts(normalize=True)
    def policies(k, delta, sd_round, sd_entry, sd_skill, sd_b, dist="normal", n=N_SIM, extra_noise=1.0, names=1):
        """False-claim / detection rates. 'corrected (known gain)' subtracts the variant's own expected
        gain (an oracle); 'corrected' subtracts the gain expected under the calibrated model, as a
        board would (revised). Strategic options: extra_noise scales the entry noise of
        entries 2..k (deliberately divergent variants); names splits the k entries across that many
        identities, one entry per identity per round, and the board corrects each identity for its
        counted k."""
        def draw(sd, size):
            if dist == "t3":
                return rng_s.standard_t(3, size) * sd / np.sqrt(3.0)
            return rng_s.normal(0, sd, size)
        rounds = int(np.ceil(k / 3)); round_of = np.repeat(np.arange(rounds), 3)[:k]
        mult = np.ones(k); mult[1:] = extra_noise
        g0 = np.zeros((n, k))
        if sd_skill > 0 and k > 1:
            g0[:, 1:] -= sd_skill * np.abs(rng_s.normal(0, 1, (n, k - 1)))
        infl = (g0 + draw(sd_round, (n, rounds))[:, round_of] + draw(sd_entry, (n, k)) * mult).max(1).mean()
        truth = np.full((n, k), delta)
        if sd_skill > 0 and k > 1:
            truth[:, 1:] -= np.abs(rng_s.normal(0, sd_skill, (n, k - 1)))
        obs_ = truth + draw(sd_round, (n, rounds))[:, round_of] + draw(sd_entry, (n, k)) * mult
        base = draw(sd_b, n)
        prior = truth + draw(sd_round, (n, 1)) + draw(sd_entry, (n, k)) * mult
        fresh_all = truth + draw(sd_round, (n, 1)) + draw(sd_entry, (n, k)) * mult
        groups = [np.arange(j, k, names) for j in range(names)]  # identity j: entries j, j+names, ...
        if names > 1:  # each identity's entries sit in different rounds; the board corrects for that k
            m_ = len(groups[0]); gg = np.zeros((n, m_))
            if sd_skill > 0:
                gg[:, 1:] -= sd_skill * np.abs(rng_s.normal(0, 1, (n, m_ - 1)))
            infl_id = (gg + draw(sd_round, (n, m_)) + draw(sd_entry, (n, m_))).max(1).mean()
        sq = corr_known = corr_cal = prim = held = np.zeros(n, bool)
        for gi in groups:
            o = obs_[:, gi]; bi = o.argmax(1); best = o[np.arange(n), bi]
            sq = sq | (best > base)
            corr_known = corr_known | (best - (infl if names == 1 else infl_id) > base)
            corr_cal = corr_cal | (best - (infl_het[len(gi)] if names == 1 else infl_id) > base)
            prim = prim | (o[np.arange(n), prior[:, gi].argmax(1)] > base)
            held = held | (fresh_all[:, gi][np.arange(n), bi] > base)
        return {"status quo": sq.mean(), "corrected (known gain)": corr_known.mean(), "corrected": corr_cal.mean(),
                "primary (prior round)": prim.mean(), "held-out round": held.mean(),
                "single entry": (obs_[:, 0] > base).mean()}
    tot = cal["sd_total"]
    def entry_variant(m):
        se = cal["sd_entry"] * m
        return float(np.sqrt(max(tot**2 - se**2, 0))), se, calib_skill(se, cal["obs_gap"], kw_b)
    c = cal
    variants = [("Calibrated (Table 4)", c["sd_round"], c["sd_entry"], c["sd_skill"], sd_base, "normal", {}),
                ("Baseline error x0.5", c["sd_round"], c["sd_entry"], c["sd_skill"], sd_base * 0.5, "normal", {}),
                ("Baseline error x1.5", c["sd_round"], c["sd_entry"], c["sd_skill"], sd_base * 1.5, "normal", {}),
                ("Skill spread x0.5", c["sd_round"], c["sd_entry"], c["sd_skill"] * 0.5, sd_base, "normal", {}),
                ("Skill spread x2", c["sd_round"], c["sd_entry"], c["sd_skill"] * 2, sd_base, "normal", {}),
                (f"Skill spread x{mult:.2f} (fits the luck-share curve)", c["sd_round"], c["sd_entry"], c["sd_skill"] * mult, sd_base, "normal", {}),
                ("Entry noise x1.2 (same total, skill refitted)", *entry_variant(1.2), sd_base, "normal", {}),
                ("Entry noise x0.6 (same total, skill refitted)", *entry_variant(0.6), sd_base, "normal", {}),
                ("Heavy-tailed noise (t, 3 df)", c["sd_round"], c["sd_entry"], c["sd_skill"], sd_base, "t3", {}),
                (f"Round noise {rn_ub:.2f} (the Naive Forecaster's round-to-round spread)", rn_ub, c["sd_entry"], c["sd_skill"], sd_base, "normal", {}),
                ("Strategic: divergent variants (entry noise x2 on extra entries)", c["sd_round"], c["sd_entry"], c["sd_skill"], sd_base, "normal", {"extra_noise": 2.0}),
                ("Strategic: entries split across three names", c["sd_round"], c["sd_entry"], c["sd_skill"], sd_base, "normal", {"names": 3})]
    sens = []
    for name, sr, se, ss, sb, dist, kw_ in variants:
        for delta in (-0.5, 0.5):
            sens.append(dict(variant=name, delta=delta, k=9, **policies(9, delta, sr, se, ss, sb, dist, **kw_)))
    sens = pd.DataFrame(sens)
    sens.to_csv(DERIVED / "rq4_sensitivity.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 250):
        print("\n[sensitivity at k = 9]")
        print(sens.to_string(index=False))
    fc, dt = sens[sens["delta"] == -0.5], sens[sens["delta"] == 0.5]
    model_rows = ~fc["variant"].str.startswith("Strategic")
    fm, dm = fc[model_rows], dt[model_rows.to_numpy()]
    fit_row = fc[fc["variant"].str.contains("luck-share")].iloc[0]
    strat = fc[~model_rows]
    record("sens", n_variants=int(model_rows.sum()) - 1, sq_min=fm["status quo"].min(), sq_max=fm["status quo"].max(),
           single_min=fm["single entry"].min(), single_max=fm["single entry"].max(),
           corr_min=fm["corrected"].min(), corr_max=fm["corrected"].max(),
           corr_excess_max=(fm["corrected"] - fm["single entry"]).max(),
           corr_det_loss_max=(dm["single entry"].to_numpy() - dm["corrected"].to_numpy()).max(),
           held_min=fm["held-out round"].min(), held_max=fm["held-out round"].max(),
           excess_min=(fm["status quo"] - fm["single entry"]).min(), excess_max=(fm["status quo"] - fm["single entry"]).max(),
           fit_sq=fit_row["status quo"], fit_single=fit_row["single entry"],
           all_held_below_single=bool((fm["held-out round"] < fm["single entry"]).all()),
           # Added after registration: break-even cost ratio (false claim : missed lead) vs the status quo,
           # range across the model variations
           **{f"c_{tag}_{b}": f((dm["status quo"].to_numpy() - dm[col].to_numpy()) / (fm["status quo"].to_numpy() - fm[col].to_numpy()))
              for tag, col in (("corr", "corrected"), ("prim", "primary (prior round)"), ("held", "held-out round"))
              for b, f in (("lo", np.min), ("hi", np.max))},
           all_prim_below_single=bool((fm["primary (prior round)"] < fm["single entry"]).all()),
           div_sq=strat.iloc[0]["status quo"], div_corr=strat.iloc[0]["corrected"], div_prim=strat.iloc[0]["primary (prior round)"],
           div_held=strat.iloc[0]["held-out round"], split_sq=strat.iloc[1]["status quo"], split_corr=strat.iloc[1]["corrected"],
           split_prim=strat.iloc[1]["primary (prior round)"], split_held=strat.iloc[1]["held-out round"])

    # Added after registration: how much is a rank worth as evidence? Likelihood ratio of "best entry
    # ranked above the superforecaster median" for a team truly +d vs -d points from it.
    rng_l = np.random.default_rng(20261134)
    def p_above(k, delta, sd_b, n=N_SIM, skill=None, rnd=None):
        skill = c["sd_skill"] if skill is None else skill
        rnd = c["sd_round"] if rnd is None else rnd
        rounds = int(np.ceil(k / 3)); round_of = np.repeat(np.arange(rounds), 3)[:k]
        truth = np.full((n, k), float(delta))
        if k > 1:
            truth[:, 1:] -= skill * np.abs(rng_l.normal(0, 1, (n, k - 1)))
        o = truth + rng_l.normal(0, rnd, (n, rounds))[:, round_of] + rng_l.normal(0, c["sd_entry"], (n, k))
        return (o.max(1) > rng_l.normal(0, sd_b, n)).mean() if sd_b > 0 else (o.max(1) > 0).mean()
    lr = []
    for k in (1, 3, 9, 22):
        for d in (0.25, 0.5, 1.0, 1.5):
            for sb_name, sb in (("baseline error", sd_base), ("no baseline error", 0.0)):
                pp, pm = p_above(k, d, sb), p_above(k, -d, sb)
                lr.append(dict(k=k, gap=d, baseline=sb_name, p_better=pp, p_worse=pm, lr=pp / pm))
    lr = pd.DataFrame(lr)
    lr.to_csv(DERIVED / "rq4_likelihood_ratio.csv", index=False)
    g = lambda k, d, b: float(lr[(lr["k"] == k) & (lr["gap"] == d) & (lr["baseline"] == b)]["lr"].iloc[0])
    record("lr", k1_05=g(1, 0.5, "baseline error"), k9_05=g(9, 0.5, "baseline error"),
           k1_05_nb=g(1, 0.5, "no baseline error"), k9_05_nb=g(9, 0.5, "no baseline error"),
           k1_10=g(1, 1.0, "baseline error"), k9_10=g(9, 1.0, "baseline error"),
           k1_10_nb=g(1, 1.0, "no baseline error"), k9_10_nb=g(9, 1.0, "no baseline error"))
    # Added after registration: the same ratio under the curve-fitted skill spread and the round-noise upper bound
    lr_var = {}
    for vname, kwv in (("fit", {"skill": c["sd_skill"] * mult}), ("rn", {"rnd": rn_ub})):
        for k in (1, 9):
            lr_var[(vname, k)] = p_above(k, 0.5, sd_base, **kwv) / p_above(k, -0.5, sd_base, **kwv)
    lr_all = [g(1, 0.5, "baseline error"), g(9, 0.5, "baseline error")] + list(lr_var.values())
    record("lr", var_min=min(lr_all), var_max=max(lr_all))
    print(f"LR (+/-0.5) under variants: { {kk: round(v, 2) for kk, v in lr_var.items()} }")
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200):
        print("\n[likelihood ratio of a rank above the median]")
        print(lr.pivot_table(index=["k", "gap"], columns="baseline", values="lr").to_string())

    # Added after registration: the board under exact parity. The three teams with entries above the
    # median, with their actual entry counts, all exactly as good as superforecasters (best system at
    # the median), plus the shared error of the human baseline. Compare with what the board shows.
    lb = pd.read_csv(DATA / "leaderboards" / "lb_tournament.csv")
    lbe = lb[lb["Team"] != BENCHMARK]
    sup_pub = float(lb[lb["Model"].str.startswith("Superforecaster")]["Overall"].iloc[0])
    top_teams = lbe[lbe["Overall"] > sup_pub]["Team"].unique()
    ks = [int((lbe["Team"] == tm).sum()) for tm in top_teams]
    full = pd.read_csv(DERIVED / "h3_entry_full.csv")
    sup_rep = float(pd.read_csv(DERIVED / "h3_threshold.csv", index_col=0).iloc[0, 0])
    obs_top, obs_n = float(full["bi"].max() - sup_rep), int((full["bi"] > sup_rep).sum())
    rng_p = np.random.default_rng(20261135)
    n = N_SIM
    ents = []
    for k in ks:
        rounds = int(np.ceil(k / 3)); round_of = np.repeat(np.arange(rounds), 3)[:k]
        truth = np.zeros((n, k))
        if k > 1:
            truth[:, 1:] -= c["sd_skill"] * np.abs(rng_p.normal(0, 1, (n, k - 1)))
        ents.append(truth + rng_p.normal(0, c["sd_round"], (n, rounds))[:, round_of] + rng_p.normal(0, c["sd_entry"], (n, k)))
    ents = np.concatenate(ents, 1)
    base = rng_p.normal(0, sd_base, n)
    top = ents.max(1) - base
    n_above = (ents > base[:, None]).sum(1)
    ents_rn = []
    for k in ks:
        rounds = int(np.ceil(k / 3)); round_of = np.repeat(np.arange(rounds), 3)[:k]
        truth = np.zeros((n, k))
        if k > 1:
            truth[:, 1:] -= c["sd_skill"] * np.abs(rng_p.normal(0, 1, (n, k - 1)))
        ents_rn.append(truth + rng_p.normal(0, rn_ub, (n, rounds))[:, round_of] + rng_p.normal(0, c["sd_entry"], (n, k)))
    top_rn = np.concatenate(ents_rn, 1).max(1) - rng_p.normal(0, sd_base, n)
    record("parity", p_top_ge_obs_rn=float((top_rn >= float(full["bi"].max() - sup_rep)).mean()))
    record("parity", ks="/".join(map(str, ks)), top_mean=top.mean(), top_p05=np.percentile(top, 5),
           top_p95=np.percentile(top, 95), p_top_ge_obs=(top >= obs_top).mean(), nabove_mean=n_above.mean(),
           p_nabove_ge_obs=(n_above >= obs_n).mean(), obs_top=obs_top, obs_n=obs_n)
    # The leading team's true lead consistent with its observed edge: the values of delta for which the
    # observed top-entry edge lies inside the simulated 5-95% range of that team's best entry.
    k_lead = ks[int(np.argmax([lbe[lbe["Team"] == tm]["Overall"].max() for tm in top_teams]))]
    ok = []
    for d_ in np.round(np.arange(-2.0, 4.001, 0.05), 2):
        rounds = int(np.ceil(k_lead / 3)); round_of = np.repeat(np.arange(rounds), 3)[:k_lead]
        truth = np.full((n, k_lead), float(d_))
        if k_lead > 1:
            truth[:, 1:] -= c["sd_skill"] * np.abs(rng_p.normal(0, 1, (n, k_lead - 1)))
        x = (truth + rng_p.normal(0, c["sd_round"], (n, rounds))[:, round_of] + rng_p.normal(0, c["sd_entry"], (n, k_lead))).max(1) \
            - rng_p.normal(0, sd_base, n)
        lo_, hi_ = np.percentile(x, [5, 95])
        if lo_ <= obs_top <= hi_:
            ok.append(d_)
    record("parity", lead_k=k_lead, lead_lo=min(ok), lead_hi=max(ok))

    # Added after registration: evidence from the observed margin, not only from "ranked above". The
    # likelihood of the leading team's observed top-entry edge as a function of its true lead delta
    # (simulated density within +/-0.05 points of the observed edge), and the posterior probability of
    # a lead (delta > 0) under a flat prior on [-2, 4] and under sceptical normal priors centred on parity.
    rng_m = np.random.default_rng(20261137)
    grid = np.round(np.arange(-2.0, 4.001, 0.1), 2)
    met = json.load(open(DERIVED / "metrics.json"))
    lead_round_shift = met["anchor.20260830.overall.naive"]  # the leading entry's round (30 August 2026)
    obs_nets = {"loo": obs_top - met["anchor.slope.loo"] * lead_round_shift,
                "fe": obs_top - met["anchor.slope.fe"] * lead_round_shift}
    lik, lik_net = [], {kk: [] for kk in obs_nets}
    for d_ in grid:
        rounds = int(np.ceil(k_lead / 3)); round_of = np.repeat(np.arange(rounds), 3)[:k_lead]
        truth = np.full((n * 2, k_lead), float(d_))
        if k_lead > 1:
            truth[:, 1:] -= c["sd_skill"] * np.abs(rng_m.normal(0, 1, (n * 2, k_lead - 1)))
        x = (truth + rng_m.normal(0, c["sd_round"], (n * 2, rounds))[:, round_of]
             + rng_m.normal(0, c["sd_entry"], (n * 2, k_lead))).max(1) - rng_m.normal(0, sd_base, n * 2)
        lik.append(np.mean(np.abs(x - obs_top) < 0.05) / 0.1)
        for kk, ov in obs_nets.items():
            lik_net[kk].append(np.mean(np.abs(x - ov) < 0.05) / 0.1)
    lik = np.array(lik)
    at = lambda d_: lik[int(np.argmin(np.abs(grid - d_)))]
    def post(prior, L=None):
        # Revised (third audit): the delta = 0 cell spans -0.05 to +0.05, so half of its
        # mass counts towards a lead.
        L = lik if L is None else np.asarray(L)
        w = L * prior; w = w / w.sum()
        return float(w[grid > 0].sum() + 0.5 * w[np.isclose(grid, 0)].sum())
    priors = (np.ones_like(grid), np.exp(-0.5 * grid**2), np.exp(-0.5 * (grid / 0.5) ** 2))
    flat, sd1, sd05 = (post(pr_) for pr_ in priors)
    net_post = {kk: [post(pr_, L) for pr_ in priors] for kk, L in lik_net.items()}
    record("post", net_min=min(min(v) for v in net_post.values()), net_max=max(max(v) for v in net_post.values()),
           shift_loo=met["anchor.slope.loo"] * lead_round_shift, shift_fe=met["anchor.slope.fe"] * lead_round_shift)
    print(f"Posterior with the estimated round shift netted: {net_post}")
    pd.DataFrame({"delta": grid, "likelihood": lik}).to_csv(DERIVED / "lead_likelihood.csv", index=False)
    record("post", lr_margin=at(0.5) / at(-0.5), mle=float(grid[lik.argmax()]), p_flat=flat, p_n1=sd1, p_n05=sd05)
    print(f"Leading team, from its observed margin: LR(+0.5 vs -0.5) = {at(0.5) / at(-0.5):.2f}; most likely lead "
          f"{grid[lik.argmax()]:+.1f}; P(lead > 0) = {flat:.2f} (flat), {sd1:.2f} (N(0,1)), {sd05:.2f} (N(0,0.5))")
    print(f"\nLeading team (k = {k_lead}): true lead consistent with +{obs_top:.2f} (90% band): {min(ok):+.2f} to {max(ok):+.2f}")
    print(f"\n[exact parity, teams with {ks} entries] top entry {top.mean():+.2f} points above the median on average "
          f"(90% {np.percentile(top, 5):+.2f} to {np.percentile(top, 95):+.2f}); P(top >= observed {obs_top:+.2f}) = "
          f"{(top >= obs_top).mean():.2f}; entries above {n_above.mean():.1f} (P(>= {obs_n}) = {(n_above >= obs_n).mean():.2f})")

    # Added after registration: out-of-sample check of the calibration against a quantity the
    # calibration did not use. H2 observed that a team's round-t best variant keeps 48% of its
    # advantage over its teammates in round t+1 (external teams, k = 2-3). Predict that share.
    rng_v = np.random.default_rng(20261122)
    pr = pd.read_csv(DERIVED / "h2_round_pairs.csv")
    ks = pr[pr["external"]]["k"].value_counts(normalize=True)
    def retention(sd_skill, n=400_000):
        num = den = 0.0
        for k, w in ks.items():
            skill = np.zeros((n, k))
            if sd_skill > 0:
                skill[:, 1:] = -np.abs(rng_v.normal(0, sd_skill, (n, k - 1)))
            t0 = skill + rng_v.normal(0, cal["sd_entry_h1"], (n, k))
            t1 = skill + rng_v.normal(0, cal["sd_entry_h1"], (n, k))
            b = t0.argmax(1)
            adv0 = t0[np.arange(n), b] - t0.mean(1)
            adv1 = t1[np.arange(n), b] - t1.mean(1)
            num += w * adv1.mean(); den += w * adv0.mean()
        return num / den
    pred_eq, pred_het = retention(0.0), retention(cal["sd_skill_h1"])
    record("cal", pred_retention_equal=pred_eq, pred_retention_hetero=pred_het)
    print(f"Predicted next-round retention of the best variant's edge: equal skill {pred_eq:.2f}, "
          f"heterogeneous {pred_het:.2f} (observed external H2: see h2.ext.retained)")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        for scen in out["scenario"].unique():
            print(f"\n[{scen}]")
            print(out[out["scenario"] == scen].drop(columns="scenario").to_string(index=False))


if __name__ == "__main__":
    main()
