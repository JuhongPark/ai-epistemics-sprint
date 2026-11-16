"""H3 (pre-registered): does selection-corrected scoring change who is "above superforecasters"?

Threshold: the superforecaster median's score replicated with the same pipeline (68.19; the
published board shows 68.1). Using the replicated threshold keeps both sides of the comparison
on the same scale (revised after review; the first draft compared replicated entries to the
rounded published 68.1).

Status quo: a team counts as above if ANY of its entries is above the threshold.
Corrected:
  (a) team mean of the team's entries' rescaled difficulty-adjusted Brier;
  (b) held-out: split question ids (all rounds) into halves A/B; pick each team's best entry on
      A and score it on B. Reported on matched halves (count on A vs count on B), 200 splits.
Registered: bootstrap CI on the counts (questions resampled with replacement, 500 replicates).
Registered robustness: dataset-only and market-only versions.
Exploratory: leaderboard-wide regression of the top-10 entries from half A to half B.
Published "Supers > Forecaster?" p-values: reverse comparisons (1 - p) with Holm correction.
"""

import numpy as np
import pandas as pd
from scipy import sparse

from common import DERIVED, DATA, BENCHMARK, load_rows, preprocess, load_qfe, brier_index, record

RNG = np.random.default_rng(20261116)
N_SPLITS = 200
N_BOOT = 500
SUPER_NAME = "Superforecaster median forecast"
COLS = ["file", "model", "morg", "org", "due", "id", "source", "direction", "res_date",
        "resolved_to", "resolved", "forecast", "imputed"]


def build():
    df = preprocess(load_rows(COLS))
    df["qfe"] = df["question_pk"].map(load_qfe())
    df = df[df["qfe"].notna()]
    keep = (df["org"] != BENCHMARK) | df["model"].isin(["Always 0.5", SUPER_NAME])
    df = df[keep].copy()
    df["adj"] = df["brier"] - df["qfe"]
    df["qid"] = df["source"].astype(str) + "_" + df["id"].astype(str)
    ent = df[["model_pk", "org", "model"]].drop_duplicates("model_pk").reset_index(drop=True)
    e_ix = pd.Series(range(len(ent)), index=ent["model_pk"])
    qids = pd.Index(df["qid"].unique())
    mats = {}
    for qt in ["dataset", "market"]:
        d = df[df["qtype"] == qt]
        r, c = e_ix[d["model_pk"]].to_numpy(), qids.get_indexer(d["qid"])
        shape = (len(ent), len(qids))
        mats[qt] = (sparse.csr_matrix((d["adj"].to_numpy(), (r, c)), shape=shape),
                    sparse.csr_matrix((np.ones(len(d)), (r, c)), shape=shape))
    return ent, qids, mats


def score(ent, mats, w):
    """Brier Index per entry for question weights w (len = n qids). Returns (bi, overall_rescaled)."""
    means = {}
    for qt, (S, C) in mats.items():
        num, den = S @ w, C @ w
        with np.errstate(invalid="ignore", divide="ignore"):
            means[qt] = np.where(den > 0, num / den, np.nan)
    overall = (means["dataset"] + means["market"]) / 2
    a05 = np.where(ent["model"].to_numpy() == "Always 0.5")[0][0]
    overall = overall + 0.25 - overall[a05]
    out = {"overall": overall, "bi": brier_index(pd.Series(overall)).to_numpy()}
    for qt in means:
        m = means[qt] + 0.25 - means[qt][a05]
        out[f"bi_{qt}"] = brier_index(pd.Series(m)).to_numpy()
    return out


def team_counts(ent, s, key="bi", sup=None):
    ext = (ent["org"] != BENCHMARK).to_numpy() & ~np.isnan(s[key])
    sup = s[key][np.where(ent["model"].to_numpy() == SUPER_NAME)[0][0]] if sup is None else sup
    d = pd.DataFrame({"org": ent["org"][ext], "bi": s[key][ext],
                      "overall": s["overall"][ext]})
    status_quo = d.groupby("org")["bi"].max()
    if key == "bi":
        tmean = pd.Series(brier_index(d.groupby("org")["overall"].mean()), name="tm")
    else:  # per-type: mean of per-type index is adequate for a robustness count
        tmean = d.groupby("org")["bi"].mean()
    return (status_quo > sup).sum(), (tmean > sup).sum(), sup, status_quo, tmean


def main():
    ent, qids, mats = build()
    w_full = np.ones(len(qids))
    s = score(ent, mats, w_full)
    sq, tm, sup, best, tmean = team_counts(ent, s)
    full = pd.DataFrame({"model_pk": ent["model_pk"], "org": ent["org"], "model": ent["model"],
                         "overall": s["overall"], "bi": s["bi"]})
    full = full[full["org"] != BENCHMARK].dropna(subset=["bi"])
    full.to_csv(DERIVED / "h3_entry_full.csv", index=False)
    above_teams = best[best > sup].index.tolist()
    print(f"Replicated superforecaster threshold: {sup:.2f}")
    print(f"Status quo: {(full['bi'] > sup).sum()} entries from {sq} teams above | team mean: {tm} teams above")
    for org in above_teams:
        print(f"    {org}: best {best[org]:.2f} | team mean {tmean[org]:.2f} | entries {(full['org'] == org).sum()}")
    print(f"  (vs rounded published 68.1: {(full['bi'] > 68.1).sum()} entries from "
          f"{full[full['bi'] > 68.1]['org'].nunique()} teams)")
    record("h3", sup=sup, sq_entries=(full["bi"] > sup).sum(), sq_teams=sq, tm_teams=tm,
           tm_min=tmean[above_teams].min(), tm_max=tmean[above_teams].max(),
           rounded_entries=(full["bi"] > 68.1).sum(), rounded_teams=full[full["bi"] > 68.1]["org"].nunique(),
           fourth_team_bi=sorted(best[best > 68.1].tolist())[0])

    # Registered: bootstrap CI on counts (questions resampled with replacement)
    bs = []
    for _ in range(N_BOOT):
        w = np.bincount(RNG.integers(0, len(qids), len(qids)), minlength=len(qids)).astype(float)
        sb = score(ent, mats, w)
        a, b, sup_b, *_ = team_counts(ent, sb)
        c, d_, *_ = team_counts(ent, sb, sup=sup)  # threshold held at the full-sample value
        bs.append((a, b, c, d_, sup_b))
    bs = np.array(bs)
    q = lambda x: f"[{np.percentile(x, 2.5):.0f}, {np.percentile(x, 97.5):.0f}] (mean {x.mean():.2f})"
    print(f"Question bootstrap ({N_BOOT}), superforecaster resampled too: status quo {q(bs[:, 0])} | team mean {q(bs[:, 1])}")
    print(f"  superforecaster score across bootstrap: 95% [{np.percentile(bs[:, 4], 2.5):.2f}, {np.percentile(bs[:, 4], 97.5):.2f}]")
    print(f"Question bootstrap, threshold fixed at {sup:.2f}: status quo {q(bs[:, 2])} | team mean {q(bs[:, 3])}")
    pct = lambda x, p: float(np.percentile(x, p))
    record("h3.boot", sq_lo=pct(bs[:, 0], 2.5), sq_hi=pct(bs[:, 0], 97.5), tm_lo=pct(bs[:, 1], 2.5),
           tm_hi=pct(bs[:, 1], 97.5), sq_fixed_lo=pct(bs[:, 2], 2.5), sq_fixed_hi=pct(bs[:, 2], 97.5),
           tm_fixed_lo=pct(bs[:, 3], 2.5), tm_fixed_hi=pct(bs[:, 3], 97.5),
           sup_lo=pct(bs[:, 4], 2.5), sup_hi=pct(bs[:, 4], 97.5))

    # Held-out on matched halves; threshold fixed at the full-sample superforecaster score
    ext = (ent["org"] != BENCHMARK).to_numpy()
    cnt_a, cnt_b, from_new, top_a, top_b, per = [], [], [], [], [], []
    for _ in range(N_SPLITS):
        mask = RNG.random(len(qids)) < 0.5
        sa, sb = score(ent, mats, mask.astype(float)), score(ent, mats, (~mask).astype(float))
        d = pd.DataFrame({"org": ent["org"], "a": sa["bi"], "b": sb["bi"]})[ext].dropna()
        pick = d.loc[d.groupby("org")["a"].idxmax()].set_index("org")
        cnt_a.append((pick["a"] > sup).sum()); cnt_b.append((pick["b"] > sup).sum())
        from_new.append(((pick["b"] > sup) & ~pick.index.isin(above_teams)).sum())
        for org in above_teams:
            per.append(dict(org=org, bi_a=pick.loc[org, "a"], bi_b=pick.loc[org, "b"]))
        top = d.nlargest(10, "a"); top_a.append(top["a"].mean()); top_b.append(top["b"].mean())
    pt = pd.DataFrame(per).groupby("org")[["bi_a", "bi_b"]].mean()
    pt.to_csv(DERIVED / "h3_heldout_teams.csv")
    record("h3.halves", count_a=np.mean(cnt_a), count_b=np.mean(cnt_b), count_b_min=min(cnt_b),
           count_b_max=max(cnt_b), from_new=np.mean(from_new), top10_a=np.mean(top_a), top10_b=np.mean(top_b),
           top10_drop=np.mean(top_a) - np.mean(top_b))
    record("h3.heldout", **{org.replace(" ", "_"): v for org, v in pt["bi_b"].items()})
    print(f"Held-out (matched halves, {N_SPLITS} splits): teams above on selection half {np.mean(cnt_a):.2f} -> "
          f"held-out half {np.mean(cnt_b):.2f} (range {min(cnt_b)}-{max(cnt_b)}); "
          f"of which teams not above in full sample: {np.mean(from_new):.2f}")
    print("  status-quo teams, best entry on selection half vs held-out half:")
    print(pt.round(2).to_string())
    print(f"Exploratory: top-10 entries on half A: {np.mean(top_a):.2f} -> half B {np.mean(top_b):.2f} "
          f"(drop {np.mean(top_a) - np.mean(top_b):.2f})")

    # Added after registration (not pre-registered): continuous held-out gaps instead of knife-edge
    # counts, at 50% and 80% selection fractions, with the range across splits; separate RNG.
    rng_x = np.random.default_rng(20261123)
    cont = []
    for frac in (0.5, 0.8):
        per_x, top_ax, top_bx, cnt_pub, cnt_rep = [], [], [], [], []
        for _ in range(N_SPLITS):
            mask = rng_x.random(len(qids)) < frac
            sa, sb = score(ent, mats, mask.astype(float)), score(ent, mats, (~mask).astype(float))
            d = pd.DataFrame({"org": ent["org"], "a": sa["bi"], "b": sb["bi"]})[ext].dropna()
            pick = d.loc[d.groupby("org")["a"].idxmax()].set_index("org")
            for org in above_teams:
                per_x.append(dict(org=org, a=pick.loc[org, "a"], b=pick.loc[org, "b"]))
            top = d.nlargest(10, "a"); top_ax.append(top["a"].mean()); top_bx.append(top["b"].mean())
            cnt_rep.append((pick.loc[above_teams, "b"] > sup).sum()); cnt_pub.append((pick.loc[above_teams, "b"] > 68.1).sum())
        px = pd.DataFrame(per_x)
        for org, g in px.groupby("org"):
            cont.append(dict(frac=frac, org=org, sel=g["a"].mean(), held=g["b"].mean(),
                             held_lo=g["b"].quantile(0.025), held_hi=g["b"].quantile(0.975),
                             gap_to_super=g["b"].mean() - sup))
        record(f"h3.frac{int(frac * 100)}", top10_a=np.mean(top_ax), top10_b=np.mean(top_bx),
               top10_drop=np.mean(top_ax) - np.mean(top_bx), above_rep=np.mean(cnt_rep), above_pub=np.mean(cnt_pub))
    cont = pd.DataFrame(cont)
    cont.to_csv(DERIVED / "h3_continuous.csv", index=False)
    print("Continuous held-out scores of the status-quo teams' best entries (mean and 95% range across splits):")
    print(cont.round(2).to_string(index=False))

    # Registered robustness: dataset-only and market-only
    for qt in ["dataset", "market"]:
        a, b, sup_q, *_ = team_counts(ent, s, key=f"bi_{qt}")
        record(f"h3.rob_{qt}", sup=sup_q, sq=a, tm=b)
        print(f"Robustness [{qt} only]: superforecaster {sup_q:.2f} | status quo {a} teams | team mean {b} teams")

    lb = pd.read_csv(DATA / "leaderboards" / "lb_tournament.csv")
    e = lb[lb["Team"] != BENCHMARK].copy()
    e["p"] = pd.to_numeric(e["p-val Supers > Forecaster?"].astype(str).str.replace("<", ""), errors="coerce")
    abv = e[e["Overall"] > 68.1]
    rev = (1 - abv["p"]).sort_values()
    m = len(e)
    holm = [min(1, p * (m - i)) for i, p in enumerate(rev)]
    record("h3.pub", n_above=len(abv), p_min=abv["p"].min(), p_max=abv["p"].max(), rev_min=rev.min(),
           rev_max=rev.max(), holm_min=min(holm), n_tested=m)
    print(f"Published 'Supers > Forecaster?' p for the {len(abv)} entries above: {sorted(abv['p'].round(2).tolist())}; "
          f"reverse (1 - p): {rev.round(2).tolist()}; Holm over {m}: {np.round(holm, 2).tolist()}")
    pd.DataFrame(dict(status_quo=bs[:, 0], team_mean=bs[:, 1], status_quo_fixed=bs[:, 2],
                      team_mean_fixed=bs[:, 3], super_boot=bs[:, 4])).to_csv(DERIVED / "h3_bootstrap_counts.csv", index=False)
    pd.Series(dict(super_replicated=sup)).to_csv(DERIVED / "h3_threshold.csv")


if __name__ == "__main__":
    main()
