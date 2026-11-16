"""Noise on one consistent basis (added after a second code audit).

The simulation and the detectability calculation previously took entry noise from ForecastBench's
published CIs (which resample each question-horizon independently, stratified by round and source)
and the superforecaster score's noise from the registered H3 bootstrap (questions resampled across
the whole board, unstratified). Here both come from one bootstrap: clusters are questions within a
round (all horizons of a question in a round together), resampled with replacement within strata of
round x source; every entry, the superforecaster median and always-0.5 are re-scored on each
replicate with the leaderboard pipeline (04_h3_consequence.score).

Outputs (data/derived/noise_bootstrap.json, metrics "noise.*"):
  sd_super      SD of the superforecaster median's Brier Index across replicates
  sd_entry_med  median SD of external single-round entries
  sup_lo/hi     2.5% and 97.5% quantiles of the superforecaster score
  per-entry SDs for the external entries (noise_bootstrap_entries.csv)
The registered H3 interval (04_h3_consequence.py) is unchanged and still reported for H3.
"""

import importlib
import json
import warnings

import numpy as np
import pandas as pd
from scipy import sparse

from common import DATA, DERIVED, BENCHMARK, load_rows, preprocess, load_qfe, record

warnings.filterwarnings("ignore", "invalid value encountered in sqrt")
h3 = importlib.import_module("04_h3_consequence")
RNG = np.random.default_rng(20261133)
N_BOOT = 500
SUP = "Superforecaster median forecast"


def build_clustered():
    df = preprocess(load_rows(h3.COLS))
    df["qfe"] = df["question_pk"].map(load_qfe())
    df = df[df["qfe"].notna()]
    keep = (df["org"] != BENCHMARK) | df["model"].isin(["Always 0.5", SUP])
    df = df[keep].copy()
    df["adj"] = df["brier"] - df["qfe"]
    df["cid"] = df["due"].astype(str) + "_" + df["source"].astype(str) + "_" + df["id"].astype(str)
    df["stratum"] = df["due"].astype(str) + "_" + df["source"].astype(str)
    ent = df[["model_pk", "org", "model"]].drop_duplicates("model_pk").reset_index(drop=True)
    e_ix = pd.Series(range(len(ent)), index=ent["model_pk"])
    cids = pd.Index(df["cid"].unique())
    mats = {}
    for qt in ["dataset", "market"]:
        d = df[df["qtype"] == qt]
        r, c = e_ix[d["model_pk"]].to_numpy(), cids.get_indexer(d["cid"])
        shape = (len(ent), len(cids))
        mats[qt] = (sparse.csr_matrix((d["adj"].to_numpy(), (r, c)), shape=shape),
                    sparse.csr_matrix((np.ones(len(d)), (r, c)), shape=shape))
    strata = df.drop_duplicates("cid").set_index("cid").loc[cids, "stratum"].to_numpy()
    return ent, cids, mats, strata


def main():
    ent, cids, mats, strata = build_clustered()
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    full = h3.score(ent, mats, np.ones(len(cids)))["bi"]
    reps = []
    for _ in range(N_BOOT):
        w = np.zeros(len(cids))
        for g in groups:
            w[g] = np.bincount(RNG.integers(0, len(g), len(g)), minlength=len(g))
        reps.append(h3.score(ent, mats, w)["bi"])
    reps = np.array(reps)
    sd = np.nanstd(reps, axis=0, ddof=1)
    i_sup = np.where(ent["model"].to_numpy() == SUP)[0][0]

    idx = pd.read_csv(DATA / "processed" / "index.csv")
    n_rounds = idx[idx["organization"] != BENCHMARK].groupby(["organization", "model"])["due"].nunique()
    e = ent.assign(bi=full, sd=sd)
    e = e[e["org"] != BENCHMARK].dropna(subset=["bi"])
    e["rounds"] = [n_rounds.get((o, m), np.nan) for o, m in zip(e["org"], e["model"])]
    e.to_csv(DERIVED / "noise_bootstrap_entries.csv", index=False)
    single = e[e["rounds"] == 1]
    out = dict(sd_super=float(sd[i_sup]), sd_entry_med=float(single["sd"].median()), n_single=len(single),
               sup=float(full[i_sup]), sup_lo=float(np.percentile(reps[:, i_sup], 2.5)),
               sup_hi=float(np.percentile(reps[:, i_sup], 97.5)), n_boot=N_BOOT, n_strata=len(groups))
    json.dump(out, open(DERIVED / "noise_bootstrap.json", "w"), indent=1)
    record("noise", **out)
    print(f"Clustered, stratified bootstrap ({N_BOOT}): superforecaster SD {out['sd_super']:.3f} "
          f"(95% {out['sup_lo']:.2f}-{out['sup_hi']:.2f}); median SD of {len(single)} single-round external "
          f"entries {out['sd_entry_med']:.3f}")


if __name__ == "__main__":
    main()
