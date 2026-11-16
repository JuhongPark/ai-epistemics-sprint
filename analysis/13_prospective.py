"""Prospective test of PROSPECTIVE.md, and an out-of-time rehearsal on the 14 November 2026 snapshot.

Modes
  13_prospective.py                     rehearsal (exploratory) on data/ (the 14 November snapshot)
  13_prospective.py --snapshot DIR      the registered test on a later snapshot; checks the trigger first
  13_prospective.py --snapshot DIR --selftest
                                        exercises the registered code path without the trigger or the
                                        run-once lock; prints a warning and records nothing

Building a later snapshot. Download the newer processed forecast sets into a new, empty directory, extract
them there, run `python -I analysis/feasibility/index.py <extracted root> DIR/processed`, and save the new
tournament question fixed effects as DIR/ref/qfe_tournament.json. The registered test runs only when every
frozen entry with a round in June-August 2026 has at least 200 newly resolved questions, or from 1 March
2027. Before that it prints how many new questions each such entry has and stops; it computes no scores.
It runs once: the result is written to DIR/prospective_result.json, and a second run stops.

Implementation details fixed on 15 November 2026, before any newer data was downloaded:
  - Scale. Each board uses one constant shift (always-0.5 scores 0.25 over the whole board), as the
    leaderboard does. A shift computed on the new questions alone would undo the difficulty adjustment.
  - Frozen board. The board as ForecastBench builds it on the freeze date: rounds due at least 50 days
    earlier, and rounds whose Naive Forecaster has at least 50 resolved market and 225 resolved dataset
    questions (leaderboard main.py: min_days, min_num_market_questions, MIN_NUM_DATASET_QUESTIONS).
  - New questions of a frozen entry. Its forecasts in the rounds on the frozen board, on questions that
    resolve after the freeze date. The overall score is the mean of the dataset and market scores, so an
    entry needs both types among its new questions; other entries are dropped and reported.
  - Random comparison (prediction 1). Ten entries drawn from the frozen entries that have new questions,
    1,000 draws. The share of draws whose mean change is at or below the top ten's is reported as a
    reference p-value.
  - Prediction 2 uses the registered threshold 68.19. The superforecaster median re-scored in the later
    snapshot is reported next to it, because the later difficulty effects may move the scale.

Rehearsal. Freeze the board at T = 1 September 2026, as ForecastBench would have built it then (questions
resolved by T), and check predictions 1 and 2 on questions that resolved between T and 14 November.
The other freeze dates (1 and 15 August, 15 September, 1 October) are robustness checks. From 15
September on, most of the top ten come from the 7 June round, whose dataset questions next resolve on 4
December (180-day horizon), so they have no new dataset questions and drop out; those dates are reported
but say little about the top ten. The threshold for prediction 2 is the
superforecaster median on the board at T. This reuses questions the report already analysed (the
board-wide estimates resample them), so it rehearses the procedure; it is not a test. Prediction 3 cannot
be rehearsed independently: team-rounds after T are already in the H1 sample. Their luck share is printed
for information only.
"""

import argparse
import importlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, DERIVED, BENCHMARK, ROUND_COLS, preprocess, load_qfe, brier_index, record

SNAP = "2026-11-14"
REG_SUP = 68.19          # registered threshold (PROSPECTIVE.md)
REG_GAP = 0.45           # registered 14 November mean gap of the seven
TRIGGER_N, TRIGGER_DATE = 200, "2027-03-01"
MIN_DAYS, MIN_MKT, MIN_DS = 50, 50, 225
N_DRAW = 1000
SUP = "Superforecaster median forecast"
REHEARSAL_T = ["2026-09-01", "2026-08-01", "2026-08-15", "2026-09-15", "2026-10-01"]
RNG = np.random.default_rng(20261115)


def load(data_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / "processed" / "long.parquet", columns=ROUND_COLS)
    idx = pd.read_csv(data_dir / "processed" / "index.csv", usecols=["file", "lb"])
    df = preprocess(df.merge(idx, on="file", how="left"))
    df["qfe"] = df["question_pk"].map(load_qfe(data_dir / "ref" / "qfe_tournament.json"))
    df["due"] = df["due"].astype(str).str[:10]
    df["res"] = df["res_date"].astype(str).str[:10]
    df["adj"] = df["brier"] - df["qfe"]
    return df


def board_rounds(df, freeze):
    """Rounds on the board built on `freeze`, and their rows resolved by then."""
    cut = (pd.Timestamp(freeze) - pd.Timedelta(days=MIN_DAYS)).strftime("%Y-%m-%d")
    d = df[(df["due"] <= cut) & (df["res"] <= freeze)]
    nv = d[d["model"] == "Naive Forecaster"].groupby(["due", "qtype"])["question_pk"].nunique().unstack()
    nv = nv.reindex(columns=["dataset", "market"]).fillna(0)
    rounds = sorted(nv[(nv["market"] >= MIN_MKT) & (nv["dataset"] >= MIN_DS)].index)
    return rounds, d[d["due"].isin(rounds) & d["qfe"].notna()]


def shift_of(d):
    a = d[d["model"] == "Always 0.5"].groupby("qtype")["adj"].mean()
    return 0.25 - (a["dataset"] + a["market"]) / 2


def scores(d, shift):
    g = d.groupby(["model_pk", "qtype"])["adj"].mean().unstack().reindex(columns=["dataset", "market"])
    return brier_index((g["dataset"] + g["market"]) / 2 + shift)


def meta(df):
    return df.drop_duplicates("model_pk").set_index("model_pk")[["org", "model"]]


def first_round(d):
    return d.groupby("model_pk")["due"].min()


def predictions(old, new, n_new, info, sup_thr, gap_ref=None, rnd=None):
    """old/new: Brier Index per frozen external entry (new may be NaN). rnd: each entry's first round on the
    frozen board, for the round-matched comparison (not registered: each top-ten entry is compared with the
    mean change of the other entries of its own round, so a round whose new questions are hard or easy for
    everyone does not count as selection). Returns a dict of results."""
    has = new.notna()
    top = old.nlargest(10).index
    top_kept = [e for e in top if has[e]]
    elig = old.index[has]
    chg = (new - old)[elig]
    top_chg = chg[top_kept].mean()
    draws = (np.array([chg.iloc[RNG.choice(len(chg), 10, replace=False)].mean() for _ in range(N_DRAW)])
             if len(chg) >= 10 else np.full(N_DRAW, np.nan))
    matched = []
    if rnd is not None:
        for e in top_kept:
            pool = [x for x in elig if x not in top_kept and rnd.get(x) == rnd.get(e)]
            if pool:
                matched.append(chg[pool].mean() - chg[e])
    above = old.index[old > sup_thr]
    above_kept = [e for e in above if has[e]]
    stay = int((new[above_kept] > sup_thr).sum())
    gap_old = float((old[above] - sup_thr).mean()) if gap_ref is None else gap_ref
    gap_new = float((new[above_kept] - sup_thr).mean()) if above_kept else np.nan
    return dict(n_frozen=len(old), n_elig=len(elig), n_dropped=int((~has).sum()),
                n_new_med=float(n_new.reindex(elig).median()),
                top_n=len(top_kept), top_old=float(old[top_kept].mean()), top_new=float(new[top_kept].mean()),
                top_chg=float(top_chg), rand_chg=float(draws.mean()), diff=float(draws.mean() - top_chg),
                p_ref=float((draws <= top_chg).mean()),
                diff_matched=float(np.mean(matched)) if matched else np.nan, n_matched=len(matched),
                top_rounds=int(rnd[top_kept].nunique()) if rnd is not None and top_kept else 0,
                sup=float(sup_thr), n_above=len(above), above_kept=len(above_kept), stay=stay,
                gap_old=gap_old, gap_new=gap_new,
                above_teams=int(info.loc[above, "org"].nunique()) if len(above) else 0)


def describe(tag, r):
    print(f"[{tag}] frozen external entries {r['n_frozen']:.0f}; with new questions of both types "
          f"{r['n_elig']:.0f} (dropped {r['n_dropped']:.0f}); median new questions per entry {r['n_new_med']:.0f}")
    print(f"  P1  top ten (n={r['top_n']:.0f} with new questions): {r['top_old']:.2f} -> {r['top_new']:.2f} "
          f"(change {r['top_chg']:+.2f}); random ten change {r['rand_chg']:+.2f}; "
          f"difference {r['diff']:+.2f} (share of random tens at or below the top ten: {r['p_ref']:.3f})")
    if r.get("n_matched"):
        print(f"      round-matched (secondary analysis, PROSPECTIVE.md addendum): top ten from {r['top_rounds']:.0f} rounds; difference from "
              f"the other entries of the same round {r['diff_matched']:+.2f} ({r['n_matched']:.0f} entries)")
    print(f"  P2  above the superforecaster median ({r['sup']:.2f}): {r['n_above']:.0f} entries from "
          f"{r['above_teams']:.0f} teams; {r['stay']:.0f} of {r['above_kept']:.0f} stay above on new questions; "
          f"mean gap {r['gap_old']:+.2f} -> {r['gap_new']:+.2f}")


def round_level(old, new, rnd, nr, naive=None):
    """Exploratory: the round holding most of the top ten, against the other rounds whose new questions are
    of the same kinds (same set of dataset horizons, plus markets), entry-weighted. Returns its lead over
    them at the freeze and on new questions."""
    t = pd.DataFrame({"old": old, "new": new, "rnd": rnd.reindex(old.index)}).dropna()
    h = nr.assign(h=np.where(nr["qtype"] == "dataset", nr["question_pk"].str.rsplit("_", n=1).str[-1], "m"))
    kinds = h.groupby("due")["h"].agg(lambda x: ",".join(sorted(set(x))))
    top_rounds = rnd.reindex(old.nlargest(10).index)
    top_r = top_rounds.mode().iloc[0]
    peers = [r for r in kinds.index if r != top_r and kinds[r] == kinds.get(top_r)]
    a, b = t[t["rnd"] == top_r], t[t["rnd"].isin(peers)]
    if a.empty or b.empty:
        return dict(top_round=top_r, n_peers=0)
    lead_old, lead_new = a["old"].mean() - b["old"].mean(), a["new"].mean() - b["new"].mean()
    out = dict(top_round=top_r, top_in_round=int((top_rounds == top_r).sum()), n_top_round=len(a), n_peer_rounds=len(peers), n_peers=len(b),
               lead_old=float(lead_old), lead_new=float(lead_new), kept=float(lead_new / lead_old))
    if naive is not None:  # the Naive Forecaster's lead in the same rounds, peers weighted as above
        w = b["rnd"].value_counts()
        for c in ("old", "new"):
            out[f"naive_{c}"] = float(naive.loc[top_r, c] - (naive.loc[w.index, c] * w).sum() / w.sum())
    return out


def h1_luck_share(df, after, until):
    """Registered H1 procedure (02_h1_inflation) on external team-rounds due after `after`."""
    h1 = importlib.import_module("02_h1_inflation")
    d = df[(df["org"] != BENCHMARK) & (df["due"] > after) & (df["res"] <= until)]
    rows = []
    for org, due, entries, M in h1.team_round_matrices(d):
        infl, adv_a, adv_b = h1.split_half_inflation(M, h1.RNG)
        rows.append(dict(org=org, due=due, inflation=infl, adv_a=adv_a))
    tr = pd.DataFrame(rows)
    if tr.empty:
        return 0, 0, np.nan
    return len(tr), tr["org"].nunique(), float(tr["inflation"].mean() / tr["adv_a"].mean())


def rehearsal():
    df = load(DATA)
    info = meta(df)
    s_rounds, s_board = board_rounds(df, SNAP)
    # Check: the snapshot board rebuilt this way reproduces the registered scores (h3_entry_full.csv).
    s_board_all = df[df["due"].isin(s_rounds) & df["qfe"].notna()]  # incl. 44 early-resolved market rows
    shift = shift_of(s_board_all)
    chk = pd.read_csv(DERIVED / "h3_entry_full.csv").set_index("model_pk")["bi"]
    err = (scores(s_board_all, shift).reindex(chk.index) - chk).abs().max()
    print(f"Snapshot board rebuilt from the round rule: {len(s_rounds)} rounds; max |difference| from "
          f"h3_entry_full.csv = {err:.2e}")
    assert err < 1e-6, "board rule does not reproduce the snapshot scores"

    out = {}
    for T in REHEARSAL_T:
        rounds, bd = board_rounds(df, T)
        old_all = scores(bd, shift)
        sup_thr = float(old_all[info.loc[old_all.index, "model"].eq(SUP).to_numpy()].iloc[0])
        ext = old_all[(info.loc[old_all.index, "org"] != BENCHMARK).to_numpy()].dropna()
        nr = df[df["model_pk"].isin(ext.index) & df["due"].isin(rounds) & (df["res"] > T)
                & (df["res"] <= SNAP) & df["qfe"].notna()]
        new = scores(nr, shift).reindex(ext.index)
        n_new = nr.groupby("model_pk")["question_pk"].nunique()
        r = predictions(ext, new, n_new, info, sup_thr, rnd=first_round(bd))
        r["n_rounds"] = len(rounds)
        r["last_round"] = rounds[-1]
        nv = df[(df["model"] == "Naive Forecaster") & df["due"].isin(rounds) & df["qfe"].notna()]
        naive = pd.DataFrame({"old": scores(nv[nv["res"] <= T].assign(model_pk=nv["due"]), shift),
                              "new": scores(nv[(nv["res"] > T) & (nv["res"] <= SNAP)].assign(model_pk=nv["due"]), shift)})
        rl = round_level(ext, new, first_round(bd), nr, naive)
        r.update({f"rl_{k}": v for k, v in rl.items()})
        print(f"\nRehearsal, board frozen {T} ({len(rounds)} rounds, last {rounds[-1]}); new questions resolve "
              f"{T} to {SNAP}")
        describe(T, r)
        if rl.get("n_peers"):
            print(f"      round level (exploratory): the {rl['top_round']} round ({rl['n_top_round']} entries) led "
                  f"{rl['n_peers']} entries of {rl['n_peer_rounds']} rounds with the same kinds of new questions by "
                  f"{rl['lead_old']:.2f} at the freeze and {rl['lead_new']:.2f} on new questions "
                  f"(kept {rl['kept']:.0%}); the Naive Forecaster's lead in the same rounds "
                  f"{rl['naive_old']:+.2f} -> {rl['naive_new']:+.2f}")
        out[T] = r
        record(f"pseudo.t{T.replace('-', '')[4:]}",
               **{k: v for k, v in r.items() if k != "last_round"})
    n_tr, n_teams, share = h1_luck_share(df, REHEARSAL_T[0], SNAP)
    print(f"\nFor information (not independent of H1): external team-rounds due after {REHEARSAL_T[0]}: "
          f"{n_tr} from {n_teams} teams; luck share {share:.2f}")
    record("pseudo.h1", n=n_tr, teams=n_teams, share=share)
    pd.DataFrame(out).T.to_csv(DERIVED / "pseudo_prospective.csv")


def registered(snapdir: Path, selftest: bool):
    """The registered test. With selftest, the same code runs with the board frozen at the first rehearsal
    date and the current snapshot as the 'later' data, so its prediction-1 numbers must match the rehearsal's."""
    lock = snapdir / "prospective_result.json"
    if lock.exists() and not selftest:
        sys.exit(f"Already run; result in {lock}. PROSPECTIVE.md allows one run.")
    freeze = REHEARSAL_T[0] if selftest else SNAP
    cur = load(DATA)
    s_rounds, s_board = board_rounds(cur, freeze)
    if selftest:
        print(f"SELFTEST: board frozen {freeze}, current snapshot used as the later data; trigger and run-once "
              f"lock bypassed. Numbers below are NOT a result and are not recorded.")
        info0 = meta(cur)
        sc = scores(s_board, shift_of(cur[cur["qfe"].notna()]))
        frozen = sc[(info0.loc[sc.index, "org"] != BENCHMARK).to_numpy()].dropna()
    else:
        frozen = pd.read_csv(DERIVED / "h3_entry_full.csv").set_index("model_pk")["bi"]
    later = load(snapdir)
    nr = later[later["model_pk"].isin(frozen.index) & later["due"].isin(s_rounds) & (later["res"] > freeze)
               & later["qfe"].notna()]
    if selftest:
        nr = nr[nr["res"] <= SNAP]
    n_new = nr.groupby("model_pk")["question_pk"].nunique().reindex(frozen.index).fillna(0)
    summer = cur[cur["model_pk"].isin(frozen.index) & cur["due"].between("2026-06-01", "2026-08-31")
                 & cur["due"].isin(s_rounds)]["model_pk"].unique()
    short = n_new[summer][n_new[summer] < TRIGGER_N]
    today = date.today().isoformat()
    print(f"Trigger: {len(summer)} frozen entries with a June-August round; {len(short)} have fewer than "
          f"{TRIGGER_N} new resolved questions (minimum {n_new[summer].min():.0f}); today {today}")
    if len(short) and today < TRIGGER_DATE and not selftest:
        print("Trigger not met; no scores computed.")
        return

    info = pd.concat([meta(later), meta(cur)])
    info = info[~info.index.duplicated()]
    shift = shift_of(later[later["qfe"].notna()])  # the later board: every question with a published effect
    new = scores(nr, shift).reindex(frozen.index)
    sup_later = scores(later[(later["model"] == SUP) & later["qfe"].notna()], shift)
    r = predictions(frozen, new, n_new, info, REG_SUP, gap_ref=None if selftest else REG_GAP,
                    rnd=first_round(s_board))
    describe("selftest" if selftest else "registered", r)
    # Secondary analysis (b), added in the PROSPECTIVE.md addendum before any newer data: round level.
    nv_old = cur[(cur["model"] == "Naive Forecaster") & cur["due"].isin(s_rounds) & cur["qfe"].notna()
                 & (cur["res"] <= freeze)]
    nv_new = later[(later["model"] == "Naive Forecaster") & later["due"].isin(s_rounds) & later["qfe"].notna()
                   & (later["res"] > freeze)]
    if selftest:
        nv_new = nv_new[nv_new["res"] <= SNAP]
    naive = pd.DataFrame({"old": scores(nv_old.assign(model_pk=nv_old["due"]), shift_of(cur[cur["qfe"].notna()])),
                          "new": scores(nv_new.assign(model_pk=nv_new["due"]), shift)})
    rl = round_level(frozen, new, first_round(s_board), nr, naive)
    r.update({f"rl_{k}": v for k, v in rl.items()})
    if rl.get("n_peers"):
        print(f"  secondary (b): the {rl['top_round']} round ({rl['top_in_round']} of the top ten) led rounds with the "
              f"same kinds of new questions by {rl['lead_old']:.2f} at the freeze and {rl['lead_new']:.2f} on new "
              f"questions (kept {rl['kept']:.0%}); the Naive Forecaster's lead {rl['naive_old']:+.2f} -> {rl['naive_new']:+.2f}")
    r["sup_rescored"] = float(sup_later.iloc[0]) if len(sup_later) else np.nan
    above = frozen.index[frozen > REG_SUP]
    r["stay_rescored"] = int((new[above] > r["sup_rescored"]).sum())
    print(f"  P2 with the superforecaster median re-scored in the later snapshot ({r['sup_rescored']:.2f}): "
          f"{r['stay_rescored']} stay above")
    n_tr, n_teams, share = h1_luck_share(later, freeze, SNAP if selftest else "9999-12-31")
    r.update(h1_n=n_tr, h1_teams=n_teams, h1_share=share)
    print(f"  P3  external team-rounds due after {freeze}: {n_tr} from {n_teams} teams; luck share {share:.2f}"
          + ("" if n_tr >= 30 else " (fewer than 30: not yet testable)"))
    verdict = dict(p1=bool(r["diff"] > 0), p2=bool(r["stay"] <= 3 and r["gap_new"] < r["gap_old"]),
                   p3=bool(0.10 <= share <= 0.35) if n_tr >= 30 else None)
    print(f"  Verdicts (True = prediction held): {verdict}")
    if not selftest:
        json.dump(dict(run_on=today, results=r, verdict=verdict), open(lock, "w"), indent=1, default=str)
        record("prospective", **{k: v for k, v in r.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.snapshot:
        registered(a.snapshot, a.selftest)
    else:
        rehearsal()
