"""Render generated tables into report/report.md from code outputs (no hand-copied numbers).

Blocks replaced (markers are HTML comments, skipped when the Word file is built):
  TABLE2  candidate explanations for the apparent edge (metrics.json)
  TABLE3  policy comparison, heterogeneous skill + baseline noise (rq4_policy_sim_extra.csv)
  TABLEA1 policy comparison, equal and heterogeneous skill without baseline noise (rq4_policy_sim.csv)
  TABLEA2 sensitivity of the k = 9 results to each simulation parameter (rq4_sensitivity.csv)
check_report.py re-formats the same CSVs (a consistency check, not an independent derivation) and adds
condition checks for the claims the text makes about these tables.
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "report.md"
DER = ROOT / "data" / "derived"
SIM = pd.read_csv(DER / "rq4_policy_sim.csv")
EXTRA = pd.read_csv(DER / "rq4_policy_sim_extra.csv")
M = json.load(open(DER / "metrics.json"))
SF = pd.read_csv(DER / "h1_selection_fraction.csv").set_index("frac")
BI = 156.25  # Brier Index points per unit Brier near the top of the board (BI 68)
KS = (1, 3, 9, 22)

# Appendix table rows: (label, column, scenarios)
ROWS = [
    ("Status quo (best entry)", "status quo (best entry)", ["equal", "heterogeneous"]),
    ("Primary, truly best system declared", "pre-declared primary entry", ["equal", "heterogeneous"]),
    ("Primary chosen on prior-round scores", "primary chosen on prior round", ["heterogeneous"]),
    ("Held-out next round", "held-out re-score of best", ["equal", "heterogeneous"]),
    ("Team mean", "team mean", ["equal", "heterogeneous"]),
]
TAG = {"equal": "Equal", "heterogeneous": "Hetero"}
# Main table rows: (label, column)
ROWS3 = [
    ("Status quo: best entry", "status quo (best entry)"),
    ("Best entry minus expected selection gain", "winner's-curse-corrected best"),
    ("Pre-declared primary entry (chosen on prior round)", "primary chosen on prior round"),
    ("Held-out next round", "held-out re-score of best"),
    ("Team mean (display only)", "team mean"),
]


def cells(df, delta, col):
    d = df[df["delta"] == delta].set_index("k")
    return " / ".join(f"{d.loc[k, col] * 100:.0f}" for k in KS)


def u(s):
    """Unicode minus for negative numbers, as in the report text."""
    return s.replace("-", "−")


def table2():
    """The board as scored: what the top shows, what parity would show, and the noise, selection and
    round terms at the board's own scale."""
    sg = lambda v, d=2: u(f"{v:+.{d}f}")
    sup_hw = 1.96 * M["noise.sd_super"]
    rows = [
        "| Quantity | Index points | Status |",
        "|---|---|---|",
        f"| Top entry's edge over the superforecaster median | {sg(M['detect.top_gap'])} ({sg(M['detect.top_lo'])} to {sg(M['detect.top_hi'])}) | observed |",
        f"| Leading team's true lead consistent with that edge (90%) | {sg(M['parity.lead_lo'])} to {sg(M['parity.lead_hi'])} | simulated |",
        f"| Expected top-entry edge if the three leading teams were exactly at parity | {sg(M['parity.top_mean'])} (90%: {sg(M['parity.top_p05'])} to {sg(M['parity.top_p95'])}) | simulated |",
        f"| Uncertainty of the superforecaster score | ±{sup_hw:.1f} | bootstrap |",
        f"| True lead a single-round entry needs to be detected half the time | {M['detect.mde50']:.1f} | calculated |",
        f"| Selection within a team's round: half the questions / all questions | {M['h1.ext.inflation'] * BI:.2f} / about {M['bsel.h1_full']:.2f} | registered / extrapolated |",
        f"| Selection across the board: half the questions / all questions | {M['bsel.f50_net']:.1f} / about {M['bsel.full']:.1f} | exploratory |",
        f"| Shift of a fixed baseline in the two rounds with every above-median entry | {sg(M['anchor.20260607.overall.naive'], 1)} and {sg(M['anchor.20260830.overall.naive'], 1)} | exploratory |",
    ]
    return "\n".join(rows)


RB = pd.read_csv(DER / "real_board_policies.csv")


def table3():
    lines = ["| Policy applied to the 14 November board | Teams above the median | Their margins (points) |",
             "|---|---|---|"]
    for _, r in RB.iterrows():
        lines.append(f"| {r['policy']} | {r['teams']} | {r['margins']} |")
    return "\n".join(lines)


LR = pd.read_csv(DER / "rq4_likelihood_ratio.csv")


def tableA4():
    lines = ["| True gap (±points) | k = 1 | k = 9 | k = 1, perfect baseline | k = 9, perfect baseline |", "|---|---|---|---|---|"]
    for g in (0.25, 0.5, 1.0, 1.5):
        f = lambda k, b: LR[(LR["k"] == k) & (LR["gap"] == g) & (LR["baseline"] == b)]["lr"].iloc[0]
        lines.append(f"| {g:g} | {f(1, 'baseline error'):.1f} | {f(9, 'baseline error'):.1f} | "
                     f"{f(1, 'no baseline error'):.1f} | {f(9, 'no baseline error'):.1f} |")
    return "\n".join(lines)


SENS = pd.read_csv(DER / "rq4_sensitivity.csv")
SENS_COLS = ["single entry", "status quo", "corrected", "primary (prior round)", "held-out round"]  # "corrected" = calibrated gain


def tableA3():
    lines = ["| Variant (k = 9) | One entry | Status quo | Minus calibrated gain | Primary (prior round) | Held-out round |",
             "|---|---|---|---|---|---|"]
    for v, d in SENS.groupby("variant", sort=False):
        lo, hi = d[d["delta"] == -0.5].iloc[0], d[d["delta"] == 0.5].iloc[0]
        cells_ = " | ".join(f"{lo[c] * 100:.0f} / {hi[c] * 100:.0f}" for c in SENS_COLS)
        lines.append(f"| {v} | {cells_} |")
    return "\n".join(lines)


def table4():
    """The simulated trade-off at k = 9 (unequal skill, noisy baseline), with the break-even cost ratio
    against the status quo (range across the model variations) and false claims against a team that
    submits deliberately divergent variants."""
    e = lambda d, k, col: EXTRA[(EXTRA["delta"] == d) & (EXTRA["k"] == k)][col].iloc[0] * 100
    div = SENS[(SENS["variant"].str.startswith("Strategic: div")) & (SENS["delta"] == -0.5)].iloc[0]
    sq = "status quo (best entry)"
    rows = [("One entry (k = 1)", e(-0.5, 1, sq), e(0.5, 1, sq), "–", "–"),
            ("Status quo: best entry", e(-0.5, 9, sq), e(0.5, 9, sq), "–", f"{div['status quo'] * 100:.0f}"),
            ("Best entry minus expected selection gain", e(-0.5, 9, "winner's-curse-corrected best"), e(0.5, 9, "winner's-curse-corrected best"),
             f"{M['sens.c_corr_lo']:.2f}–{M['sens.c_corr_hi']:.2f}", f"{div['corrected'] * 100:.0f}"),
            ("Pre-declared primary entry", e(-0.5, 9, "primary chosen on prior round"), e(0.5, 9, "primary chosen on prior round"),
             f"{M['sens.c_prim_lo']:.2f}–{M['sens.c_prim_hi']:.2f}", f"{div['primary (prior round)'] * 100:.0f}"),
            ("Held-out next round", e(-0.5, 9, "held-out re-score of best"), e(0.5, 9, "held-out re-score of best"),
             f"{M['sens.c_held_lo']:.2f}–{M['sens.c_held_hi']:.2f}", f"{div['held-out round'] * 100:.0f}"),
            ("Team mean (display only)", e(-0.5, 9, "team mean"), e(0.5, 9, "team mean"),
             f"{(e(0.5, 9, sq) - e(0.5, 9, 'team mean')) / (e(-0.5, 9, sq) - e(-0.5, 9, 'team mean')):.2f}", "–")]
    lines = ["| Policy (k = 9 unless stated) | False claims (%) | Detection (%) | Beats the status quo when a false claim costs more than this many missed leads | False claims against divergent variants (%) |",
             "|---|---|---|---|---|"]
    for lab, fpv, dv, c_, dvv in rows:
        lines.append(f"| {lab} | {fpv:.0f} | {dv:.0f} | {c_} | {dvv} |")
    return "\n".join(lines)


def tableA1():
    lines = ["| Policy | Team 0.5 below: k = 1 / 3 / 9 / 22 | Team 0.5 above: k = 1 / 3 / 9 / 22 |", "|---|---|---|"]
    for label, col in ROWS3:
        lines.append(f"| {label} | {cells(EXTRA, -0.5, col)} | {cells(EXTRA, 0.5, col)} |")
    return "\n".join(lines)


def tableA2():
    lines = ["| Policy | Scenario | 0.5 below, k = 1 / 3 / 9 / 22 | 0.5 above, k = 1 / 3 / 9 / 22 |",
             "|---|---|---|---|"]
    for scen in ["equal", "heterogeneous"]:
        for label, col, scens in ROWS:
            if scen in scens:
                d = SIM[SIM["scenario"] == scen]
                lines.append(f"| {label} | {TAG[scen]} | {cells(d, -0.5, col)} | {cells(d, 0.5, col)} |")
    return "\n".join(lines)


def main():
    text = REPORT.read_text()
    for name, fn in [("TABLE2", table2), ("TABLE3", table3), ("TABLE4", table4), ("TABLEA1", tableA1), ("TABLEA2", tableA2),
                     ("TABLEA3", tableA3), ("TABLEA4", tableA4)]:
        block = f"<!-- {name}:START -->\n{fn()}\n<!-- {name}:END -->"
        text, n = re.subn(rf"<!-- {name}:START -->.*?<!-- {name}:END -->", block, text, flags=re.S)
        assert n == 1, f"{name} markers not found"
    REPORT.write_text(text)
    print(table2()); print(); print(table3()); print(); print(tableA4())


if __name__ == "__main__":
    main()
