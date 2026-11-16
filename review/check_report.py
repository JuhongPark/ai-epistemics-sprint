"""Deterministic report checks.

This script fails if any numeric claim in report/report.md differs from
the code output recorded in data/derived/metrics.json and CSVs, or if a structural requirement of
the sprint (abstract length, required sections, figures, references, page limit) is not met.

Usage:  .venv/bin/python review/check_report.py [--offline] [--no-pdf]
Exit code 0 = all checks pass.
"""

import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "report.md"
DER = ROOT / "data" / "derived"
OFFLINE = "--offline" in sys.argv
NO_PDF = "--no-pdf" in sys.argv

M = json.load(open(DER / "metrics.json"))
SIM = pd.read_csv(DER / "rq4_policy_sim.csv")
SF = pd.read_csv(DER / "h1_selection_fraction.csv").set_index("frac")
REP = pd.read_csv(DER / "replicated_entry_scores.csv")
SUP_ROW = REP[REP["model"] == "Superforecaster median forecast"].iloc[0]


VAL = pd.read_csv(DER / "rq4_validation_fraction.csv", index_col=0)
SUP_ERR = (M["h3.boot.sup_hi"] - M["h3.boot.sup_lo"]) / 2


def sim(scen, delta, k, col):
    r = SIM[(SIM["scenario"] == scen) & (SIM["delta"] == delta) & (SIM["k"] == k)]
    return float(r[col].iloc[0])


SQ, PRIM, HELD, TMEAN = ("status quo (best entry)", "pre-declared primary entry",
                         "held-out re-score of best", "team mean")
f0 = lambda v: f"{v:.0f}"
f1 = lambda v: f"{v:.1f}"
f2 = lambda v: f"{v:.2f}"
f3 = lambda v: f"{v:.3f}"
f4 = lambda v: f"{v:.4f}"
pc = lambda v: f"{v * 100:.0f}%"
pn = lambda v: f"{v * 100:.0f}"
kb = lambda v: f"{v * 1000:.2f}".replace("-", "−")  # report uses the Unicode minus
kb1 = lambda v: f"{v * 1000:.1f}"
comma = lambda v: f"{v:,.0f}"
m = lambda key: M[key]

# Each claim: (template with {} placeholders, [(value, formatter), ...]).
pt = lambda v: f"{v * 156.25:.2f}".replace("-", "−")  # Brier -> Brier Index points near BI 68
pt1 = lambda v: f"{v * 156.25:.1f}"
EX = pd.read_csv(DER / "rq4_policy_sim_extra.csv")
CONT = pd.read_csv(DER / "h3_continuous.csv")
# The three teams above the median, ordered by their best entry (Teams A, B, C in the report). Looked up
# from the derived data so that the code itself names no team.
_BEST = pd.read_csv(DER / "h3_entry_full.csv").groupby("org")["bi"].max().sort_values(ascending=False)
TEAM_A, TEAM_B, TEAM_C = [o for o in _BEST.index if o in set(CONT["org"])][:3]
def ex(delta, k, col):
    r = EX[(EX["delta"] == delta) & (EX["k"] == k)]
    return float(r[col].iloc[0])
def gap(frac, org):
    r = CONT[(CONT["frac"] == frac) & (CONT["org"] == org)]
    return float(r["gap_to_super"].iloc[0])
sgn = lambda v: (f"+{v:.1f}" if round(v, 1) > 0 else f"{v:.1f}".replace("-", "−")).replace("−0.0", "+0.0")
CORR, PRIOR = "winner's-curse-corrected best", "primary chosen on prior round"
EQC = "equal-skill-corrected best"
NET = pd.read_csv(DER / "net_of_naive.csv").sort_values("gap", ascending=False)  # first row = top entry
SUP_HW = 1.96 * M["noise.sd_super"]
NAIVE_MEAN = (M["anchor.20260607.overall.naive"] + M["anchor.20260830.overall.naive"]) / 2
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
word = lambda v: WORDS[int(round(v))] if v == v else "WORD FAILED"
sf2 = lambda v: (f"+{v:.2f}" if round(v, 2) >= 0 else f"{v:.2f}".replace("-", "−")).replace("−0.00", "+0.00")
sf2n = lambda v: f"{v:.2f}".replace("-", "−")
pt3 = lambda v: f"{v * 156.25:.3f}"
cond = lambda ok, text: text if ok else "CONDITION FAILED: " + text
lr_ok = lambda v: 1.6 <= v <= 2.5
single9, single9d = ex(-0.5, 9, SQ), ex(0.5, 9, SQ)
single1, single1d = ex(-0.5, 1, SQ), ex(0.5, 1, SQ)
split = [M[f"sens.split_{x}"] for x in ("sq", "corr", "prim", "held")]
SENS = pd.read_csv(DER / "rq4_sensitivity.csv")
div_d = SENS[(SENS["variant"].str.startswith("Strategic: div")) & (SENS["delta"] == 0.5)].iloc[0]
P, P0801, P0815 = "pseudo.t0901.", "pseudo.t0801.", "pseudo.t0815."
_PRO = (ROOT / "PROSPECTIVE.md").read_text()
PRO_P1 = re.search(r"Central prediction: a difference of about (\S+) index points", _PRO).group(1)
PRO_P2 = re.search(r"at most (\w+) of the (\w+) score above the superforecaster median", _PRO).groups()
PRO_P3 = re.search(r"luck share between (\d+)% and (\d+)%", _PRO).groups()
PS = (P, P0801, P0815)
CLAIMS = [
    # Abstract
    ("replicating ForecastBench's scoring from public data (r = {})", [(m("rep.r"), f3)]),
    ("largely because the human baseline is a single 2024 round, uncertain by ±{} points: in a calibrated model, a team 0.5 points better is {} to rank above it as one 0.5 points worse",
     [(SUP_HW, f1), (m("lr.k1_05"), lambda v: "only about twice as likely" if lr_ok(v) else "LR OUT OF RANGE")]),
    ("a perfectly measured baseline would {} that ratio", [(m("lr.k1_05_nb") / m("lr.k1_05"), lambda v: "nearly double" if 1.6 <= v <= 2.1 else "RATIO CHANGED")]),
    # Odds range: lowest = net of the round shift (§4.1), highest = flat prior without netting.
    ("The leading team's posterior odds of a real lead are about {}:1 to {}:1",
     [(min(m("post.net_min"), m("post.p_n05")), lambda v: f"{v / (1 - v):.1f}"), (m("post.p_flat"), lambda v: f"{v / (1 - v):.0f}")]),
    ("the ordinary winner's curse costs the top entries about {} points (exploratory; extrapolated), and a team's choice of its best entry within a round about {} (pre-registered on half the questions, then extrapolated)",
     [(m("bsel.full"), f1), (m("bsel.h1_full"), f1)]),
    ("on the board of 1 September, the round holding {} of the top ten kept {} of its lead over comparable rounds on later questions (exploratory)",
     [(m(P + "rl_top_in_round") if m(P + "rl_top_round") == "2026-06-07" and m(P + "top_n") == 10 else float("nan"), word),
      (m(P + "rl_kept"), lambda v: "a quarter" if 0.2 <= v < 0.3 else "KEPT SHARE CHANGED")]),
    ("{}", [(0, lambda v: cond(ex(-0.5, 9, PRIOR) < single1 and ex(-0.5, 9, HELD) < single1 and min(split[2:]) > single1,
                                "held-out confirmation or a pre-declared primary entry, counted per team, cuts false outperformance claims at some cost in missed leads") if ex(0.5, 9, PRIOR) < single1d and ex(0.5, 9, HELD) < single1d else "DETECTION NOT LOWER")]),
    # Introduction
    ("make up {} of {} external teams and {} of external entries ({} of {})",
     [(m("board.big_teams"), f0), (m("board.ext_teams"), f0), (m("board.big_share"), pc),
      (m("board.big_entries"), f0), (m("board.ext_entries"), f0)]),
    ("and {} of the top 20 external entries", [(m("board.top20_big"), f0)]),
    ("stronger on their median entry (Spearman {})", [(m("kcorr.median"), f2)]),
    ("The superforecaster score is uncertain by ±{} points, so a single-round entry would need a true lead of {} points to be detected half the time. With one entry, the baseline's error alone {} the evidence a rank carries. For the leading team the data mildly favour a lead (posterior odds about {}:1 to {}:1) but allow anything from {} to +{} points (90%)",
     [(SUP_HW, f1), (m("detect.mde50"), f1), (m("lr.k1_05") / m("lr.k1_05_nb"), lambda v: "nearly halves" if 0.5 <= v <= 0.65 else "RATIO CHANGED"),
      (min(m("post.net_min"), m("post.p_n05")), lambda v: f"{v / (1 - v):.1f}"), (m("post.p_flat"), lambda v: f"{v / (1 - v):.0f}"), (m("parity.lead_lo"), sf2n), (m("parity.lead_hi"), f2)]),
    ("Extrapolated from held-out splits, the ten best entries lose about {} points relative to random entries (exploratory); a team's choice of its best entry within a round adds {} points on half the questions (pre-registered), about {} on all",
     [(m("bsel.full"), f1), (m("h1.ext.inflation"), pt), (m("bsel.h1_full"), f2)]),
    ("{}", [(0, lambda v: cond(m("sens.div_prim") < single1 and m("sens.div_held") < single1 and min(split) > single1,
                                "In simulation, a held-out round or a pre-declared primary entry keeps false outperformance claims below the single-entry rate, even against deliberately divergent variants, but misses more real leads") if 0.8 <= m("loss.c_prim") <= 1.3 and 0.8 <= m("loss.c_held") <= 1.3 else "COST RATIO NOT ABOUT ONE")]),
    # Methods
    ("{} forecast sets over {} rounds, of which {} are external tournament submissions from {} organisations across {} rounds",
     [(m("data.files"), comma), (m("data.rounds"), f0), (m("data.ext_files"), f0), (m("data.ext_orgs"), f0), (m("data.ext_rounds"), f0)]),
    ("({} entries, {} teams); {} of {} external team-rounds comply",
     [(m("board.rows"), f0), (m("board.teams"), f0), (m("data.ext_team_rounds_le3"), f0), (m("data.ext_team_rounds"), f0)]),
    ("Of {} external entries, {} match published rows one-to-one, and their scores correlate at r = {} (mean absolute difference {}, max {})",
     [(m("rep.n_ext"), f0), (m("rep.n_matched"), f0), (m("rep.r"), f4), (m("rep.mad"), f2), (m("rep.maxdiff"), f2)]),
    ("for the top 20 external entries the mean difference is {} and the maximum {}", [(m("rep.top20.mae"), f2), (m("rep.top20.max"), f2)]),
    ("replicates at {} against a published {}", [(m("h3.sup"), f2), (m("board.sup_pub"), f1)]),
    ("For each of {} external team-rounds with at least two entries (from {} of the {} organisations; {} of these are among the board's teams)",
     [(m("h1.n_ext"), f0), (m("h1.teams_ext"), f0), (m("data.ext_orgs"), f0), (m("named.h1_board_overlap"), f0)]),
    ("entry noise {}, round noise {}, baseline error {}, and a skill spread of {} that reproduces the observed best-minus-mean gap",
     [(m("cal.sd_entry"), f2), (m("cal.sd_round"), f2), (m("cal.sd_base"), f2), (m("cal.sd_skill"), f2)]),
    ("under {} variations of the model and two strategic teams", [(m("sens.n_variants"), word)]),
    # Table 1
    ("| {} team-rounds, {} organisations | {}; luck share {} | {}–{} (team-rounds); {}–{} (teams) |",
     [(m("h1.n_ext"), f0), (m("h1.teams_ext"), f0), (m("h1.ext.inflation"), pt), (m("h1.ext.share"), pc),
      (m("h1.ext.lo"), pt3), (m("h1.ext.hi"), pt3), (m("h1.ext.team.lo"), pt3), (m("h1.ext.team.hi"), pt3)]),
    ("| {} team-rounds | {}; luck share {} | {}–{} |",
     [(m("h1.n_all"), f0), (m("h1.all.inflation"), pt), (m("h1.all.share"), pc), (m("h1.all.lo"), pt), (m("h1.all.hi"), pt)]),
    ("| {} pairs, {} teams | {}; {} retained | {}–{} | {} (bootstrap); {} (sign-flip) |",
     [(m("h2.ext.pairs"), f0), (m("h2.ext.teams"), f0), (m("h2.ext.shrink"), pt), (m("h2.ext.retained"), pc),
      (m("h2.ext.lo"), pt), (m("h2.ext.hi"), pt), (2 * m("h2.ext.p"), f3), (min(1, 2 * m("h2.ext.signflip_team")), f2)]),
    ("| {} pairs, {} teams | {}; {} retained | {}–{} | {}; {} |",
     [(m("h2.all.pairs"), f0), (m("h2.all.teams"), f0), (m("h2.all.shrink"), pt), (m("h2.all.retained"), pc),
      (m("h2.all.lo"), pt), (m("h2.all.hi"), pt), (2 * m("h2.all.p"), f3), (min(1, 2 * m("h2.all.signflip_team")), f2)]),
    ("| {} teams ({} entries) | {}, {}, {} (20% held out) |",
     [(m("h3.sq_teams"), f0), (m("h3.sq_entries"), f0), (gap(0.8, TEAM_A), sgn),
      (gap(0.8, TEAM_B), sgn), (gap(0.8, TEAM_C), sgn)]),
    # 4.1
    ("rests on {} questions from one 2024 round; only {} are market questions", [(SUP_ROW["n"], f0), (SUP_ROW["count_market"], f0)]),
    ("Its 95% interval is {}–{}, about ±{} points ({}–{} in the registered, unstratified bootstrap). A single-round entry's own score is uncertain by about ±{}, so combining the two, it would need a true lead of {} points to be detected half the time ({} for 80% power)",
     [(m("noise.sup_lo"), f1), (m("noise.sup_hi"), f1), (SUP_HW, f1), (m("h3.boot.sup_lo"), f1), (m("h3.boot.sup_hi"), f1),
      (1.96 * m("noise.sd_entry_med"), f1), (m("detect.mde50"), f1), (m("detect.mde80"), f1)]),
    ("Across model variants the ratio is {}–{}", [(m("lr.var_min"), f1), (m("lr.var_max"), f1)]),
    ("These are posterior odds of a lead of about {}:1 to {}:1; netting out the round shift estimated in §4.2 lowers the posterior to {}–{}, odds of about {}:1 to {}:1",
     [(m("post.p_n05"), lambda v: f"{v / (1 - v):.0f}"), (m("post.p_flat"), lambda v: f"{v / (1 - v):.0f}"), (m("post.net_min"), f2), (m("post.net_max"), f2), (m("post.net_min"), lambda v: f"{v / (1 - v):.1f}"), (m("post.net_max"), lambda v: f"{v / (1 - v):.1f}")]),
    ("the reverse one-sided values for the seven entries are {}–{}, and {} after Holm correction over {} entries",
     [(m("h3.pub.rev_min"), f2), (m("h3.pub.rev_max"), f2), (m("h3.pub.holm_min"), f1), (m("h3.pub.n_tested"), f0)]),
    ("ranking above them is {} times as likely if it is better with one entry, and {} times with nine; a perfectly measured human baseline would raise these likelihood ratios to {} and {}, so for one entry the baseline's error alone {} the ratio",
     [(m("lr.k1_05"), f2), (m("lr.k9_05"), f2), (m("lr.k1_05_nb"), f2), (m("lr.k9_05_nb"), f2),
      (m("lr.k1_05") / m("lr.k1_05_nb"), lambda v: "nearly halves" if 0.5 <= v <= 0.65 else "RATIO CHANGED")]),
    ("the posterior probability of a real lead is {}–{} across a flat prior and priors centred on parity",
     [(m("post.p_n05") if m("post.p_n05") <= m("post.p_n1") <= m("post.p_flat") else float("nan"), f2), (m("post.p_flat"), f2)]),
    ("({} under a flat prior; {} and {} under normal priors centred on parity with standard deviation 1 and 0.5 points)",
     [(m("post.p_flat"), f2), (m("post.p_n1"), f2), (m("post.p_n05"), f2)]),
    ("its true lead could be anywhere from {} to +{} points (90%)", [(m("parity.lead_lo"), sf2n), (m("parity.lead_hi"), f2)]),
    # 4.2
    ("the ten best entries lose {} points on the other half, while ten random entries lose {} and the bottom ten gain {}",
     [(m("bsel.f50_top"), f2), (m("bsel.f50_rand") if m("bsel.f50_rand") > 0 else float("nan"), f2), (-m("bsel.f50_bottom"), f2)]),
    ("Net of the random entries the drop is {} points (question-bootstrap 95% interval {}–{}); a log-log fit extrapolates it to about {} points when the board picks on all questions (Figure 2b). With one entry per team the drop is still {}",
     [(m("bsel.f50_net"), f2), (m("bsel.f50_ci_lo"), f2), (m("bsel.f50_ci_hi"), f2), (m("bsel.full"), f2), (m("bsel.f50_teams_net"), f2)]),
    ("is {} points on the questions used to pick it and {} points on held-out questions: a selection gain of {} points (team-clustered 95% interval {}–{}) and a luck share of {} ({}–{}; {} in the registered primary set)",
     [(m("h1.ext.adv_a"), pt), (m("h1.ext.adv_b"), pt), (m("h1.ext.inflation"), pt), (m("h1.ext.team.lo"), pt),
      (m("h1.ext.team.hi"), pt), (m("h1.ext.share"), pc), (m("h1.ext.team.share_lo"), pn), (m("h1.ext.team.share_hi"), pc),
      (m("h1.all.share"), pc)]),
    ("The shuffled-label control returns {} and a held-out advantage of {} points ({} to {})",
     [(m("h1.posctl.share"), pc), (m("h1.posctl.adv_b"), pt), (m("h1.posctl.adv_b_lo"), pt), (m("h1.posctl.adv_b_hi"), pt)]),
    ("from {} at 30% to {} at 90% (Figure 2a); extrapolated after registration to selection on all questions, the gain is about {} points",
     [(SF.loc[0.3, "luck_share"], pc), (SF.loc[0.9, "luck_share"], pc), (m("bsel.h1_full"), f2)]),
    ("with {} entries, were exactly as good as superforecasters, the board would show a top entry {} points above the median on average (90% range {} to +{}). A top entry at least as far above as the observed one (+{}) would occur {} of the time, and {} or more entries above the median {} of the time",
     [(m("parity.ks"), lambda s: s.replace("/", ", ", 1).replace("/", " and ")), (m("parity.top_mean"), f2), (m("parity.top_p05"), sf2n),
      (m("parity.top_p95"), f2), (m("parity.obs_top"), f2), (m("parity.p_top_ge_obs"), pc),
      (m("parity.obs_n"), word), (m("parity.p_nabove_ge_obs"), pc)]),
    ("the leading team's best entry stays {} points above the median on the point estimate, not significantly; the other two fall to {} and {} (Appendix Figure A1)",
     [(gap(0.8, TEAM_A), f1), (gap(0.8, TEAM_B), sgn), (gap(0.8, TEAM_C), sgn)]),
    ("the entry ranked first on a selection half loses {} points and the top three {}, but at 80% only {} and {}",
     [(m("bsel.f50_top1_net"), f2), (m("bsel.f50_top3_net"), f2), (m("bsel.f80_top1_net"), f2), (m("bsel.f80_top3_net"), f2)]),
    # Out-of-time rehearsal (13_prospective.py)
    ("scored each entry on its questions that resolved between then and our snapshot (median {} per entry; data we had already analysed)", [(m(P + "n_new_med"), f0)]),
    ("Scored on those later questions, the ten best entries on that board lost {} points while ten random entries gained {}, a relative drop of {}; {} of random sets of ten did as badly",
     [(-m(P + "top_chg") if m(P + "top_chg") < 0 else float("nan"), f2), (m(P + "rand_chg") if m(P + "rand_chg") > 0 else float("nan"), f2), (m(P + "diff"), f2), (m(P + "p_ref"), pc)]),
    ("{} of them came from the 7 June round, and against the other entries of their own rounds they did only {} worse",
     [(m(P + "rl_top_in_round") if m(P + "rl_top_round") == "2026-06-07" and m(P + "top_n") == 10 else float("nan"), lambda v: word(v).capitalize()),
      (m(P + "diff_matched"), f2)]),
    ("the 7 June round's entries as a whole kept {} ({} points) of their earlier {}-point lead", [(m(P + "rl_kept"), pc), (m(P + "rl_lead_new"), f2), (m(P + "rl_lead_old"), f2)]),
    ("The Naive Forecaster's lead in that round grew from {} to {} points", [(m(P + "rl_naive_old"), f2), (m(P + "rl_naive_new"), f2)]),
    ("{}", [(0, lambda v: cond(all(m(q + "rl_naive_new") > m(q + "rl_naive_old") for q in PS) and all(m(q + "rl_top_round") == "2026-06-07" for q in PS),
                                "so by this fixed baseline the later questions were not harder for that round"))]),
    ("{}", [(0, lambda v: cond(all(m(q + "top_n") == 10 and 0 < m(q + "diff_matched") < m(q + "diff") and m(q + "rl_kept") < 0.5 for q in PS),
                                "Boards frozen on 1 and 15 August show the same pattern for the top ten"))]),
    ("Of the {} entries above the superforecaster median on 1 September, none scored above it on the later questions (August boards: {} of {} and {} of {})",
     [(m(P + "n_above") if m(P + "stay") == 0 and m(P + "above_kept") == m(P + "n_above") else float("nan"), word),
      (m(P0801 + "stay"), f0), (m(P0801 + "above_kept"), f0), (m(P0815 + "stay"), f0), (m(P0815 + "above_kept"), f0)]),
    ("On the boards of 1 and 15 August the top ten did {} and {} points worse than random entries ({} and {} against {} and {}) and {} and {} worse than the other entries of their rounds, and the 7 June entries kept {} and {} of their lead over comparable rounds while the Naive Forecaster's lead grew",
     [(m(P0801 + "diff"), f2), (m(P0815 + "diff"), f2), (m(P0801 + "top_chg"), sf2), (m(P0815 + "top_chg"), sf2), (m(P0801 + "rand_chg"), sf2), (m(P0815 + "rand_chg"), sf2),
      (m(P0801 + "diff_matched"), f2), (m(P0815 + "diff_matched"), f2), (m(P0801 + "rl_kept"), pc), (m(P0815 + "rl_kept"), pc)]),
    ("Of random sets of ten entries, {}–{} did at least as badly as the top ten", [(min(m(q + "p_ref") for q in PS), pn), (max(m(q + "p_ref") for q in PS), pc)]),
    ("External team-rounds after 1 September give an H1 luck share of {}", [(m("pseudo.h1.share"), pc)]),
    ("round noise at the Naive Forecaster's round-to-round spread ({}), such a top entry would occur {} of the time", [(m("cal.sd_round_ub"), f2), (m("parity.p_top_ge_obs_rn"), pc)]),
    ("horizons up to {} and {} days, against up to {} days", [(m("rounds.hz_jun07"), f0), (m("rounds.hz_aug30"), f0), (m("rounds.hz_super"), f0)]),
    ("the superforecaster median stays at {} and the seven gaps widen by {} points on average",
     [(m("conf.h90.sup") if abs(m("conf.h90.sup") - m("conf.all.sup")) < 0.005 else float("nan"), f2), (m("conf.h90.gap_change"), f2)]),
    ("ForecastBench's Naive Forecaster scores {} and {} points higher in these rounds than in other rounds of the same period, {} and {} standard deviations of its spread over the other 2026 rounds",
     [(m("anchor.20260607.overall.naive"), f2), (m("anchor.20260830.overall.naive"), f2), (m("anchor.naive.z_jun07"), f2), (m("anchor.naive.z_aug30"), f2)]),
    ("Fitted on the other rounds, the external round mean rises {} points per Naive Forecaster point (95% interval {}–{}; {} within teams)",
     [(m("anchor.slope.loo"), f2), (m("anchor.slope.loo_lo"), f2), (m("anchor.slope.loo_hi"), f2), (m("anchor.slope.fe"), f2)]),
    ("({} of the three teams' entries above the median on average)", [(m("parity.nabove_mean"), f1)]),
    ("the top entry stays {} to {} points above the median across the slope's interval and {} at the within-team slope",
     [(m("net.top_b_lo"), f2), (m("net.top_b_hi"), f2), (m("net.top_fe"), f2)]),
    ("Subtracting the shift at that slope leaves the top entry {} points above the median (Appendix B), and {}",
     [(m("net.top_b"), f2),
      (abs(m("net.top_full")) if m("anchor.slope.loo_hi") < 1 else float("nan"), lambda v: "only full carry-over, outside that interval, erases it" if v < 0.05 else "NOT ON MEDIAN")]),
    # 4.3
    ("{}", [(0, lambda v: cond(m("realb.sq_entries") == 7 and m("realb.one_teams") == 3 and m("realb.corr_teams") == 2 and m("realb.held_teams") == 2,
                                "counting one entry per team turns seven entries into three teams, and subtracting each team's expected selection gain or re-scoring the best entry of each team above on held-out questions leaves two"))]),
    ("the second by {} points or less", [(np.ceil(m("realb.second_margin") * 100) / 100, f2)]),
    ("appears above them {} of the time with one entry; the status quo raises this to {} with nine entries and {} with 22",
     [(single1, pc), (single9, pc), (ex(-0.5, 22, SQ), pc)]),
    ("the model's expected selection gain is {} points, against {} extrapolated within a round", [(m("cal.gain_het_k9"), f2), (m("bsel.h1_full"), f2)]),
    ("a threshold {} points above the median matches the primary entry's false-claim rate and detects {} of real leads, against {} and {}",
     [(m("roc.t"), f2), (m("roc.det_sq") if m("roc.det_sq") > max(m("roc.det_prim"), m("roc.det_held")) else float("nan"), pc),
      (m("roc.det_prim"), pc), (m("roc.det_held"), pc)]),
    ("{}", [(0, lambda v: cond(m("sens.div_corr") > single1 + 0.05, "divergent variants defeat a raised bar, as they defeat the correction"))]),
    ("hold false claims below the single-entry rate in every one of {} variations of the model and against a team that submits deliberately divergent variants, and they pay off once a false outperformance claim costs {}",
     [(m("sens.n_variants") if m("sens.all_prim_below_single") * m("sens.all_held_below_single") == 1 and max(m("sens.div_prim"), m("sens.div_held")) < single1 else float("nan"), word),
      ((m("sens.c_prim_lo"), m("sens.c_prim_hi"), m("sens.c_held_lo"), m("sens.c_held_hi")),
       lambda v: "about as much as a missed lead or more" if 0.8 <= min(v) and max(v) <= 1.3 else "BREAK-EVEN FAR FROM ONE")]),
    ("defeats every per-name policy ({}–{})", [(min(split) if min(split) > single1 else float("nan"), pn), (max(split), pc)]),
    ("with equal costs, total error is {}–{} against {} for a single entry",
     [(m("loss.k9_min"), f2), (m("loss.k9_max"), f2), (m("loss.single"), f2)]),
    # Discussion, limitations, conclusion
    ("**About {} points at the board's scale (extrapolated)", [(m("bsel.full"), f1)]),
    ("### 4.2 RQ2: About {} points of selection at the board's scale", [(m("bsel.full"), f1)]),
    ("bootstrap-only support on all teams, where {} of the advantage is retained", [(m("h2.all.retained"), pc)]),
    ("the plotted 50% point in Figure 2a, {}, uses its own splits", [(SF.loc[0.5, "luck_share"], pc)]),
    ("at least 500 questions {} / {}; at least 1,000 questions {} / {}", [(m("rob.h3.entries_with_n_ge_500.sq"), f0), (m("rob.h3.entries_with_n_ge_500.tm"), f0), (m("rob.h3.entries_with_n_ge_1000.sq"), f0), (m("rob.h3.entries_with_n_ge_1000.tm"), f0)]),
    ("without the largest team {} / {}", [(m("rob.h3.excluding_largest_team.sq"), f0), (m("rob.h3.excluding_largest_team.tm"), f0)]),
    ("Bootstrap 95% intervals on the counts: status quo {}–{} teams ({}–{} with the threshold fixed), team mean {}–{}", [(m("h3.boot.sq_lo"), f0), (m("h3.boot.sq_hi"), f0), (m("h3.boot.sq_fixed_lo"), f0), (m("h3.boot.sq_fixed_hi"), f0), (m("h3.boot.tm_lo"), f0), (m("h3.boot.tm_hi"), f0)]),
    ("H2 Spearman correlation of variant ranks across rounds: {} on all teams", [(m("h2.all.spearman"), f2)]),
    ("The measured edges (up to {} points) are smaller than the {}-point lead a single-round entry needs to be detected half the time. They are also of the same order as the winner's curse across the board (about {} points)",
     [(m("rounds.best_2026_08") - m("h3.sup"), f1), (m("detect.mde50"), f1), (m("bsel.full"), f1)]),
    ("overpredicts H1's luck share at every selection fraction by {}–{} points",
     [(m("val.over_min") * 100 if m("val.over_min") > 0 else float("nan"), f0), (m("val.over_max") * 100, f0)]),
    ("The external H2 sample is small ({} pairs)", [(m("h2.ext.pairs"), f0)]),
    ("For the leading team the odds of a real lead are about {}:1 to {}:1",
     [(min(m("post.net_min"), m("post.p_n05")), lambda v: f"{v / (1 - v):.1f}"), (m("post.p_flat"), lambda v: f"{v / (1 - v):.0f}")]),
    ("from two recent rounds that {}", [(NAIVE_MEAN, lambda v: "a fixed baseline found easier" if v > 1 else "NAIVE SHIFT SMALL")]),
    ("the round holding {} of the top ten on the 1 September board kept {} of its lead on later questions (both exploratory)",
     [(m(P + "rl_top_in_round") if m(P + "rl_top_round") == "2026-06-07" and m(P + "top_n") == 10 else float("nan"), word),
      (m(P + "rl_kept"), lambda v: "a quarter" if 0.2 <= v < 0.3 else "KEPT SHARE CHANGED")]),
    # Future work: the registered predictions as stated in PROSPECTIVE.md
    ("the top ten lose more than random entries (central estimate {} points); at most {} of the {} score above the median; and H1's luck share in later rounds is {}–{}%",
     [(PRO_P1, str), (PRO_P2[0], str), (PRO_P2[1], str), (PRO_P3[0], str), (PRO_P3[1], str)]),
    ("a rank above the median raises the odds of a real lead {}", [(m("lr.k1_05"), lambda v: "only about twofold" if lr_ok(v) else "LR OUT OF RANGE")]),
    ("The ordinary winner's curse costs the top entries about {} points (extrapolated from held-out splits), and luck shared by a whole round can cost more", [(m("bsel.full"), f1)]),
    # Appendix B
    ("which counted a fourth team at {}; Team D's entry sits within replication error of the threshold, and using 68.1 raises every gap by {} points",
     [(m("h3.fourth_team_bi"), f2), (m("h3.sup") - 68.1, f2)]),
    ("formula for normally distributed skill ({}) although the simulation draws half-normal gaps, so the model reproduced a best-minus-mean gap of only {} points against {} observed",
     [(m("cal.sd_skill_normal"), f2), (m("cal.model_gap_old"), f2), (m("cal.obs_gap_h1"), f2)]),
    ("The status-quo false-claim rate at k = 9 moved from 48% to {}.", [(single9, pc)]),
    ("(0.62–0.82 became {}–{}), that a round-noise variant used the Naive Forecaster's total spread instead of its round component (1.01 became {}), that the carry-over slope was fitted partly on the two rounds it is used to judge (0.44 became {} without them)",
     [(m("post.p_n05"), f2), (m("post.p_flat"), f2), (m("cal.sd_round_ub"), f2), (m("anchor.slope.loo"), f2)]),
    ("H1 holds on dataset questions only ({}), entries with at least 500 scored rows ({}), without the most active team ({}) and without anonymous teams ({}), and is larger on market questions only ({})",
     [(m("rob.h1.dataset.share"), pc), (m("rob.h1.min500.share"), pc), (m("rob.h1.drop_top.share"), pc),
      (m("named.h1_share"), pc), (m("rob.h1.market.share"), pc)]),
    ("the calibrated model predicts {} retention with unequal skill and {} with equal skill",
     [(m("cal.pred_retention_hetero") if m("cal.pred_retention_equal") < m("h2.ext.retained") < m("cal.pred_retention_hetero") else float("nan"), pc),
      (abs(m("cal.pred_retention_equal")), pc)]),
    ("with {} pairs, the test has 80% power only against a loss of at least {} points", [(m("h2.ext.pairs"), f0), (m("h2.ext.mde"), pt)]),
    ("of {} round-level best variants above the median, {} reappears in a later round, and it fell from {} to {}",
     [(m("backtest.claims"), f0), (m("backtest.tested"), f0), (m("backtest.mean_bi_t"), f2), (m("backtest.mean_bi_next"), f2)]),
]


sys.path.insert(0, str(ROOT / "review"))
from render_tables import table2, table3, table4, tableA1, tableA2, tableA3, tableA4  # consistency with the CSVs
for fn in (table2, table3, table4, tableA1, tableA2, tableA3, tableA4):
    for line in fn().splitlines()[2:]:
        CLAIMS.append((line, []))


def main():
    text = REPORT.read_text()
    fails, passes = [], 0

    for tmpl, vals in CLAIMS:
        expected = tmpl.format(*[fmt(v) for v, fmt in vals]) if vals else tmpl
        if expected in text:
            passes += 1
        else:
            anchor = tmpl.split("{}")[0][:40]
            i = text.find(anchor) if anchor else -1
            found = text[i:i + len(expected) + 20].replace("\n", " ") if i >= 0 else "(anchor not found)"
            fails.append(f"NUMBER  expected: {expected}\n        report:   {found}")

    abstract = text.split("## Abstract")[1].split("## 1.")[0]
    n_words = len(abstract.split())
    (passes := passes + 1) if 150 <= n_words <= 250 else fails.append(f"ABSTRACT {n_words} words (150-250)")

    for h in ["## Abstract", "## 1. Introduction", "## 2. Related Work", "## 3. Methods", "## 4. Results",
              "## 5. Discussion and Limitations", "## 6. Conclusion", "## References",
              "## Appendix A. Limitations and Dual-Use / Ethical Considerations", "## LLM Usage Statement"]:
        (passes := passes + 1) if h in text else fails.append(f"SECTION missing: {h}")

    for fig in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        (passes := passes + 1) if (REPORT.parent / fig).resolve().exists() else fails.append(f"FIGURE missing: {fig}")

    refs = text.split("## References")[1].split("## Appendix A")[0]
    if not OFFLINE:
        import time
        for aid in re.findall(r"arXiv:(\d{4}\.\d{4,5})", refs):
            ok = False
            for attempt in range(4):  # the arXiv API rate-limits; it asks for ~3 s between calls
                time.sleep(3 * (attempt + 1))
                try:
                    xml = urllib.request.urlopen(f"https://export.arxiv.org/api/query?id_list={aid}", timeout=30).read().decode()
                    ok = xml.count("<entry>") == 1 and "<title>Error</title>" not in xml
                except Exception:  # network problem is a failure unless a retry succeeds
                    ok = False
                if ok:
                    break
            if not ok:  # fallback: the abstract page carries the id in its citation metadata
                try:
                    req = urllib.request.Request(f"https://arxiv.org/abs/{aid}", headers={"User-Agent": "Mozilla/5.0"})
                    html = urllib.request.urlopen(req, timeout=30).read().decode()
                    ok = f'citation_arxiv_id" content="{aid}' in html
                except Exception:
                    ok = False
            (passes := passes + 1) if ok else fails.append(f"REF arXiv:{aid} did not resolve")
        for doi in re.findall(r"doi:(\S+?)\.?$", refs, flags=re.M):
            try:
                urllib.request.urlopen(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}", timeout=30)
                passes += 1
            except Exception:
                fails.append(f"REF doi:{doi} did not resolve")
        for url in re.findall(r"(https://[^\s;)]+)", refs):
            if "github.com" in url or "arxiv.org" in url:
                continue
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                code = urllib.request.urlopen(req, timeout=30).status
                (passes := passes + 1) if code == 200 else fails.append(f"REF {url} -> HTTP {code}")
            except Exception as e:
                fails.append(f"REF {url} -> {type(e).__name__}")

    if not NO_PDF:
        soffice = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
        subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "report/build_docx.py")], check=True,
                       capture_output=True)
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(ROOT / "report"),
                        str(ROOT / "report/report.docx")], capture_output=True)
        pdf = ROOT / "report/report.pdf"
        pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                                                                text=True).stdout).group(1))
        page_lines = {p: [l for l in subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), str(pdf), "-"],
                                                    capture_output=True, text=True).stdout.splitlines() if l.strip()]
                      for p in range(1, pages + 1)}
        ref_page = next(p for p in page_lines if "References" in page_lines[p])
        main_end = ref_page - 1 if page_lines[ref_page][0].strip() == "References" else ref_page
        body = " ".join(" ".join(page_lines[p]) for p in range(1, ref_page + 1)).split("References")[0]
        for bad in ["<!--", "-->", "TABLE2", "render_tables", ".py`", "{}", "TODO"]:
            (passes := passes + 1) if bad not in body else fails.append(f"RENDER main body contains '{bad}'")
        (passes := passes + 1) if main_end <= 8 else fails.append(f"PAGES main body ends on page {main_end} (> 8)")
        print(f"[layout] main body ends on page {main_end}; References start on page {ref_page}; total {pages} pages")

    print(f"{passes} checks passed, {len(fails)} failed")
    for f in fails:
        print(" -", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
