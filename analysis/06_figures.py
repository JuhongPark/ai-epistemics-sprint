"""Figures for the report (static PDF + PNG, light mode, print).

Main text: Fig 1 leaderboard top 20; Fig 2 H3 teams; Fig 3 policy simulation.
Appendix:  Fig A1 H1 luck share by selection fraction; Fig A2 H2 cross-round scatter.
Palette: reference categorical slots 1-4 (validated light mode; slots 3-4 below 3:1 contrast, so
series carry direct labels or a legend plus distinct marker shapes and line styles).
Teams are anonymized as Team A, B, ... by best rank on the published board.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, DERIVED, FIGS, BENCHMARK, brier_index

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#ecebe7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)


def team_letters(lb):
    ext = lb[lb["Team"] != BENCHMARK]
    order = ext.groupby("Team")["Rank"].min().sort_values().index
    return {t: (chr(ord("A") + i) if i < 26 else f"A{chr(ord('A') + i - 26)}") for i, t in enumerate(order)}


def fig1_leaderboard():
    lb = pd.read_csv(DATA / "leaderboards" / "lb_tournament.csv")
    letters = team_letters(lb)
    counts = lb.groupby("Team").size()
    sup_row = lb[lb["Model"].str.startswith("Superforecaster")].iloc[0]
    sup_ci = [float(x) for x in sup_row["Overall 95% CI"].strip("[]").split(",")]
    # Top 20 external entries (as counted in the text) plus the superforecaster median at its rank
    ext = lb[(lb["Team"] != BENCHMARK) & ~lb["Model"].str.startswith("Superforecaster")]
    top = pd.concat([ext.sort_values("Overall", ascending=False).head(20), sup_row.to_frame().T])
    top = top.astype({"Overall": float, "Rank": int}).sort_values(["Overall", "Rank"], ascending=[False, True])
    top = top.iloc[::-1].reset_index(drop=True)
    ci = top["Overall 95% CI"].str.extract(r"\[([\d.]+),\s*([\d.]+)\]").astype(float)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axvspan(sup_ci[0], sup_ci[1], color=BAND, zorder=0, lw=0)
    groups = [("1 entry on board", lambda c: c == 1, S1), ("2–5 entries", lambda c: 2 <= c <= 5, S3),
              ("6+ entries", lambda c: c >= 6, S2)]
    labels, present = [], set()
    for i, r in top.iterrows():
        xerr = [[r["Overall"] - ci.loc[i, 0]], [ci.loc[i, 1] - r["Overall"]]]
        if r["Model"].startswith("Superforecaster"):
            ax.errorbar(r["Overall"], i, xerr=xerr, fmt="D", color=INK, ms=6, lw=1.2, zorder=3)
            labels.append("Superforecaster median"); continue
        c = counts[r["Team"]]
        name, _, color = next(g for g in groups if g[1](c))
        present.add(name)
        ax.errorbar(r["Overall"], i, xerr=xerr, fmt="o", color=color, ms=5, lw=1, zorder=3)
        team = "ForecastBench" if r["Team"] == BENCHMARK else f"Team {letters[r['Team']]}"
        labels.append(f"#{r['Rank']}  {team} ({c} {'entry' if c == 1 else 'entries'})")
    ax.axvline(sup_row["Overall"], color=INK, lw=1, ls="--", zorder=1)
    sup_rep = float(pd.read_csv(DERIVED / "h3_threshold.csv", index_col=0).iloc[0, 0])
    ax.axvline(sup_rep, color=INK2, lw=1, ls=":", zorder=1)  # replicated median used in all analyses
    ax.set_yticks(range(len(top)), labels, fontsize=7.5)
    ax.set_xlabel("Brier Index, difficulty-adjusted (higher is better); bars = published 95% CI")
    for name, _, col in groups:
        if name in present:
            ax.plot([], [], "o", color=col, label=name)
    ax.plot([], [], "D", color=INK, label="Superforecaster median (shaded: its 95% CI)")
    ax.plot([], [], ":", color=INK2, label=f"replicated median ({sup_rep:.2f})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, fontsize=8)
    ax.set_title("Top 20 external entries on the ForecastBench tournament leaderboard (14 Nov 2026)", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", visible=False)
    save(fig, "fig1_leaderboard_top20")


def fig2_h3():
    lb = pd.read_csv(DATA / "leaderboards" / "lb_tournament.csv")
    letters = team_letters(lb)
    sup = float(pd.read_csv(DERIVED / "h3_threshold.csv", index_col=0).iloc[0, 0])
    band = np.percentile(pd.read_csv(DERIVED / "h3_bootstrap_counts.csv")["super_boot"], [2.5, 97.5])
    full = pd.read_csv(DERIVED / "h3_entry_full.csv")
    cont = pd.read_csv(DERIVED / "h3_continuous.csv")
    cont = cont[cont["frac"] == 0.8].set_index("org")
    rows = []
    for org in cont.index:
        d = full[full["org"] == org]
        rows.append(dict(team=f"Team {letters.get(org, '?')} ({len(d)} entries)", best=d["bi"].max(),
                         held=cont.loc[org, "held"], lo=cont.loc[org, "held_lo"], hi=cont.loc[org, "held_hi"],
                         mean=float(brier_index(pd.Series([d["overall"].mean()])).iloc[0])))
    t2 = pd.DataFrame(rows).sort_values("best").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ax.axvspan(band[0], band[1], color=BAND, zorder=0, lw=0)
    for i, r in t2.iterrows():
        ax.plot([r["lo"], r["hi"]], [i - 0.12, i - 0.12], color=S3, lw=1.2, zorder=2)
        ax.scatter(r["held"], i - 0.12, color=S3, marker="^", s=42, edgecolor=SURF, linewidth=1, zorder=3,
                   label="best entry re-scored on held-out 20% (mean, 95% range over splits)" if i == 0 else None)
        ax.scatter(r["best"], i + 0.12, color=S2, marker="o", s=42, edgecolor=SURF, linewidth=1, zorder=3,
                   label="best entry as ranked" if i == 0 else None)
        ax.scatter(r["mean"], i + 0.12, color=S1, marker="s", s=36, edgecolor=SURF, linewidth=1, zorder=3,
                   label="team mean (display only)" if i == 0 else None)
    ax.axvline(sup, color=INK, lw=1, ls="--")
    ax.annotate(f"superforecaster median {sup:.2f}\n(shaded: its 95% interval)", (sup, len(t2) - 0.45),
                xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK, va="top")
    ax.set_yticks(range(len(t2)), t2["team"])
    ax.set_ylim(-0.6, len(t2) - 0.1)
    ax.set_xlabel("Brier Index (difficulty-adjusted, replicated)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=2, fontsize=7.2)
    ax.set_title("Best entry as ranked and re-scored on held-out questions", loc="left", fontsize=9.5, color=INK)
    ax.grid(axis="y", visible=False)
    save(fig, "figA1_h3_teams")


def fig3_policy():
    s = pd.read_csv(DERIVED / "rq4_policy_sim_extra.csv")
    pols = [("status quo (best entry)", S2, "o", "-", "status quo (best entry)"),
            ("primary chosen on prior round", S1, "s", "-", "pre-declared primary entry"),
            ("held-out re-score of best", S3, "^", (0, (4, 2)), "held-out next round"),
            ("winner's-curse-corrected best", INK2, "v", "-", "best minus expected selection gain"),
            ("team mean", S4, "D", "-", "team mean")]
    panels = [(-0.5, "team truly 0.5 points BELOW superforecasters\n(false claims)"),
              (0.5, "team truly 0.5 points ABOVE superforecasters\n(correct claims)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, (dlt, title) in zip(axes, panels):
        d = s[s["delta"] == dlt]
        for col, c, mk, ls, lab in pols:
            ax.plot(d["k"], d[col] * 100, color=c, lw=2, ls=ls, marker=mk, ms=5, label=lab)
        ax.set_xscale("log"); ax.set_xticks(d["k"], d["k"].astype(int)); ax.minorticks_off()
        ax.set_xlabel("entries submitted by the team (k, log scale)")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_ylim(-2, 102)
    axes[0].set_ylabel("chance the team appears above\nthe superforecaster median (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig3_policy_simulation")


def figA1_selection_fraction():
    """(a) H1 luck share by selection fraction, observed vs the calibrated model's prediction.
    (b) Board-wide: held-out drop of the top ten minus ten random entries, with the 1/f and
    1/sqrt(f) fits used to extrapolate to selection on all questions."""
    sf = pd.read_csv(DERIVED / "h1_selection_fraction.csv")
    pv = pd.read_csv(DERIVED / "rq4_validation_fraction.csv", index_col=0)
    bc = pd.read_csv(DERIVED / "board_selection_curve.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.7))
    x = sf["frac"] * 100
    a1.errorbar(x, sf["luck_share"] * 100, yerr=[(sf["luck_share"] - sf["lo"]) * 100, (sf["hi"] - sf["luck_share"]) * 100],
                fmt="o-", color=S1, lw=1.5, ms=5, label="observed (95% CI)")
    a1.plot(pv.index * 100, pv["predicted"] * 100, "s--", color=S2, lw=1.2, ms=4, label="calibrated model")
    a1.plot(pv.index * 100, pv["fitted_spread"] * 100, ":", color=S2, lw=1.2, label="spread fitted to this curve")
    a1.set_xlabel("share of questions used to pick (%)"); a1.set_ylabel("luck share of edge (%)")
    a1.set_ylim(0, 50); a1.legend(loc="upper right", fontsize=7.5)
    a1.set_title("(a) Within a team (H1)", loc="left", fontsize=9.5, color=INK)
    f = bc["f"].to_numpy()
    a2.plot(f * 100, bc["net"], "o-", color=S1, lw=1.5, ms=5, label="top ten minus ten random")
    a2.plot(f * 100, bc["rand"], "^:", color=INK2, lw=1, ms=4, label="ten random (null)")
    b, a = np.polyfit(np.log(1 / f), np.log(bc["net"].to_numpy()), 1)
    ff = np.linspace(0.5, 1.0, 50)
    a2.plot(ff * 100, np.exp(a) * (1 / ff) ** b, "--", color=S2, lw=1.2, label="log-log fit, extrapolated")
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_xlabel("share of questions used to pick (%)"); a2.set_ylabel("held-out drop (index points)")
    a2.legend(loc="upper right", fontsize=7.5)
    a2.set_title("(b) Across the board", loc="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    save(fig, "fig2_selection")


def figA2_h2():
    pr = pd.read_csv(DERIVED / "h2_round_pairs.csv")
    d = pr[pr["external"]]
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    x, y = d["adv_t"] * 156.25, d["adv_t1"] * 156.25  # index points near the top of the board
    xlim = [-0.15, x.max() + 0.25]; ylim = [y.min() - 0.25, y.max() + 0.25]
    ax.plot([0, xlim[1]], [0, xlim[1]], color=INK2, lw=1, ls="--")
    ax.annotate("no regression (y = x)", (xlim[1], xlim[1]), xytext=(-4, 6), textcoords="offset points",
                ha="right", fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.scatter(x, y, s=28, color=S1, edgecolor=SURF, linewidth=1, zorder=3)
    ax.scatter([x.mean()], [y.mean()], s=70, marker="D", color=S2, edgecolor=SURF, linewidth=1.5, zorder=4)
    ax.annotate(f"mean: {x.mean():.2f} → {y.mean():.2f}", (x.mean(), y.mean()), xytext=(8, -4),
                textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("advantage of round-t best variant in round t\n(index points)")
    ax.set_ylabel("same variant's advantage in round t+1")
    ax.set_title(f"H2: external teams ({len(d)} round pairs)", loc="left", fontsize=9.5, color=INK)
    save(fig, "figA2_h2_cross_round")


if __name__ == "__main__":
    for p in FIGS.glob("f[0-9]_*"):
        p.unlink()
    fig1_leaderboard(); fig2_h3(); fig3_policy(); figA1_selection_fraction(); figA2_h2()
    print("figures written to", FIGS, sorted(p.name for p in FIGS.glob("*.png")))
