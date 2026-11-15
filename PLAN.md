## 4. Pre-registration (English, frozen before confirmatory analysis)

*This section is committed to git before any confirmatory analysis is run. An exploratory estimate of the best-vs-mean gap (median Brier difference ≈ 0.0029) was seen during feasibility checks and is disclosed as such.*

### Data and filters
- Source: ForecastBench processed forecast sets (CC BY-SA 4.0), all rounds available on 2026-11-14.
- Unit: forecast row = (team, entry, round, question id, horizon).
- Keep rows that are resolved, binary, non-imputed, and non-combination questions.
- Keep entries that are `leaderboard_eligible`.
- Team = `organization` field. ForecastBench's own baseline runs are a separate team, "ForecastBench".
- Superforecaster and public medians: taken from the human forecast sets / leaderboard rows.

### Scoring
- Primary metric: replicate ForecastBench's tournament scoring.
  - Difficulty-adjusted Brier via the two-way fixed-effects model in `src/leaderboard/main.py`
  - Brier Index = (1 − √Brier) × 100
- Validation target: Pearson r > 0.95 between replicated and published Overall scores for leaderboard entries.
- Fallback if the target is not met: raw Brier on identical question sets (no fixed effects). Report which metric was used.

### Confirmatory hypotheses
Core estimand: **selection inflation** = (best entry's advantage over its team mean, measured on the questions used to select it) − (the same entry's advantage measured on held-out questions). Inflation > 0 means part of the best entry's apparent edge is luck from best-of-k selection.

- **H1 (within-round inflation, split-half)**: For team-rounds with ≥2 entries scored on an identical question set:
  - Procedure: split question ids at random into halves A and B; select the best entry on A; measure its advantage over the team mean on A and on B. Repeat over 200 random splits and average.
  - Hypothesis: the mean inflation pooled across team-rounds is > 0.
  - Also report the luck share = inflation / in-sample advantage.
  - Inference: one-sided, cluster bootstrap over team-rounds, α = 0.05.
  - Splits are by question id (all horizons of a question go to the same half).
- **H2 (cross-round regression to the mean)**: For variants of the same team appearing in consecutive rounds, the round-t best variant's advantage over its teammates shrinks in round t+1.
  - Estimand: mean shrinkage > 0.
  - Inference: cluster bootstrap by team, α = 0.05.
  - Also report the Spearman correlation of within-team variant ranks across rounds.
- **Reference null (descriptive)**: the expected best-minus-mean gap if all of a team's entries had equal skill, obtained by permuting entry labels within team × question. Comparing the observed gap to this null shows how much of the gap is consistent with pure noise.
- **H3 (consequence)**: The number of teams counted as "above the superforecaster median" is lower under selection-corrected scoring than under the published entry-level ranking (where a team counts if any of its entries is above). Corrected scoring means (a) the team-mean entry and (b) the held-out score of the selected best entry.
  - Descriptive, with bootstrap CI on the counts.
  - The "Supers > forecaster" p-values are additionally reported with Holm correction across all entries tested.

### Policy evaluation (RQ4)
- Simulation: teams of equal true skill.
  - k ∈ {1, 2, 3, 6, 9, 22} entries per team
  - Noise calibrated to the observed per-question Brier variance
- Outcomes: expected rank inflation, and the false "beats superforecasters" rate.
- Policies compared: entry-level best (status quo), team mean, pre-declared primary entry, next-round holdout.
- The same policies are then applied to the real data.

### Robustness (reported regardless of outcome)
- Dataset vs market questions separately
- Minimum questions per entry: 500 and 1,000
- Raw vs difficulty-adjusted Brier
- Excluding the ForecastBench baseline team
- Excluding the single largest team

### What would count against our thesis
- H1 not supported, or H2 shows little shrinkage (i.e., best variants persist).
- In that case the paper reports that the leaderboard is robust to best-of-k selection. The audit method is still offered as the contribution.
