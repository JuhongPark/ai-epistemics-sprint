# Prospective registration: predictions for questions resolving after the 14 November 2026 snapshot

Registered on 15 November 2026 (EST). This file is committed before any ForecastBench data newer than the 14 November 2026 snapshot has been downloaded or inspected. Input hashes for that snapshot are in `SNAPSHOT.sha256`.

## Why

Two of the report's estimates are exploratory or rest on extrapolation:
- **Selection across the board:** the ten best entries lose about 1.2 index points on held-out questions relative to random entries, and an extrapolated 0.7 points when picked on all questions.
- **Above-median ranks:** these are weak evidence, because three teams at exact parity would often produce the board we see.

Questions that resolve after the snapshot give a held-out test that nobody, including us, could select on.

## Frozen objects

- **Entries.** The external entries on the 14 November 2026 tournament leaderboard (`data/leaderboards/lb_tournament.csv`, hash in `SNAPSHOT.sha256`), scored with the replicated pipeline at the commit that adds this file.
- **The seven.** The seven external entries with replicated scores above the replicated superforecaster median (68.19). They are listed in `data/derived/h3_entry_full.csv` (score > 68.19).
- **The top ten.** The ten external entries with the highest replicated scores on 14 November.
- **New questions.** For each frozen entry, the questions from that entry's own forecast sets whose resolution date is after 14 November 2026. They are scored with the difficulty effects published in the later snapshot, using the code at this commit (`analysis/common.py`, `analysis/04_h3_consequence.py`). Only the input files change.

## Predictions

1. **Selection across the board.** On new questions, the top ten lose more of their 14 November score than ten random external entries do. We average random draws of ten over 1,000 draws. Central prediction: a difference of about 0.7 index points. The prediction fails if the difference is zero or negative.
2. **Above-median entries.** On new questions, at most three of the seven score above the superforecaster median (68.19), and their mean gap is below their 14 November mean gap (+0.45 points). The prediction fails if four or more stay above, or if the mean gap does not fall.
3. **Within-team selection (replication of H1).** The registered H1 procedure, run on external team-rounds dated after 14 November 2026 once at least 30 such team-rounds have resolved questions, gives a luck share between 10% and 35%.

## Analysis rules

- Tests are run once, when every frozen entry with a round in June–August 2026 has at least 200 newly resolved questions, or on 1 March 2027, whichever comes first.
- Entries without new questions are dropped and reported.
- No other selection is made.
- Results are reported whether they confirm or contradict the predictions.

## Addendum, 15 November 2026 (written before any data newer than the 14 November snapshot was downloaded)

The predictions, thresholds and analysis rules above are unchanged. This addendum fixes the implementation and adds secondary analyses, both before any newer data exists.

**Implementation.** The test is `analysis/13_prospective.py --snapshot DIR`, where DIR holds the later snapshot built the same way as `data/` (see the script's docstring). It prints the trigger status and stops until the trigger is met. Then it runs once and writes `DIR/prospective_result.json`; a second run refuses. Details fixed in the script:
- Each board uses one constant scale shift, so always-0.5 scores 0.25 over the whole board.
- A frozen entry's new questions are its forecasts in the rounds on the 14 November board that resolve after 14 November.
- An entry needs both question types among its new questions; other entries are dropped and reported.
- The random tens of prediction 1 are drawn from frozen entries that have new questions.
- Prediction 2 also reports the superforecaster median re-scored in the later snapshot. The verdict uses 68.19 as registered.

**Rehearsal.** We rehearsed the procedure on the 14 November snapshot by freezing the board on 1 September 2026. This is exploratory and uses data already analysed (report §4.2). Nine of that board's top ten came from one round (7 June). Most of their drop relative to random entries came from that round as a whole, not from selection within it. A comparison with random entries therefore mixes the two.

**Secondary analyses** (reported next to prediction 1, which is still judged as registered):
- (a) Round-matched comparison: each top-ten entry's change minus the mean change of the non-top-ten entries of its own round.
- (b) Round level: the round holding most of the top ten, against rounds whose new questions are of the same kinds (the same dataset horizons, plus markets). Measures: its lead over them on 14 November and on new questions, and the Naive Forecaster's lead in the same rounds.
