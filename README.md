# What Is a Forecasting Leaderboard Rank Worth?

**Selection and noise in ForecastBench's AI-versus-superforecaster comparison**

Juhong Park (Massachusetts Institute of Technology) — Apart Research AI x Epistemics Research Sprint, 13–15 Nov 2026

## Question

In July 2026 ForecastBench's maintainers concluded that AI forecasters have likely reached parity with superforecasters, while 17 submissions ranked above them. This project asks what such a rank is worth as evidence. It replicates ForecastBench's scoring from public data and measures how much the single 2024 human baseline costs, how much selection lifts the top of the board on held-out questions, and how display policies would trade false outperformance claims against missed leads. The report is `report/report.md`; the submitted PDF is `submission/report.pdf`, with slides in `submission/slides.pdf`.

The pre-registered hypotheses, estimands and analysis plan are in [PLAN.md](PLAN.md), committed before any confirmatory analysis (14 November 2026, 21:53 EST). Predictions for questions that resolve after the 14 November 2026 snapshot are registered in [PROSPECTIVE.md](PROSPECTIVE.md).

## Repository layout

| Path | Contents |
|---|---|
| `PLAN.md` | Pre-registration: hypotheses, estimands and analysis plan |
| `PROSPECTIVE.md` | Prospective registration of predictions for later-resolving questions |
| `analysis/` | Analysis code; `run_all.sh` rebuilds every number from the raw snapshot |
| `analysis/feasibility/` | `index.py` builds the question-level table from the raw snapshot; `multisub.py` is the feasibility check run before registration |
| `figures/` | Figures (PDF and PNG) |
| `report/` | Report source (`report.md`), Word version (`report.docx`) and its builder (`build_docx.py`) |
| `submission/` | Submitted PDFs: the report in the official sprint template (`report.pdf`) and the slides (`slides.pdf`) |
| `review/` | `check_report.py` checks every number in the report against the code output; `render_tables.py` renders the report's tables |
| `SNAPSHOT.sha256` | SHA-256 hashes of the input files |
| `data/` | Not tracked; see below |

## Data (not tracked in git)

- ForecastBench processed forecast sets, CC BY-SA 4.0:
  `https://www.forecastbench.org/assets/data/processed-forecast-sets/processed_forecast_sets.tar.gz`
- ForecastBench leaderboard CSVs:
  `https://github.com/forecastingresearch/forecastbench-datasets` (`leaderboards/csv/`)
- ForecastBench scoring code (MIT License), used to replicate the scoring:
  `https://github.com/forecastingresearch/forecastbench/blob/main/src/leaderboard/main.py`

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Disclosure

- **Pre-existing work**: none. All code in this repository was written during the sprint (from 14 Nov 2026).
- **Contributions**: Juhong Park set the direction of the research idea and reviewed and checked the work. Claude, used through Claude Code, wrote the code; replicated ForecastBench's scoring; ran the tests, the exploratory analyses and the simulation; wrote the code for the prospective test; produced and analysed the figures and tables; and drafted and revised the pre-registration, the report and the slides. All reported numbers come from the code in this repository.
- **Teams**: tournament teams are anonymized in all outputs.

## License

The code in this repository is released under the MIT License (see `LICENSE`). ForecastBench's data are licensed CC BY-SA 4.0 and are not redistributed here; material adapted from them keeps that licence.
