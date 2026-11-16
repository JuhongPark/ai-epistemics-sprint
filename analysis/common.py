"""Shared loading and preprocessing that mirrors ForecastBench's leaderboard pipeline.

Reference: forecastbench/src/leaderboard/main.py (MIT License), functions get_df_info,
score_models, two_way_fixed_effects, rescale_difficulty_adjusted_brier,
apply_brier_index_transform. A local copy is in data/ref/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DERIVED = DATA / "derived"
FIGS = ROOT / "figures"

DATASET_SOURCES = {"acled", "dbnomics", "fred", "wikipedia", "yfinance"}
MARKET_SOURCES = {"manifold", "metaculus", "polymarket", "infer"}
BENCHMARK = "ForecastBench"
IMPUTED_CUTOFF_PCT = 5


def load_rows(columns=None) -> pd.DataFrame:
    """All forecast rows from the processed forecast sets (one row per file x question)."""
    df = pd.read_parquet(DATA / "processed" / "long.parquet", columns=columns)
    idx = pd.read_csv(DATA / "processed" / "index.csv", usecols=["file", "lb"])
    df = df.merge(idx, on="file", how="left")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ForecastBench's per-file filters and build question keys.

    - drop combination questions
    - drop files over the 5% imputed cutoff (overall, market, or resolved dataset),
      except ForecastBench's own dummy/comparison models
    - keep resolved dataset questions and resolved market questions
    - question_pk: dataset = due_source_id_horizon ; market = due_source_id
    """
    df = df[df["lb"].fillna(True)].copy()
    direction = df["direction"].astype(str)
    df = df[(direction == "") | (direction == "None") | (direction == "nan")]

    is_dataset = df["source"].isin(DATASET_SOURCES)
    is_market = df["source"].isin(MARKET_SOURCES)
    resolved = df["resolved"].astype(bool)

    dummy = (df["org"] == BENCHMARK) & (df["morg"] == BENCHMARK)
    g = df.assign(
        imp=df["imputed"].astype(float),
        imp_m=np.where(is_market, df["imputed"].astype(float), np.nan),
        imp_d=np.where(is_dataset & resolved, df["imputed"].astype(float), np.nan),
    ).groupby("file")[["imp", "imp_m", "imp_d"]].mean() * 100
    bad = g[(g["imp"] > IMPUTED_CUTOFF_PCT) | (g["imp_m"] > IMPUTED_CUTOFF_PCT)
            | (g["imp_d"] > IMPUTED_CUTOFF_PCT)].index
    df = df[~df["file"].isin(bad) | dummy]

    is_dataset = df["source"].isin(DATASET_SOURCES)
    is_market = df["source"].isin(MARKET_SOURCES)
    resolved = df["resolved"].astype(bool)
    df = df[(is_dataset & resolved) | (is_market & resolved)].copy()

    df["qtype"] = np.where(df["source"].isin(DATASET_SOURCES), "dataset", "market")
    due = pd.to_datetime(df["due"])
    res = pd.to_datetime(df["res_date"])
    df["horizon"] = (res - due).dt.days
    base = df["due"].astype(str) + "_" + df["source"].astype(str) + "_" + df["id"].astype(str)
    df["question_pk"] = np.where(df["qtype"] == "dataset", base + "_" + df["horizon"].astype(str), base)
    df["brier"] = (df["forecast"] - df["resolved_to"]) ** 2
    df["model_pk"] = df["org"].astype(str) + "_" + df["morg"].astype(str) + "_" + df["model"].astype(str)
    return df


def load_qfe(path=DATA / "ref" / "qfe_tournament.json") -> pd.Series:
    """Published per-question fixed effects for the tournament leaderboard."""
    q = pd.DataFrame(json.load(open(path)))
    is_ds = q["source"].isin(DATASET_SOURCES)
    base = q["forecast_due_date"].astype(str) + "_" + q["source"].astype(str) + "_" + q["id"].astype(str)
    hz = q["horizon"].fillna(-1).astype(int).astype(str)
    q["question_pk"] = np.where(is_ds, base + "_" + hz, base)
    return q.set_index("question_pk")["question_fixed_effect"]


def brier_index(adj_brier: pd.Series) -> pd.Series:
    return (1 - np.sqrt(adj_brier)) * 100


def record(prefix: str, **values) -> None:
    """Store headline numbers in data/derived/metrics.json so review/check_report.py can
    verify every numeric claim in the report against code output (no hand-copied numbers)."""
    DERIVED.mkdir(parents=True, exist_ok=True)
    path = DERIVED / "metrics.json"
    data = json.load(open(path)) if path.exists() else {}
    for k, v in values.items():
        data[f"{prefix}.{k}"] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
    json.dump(data, open(path, "w"), indent=1, sort_keys=True)


ROUND_COLS = ["file", "model", "morg", "org", "due", "id", "source", "direction", "res_date",
              "resolved_to", "resolved", "forecast", "imputed"]


def round_scores(include_benchmark=False) -> pd.DataFrame:
    """Difficulty-adjusted score of every (round, team, entry) on that round's questions only,
    on the leaderboard's global scale: the same shift as the leaderboard (always-0.5 scores 0.25
    over all of its questions), so round-level scores are comparable with the superforecaster
    score. Columns: due, org, model, dataset, market, overall, bi."""
    df = preprocess(load_rows(ROUND_COLS))
    df["qfe"] = df["question_pk"].map(load_qfe())
    df = df[df["qfe"].notna()]
    df["adj"] = df["brier"] - df["qfe"]
    if not include_benchmark:
        df = df[(df["org"] != BENCHMARK) | (df["model"] == "Always 0.5")]
    g = df.groupby(["due", "org", "model", "qtype"])["adj"].mean().unstack("qtype")
    g["overall"] = (g["dataset"] + g["market"]) / 2
    g = g.dropna(subset=["overall"]).reset_index()
    a05 = df[df["model"] == "Always 0.5"].groupby("qtype")["adj"].mean()
    g["overall"] = g["overall"] + 0.25 - (a05["dataset"] + a05["market"]) / 2
    if not include_benchmark:
        g = g[g["org"] != BENCHMARK].copy()
    g["bi"] = brier_index(g["overall"])
    return g.dropna(subset=["bi"])
