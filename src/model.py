"""Ensemble machine-learning layer: predict playoff qualification, how far a
team advances, and championship odds from the composite feature table.

Evaluation uses a time-based split (train on earlier seasons, test on the
most recent ones) rather than a random shuffle, since the real use case is
forecasting a *future* postseason from past data -- a random split would let
the model see, e.g., 2023 data while "predicting" 2021, which overstates
real-world accuracy.

Three targets are modeled:
  - made_playoffs   (binary):            logistic regression + random forest
                                          + gradient boosting, soft-voted
  - rounds_won      (ordinal 0-5):       random forest + gradient boosting,
                                          averaged probabilities
  - champion        (binary, rare event): gradient boosting only (too few
                                          positive examples for a stable
                                          multi-model ensemble)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (
    CONFIDENCE_DECAY_RATE, CONFIDENCE_FLOOR, MIN_TRAIN_SEASONS, N_ROUND_CLASSES,
    OUTPUT_DIR, PRESENT_YEAR, RANDOM_SEED,
)

FEATURE_COLS = [
    "composite_team_rating", "srs", "net_rating", "pyth_win_pct", "win_pct",
    "avg_three_att_rate", "avg_ts_pct", "avg_tov_per36", "pace",
    "team_injury_risk", "team_playoff_dropoff_score", "roster_continuity",
    "coach_stability_index", "home_court_index", "big_market_flag",
]


def build_ml_dataset(composite_team_df: pd.DataFrame, playoff_results: pd.DataFrame) -> pd.DataFrame:
    """Attach playoff outcome labels to the composite team-season table."""
    labels = playoff_results[["season", "year", "team", "made_playoffs", "rounds_won", "champion"]]
    return composite_team_df.merge(labels, on=["season", "year", "team"], how="inner")


def time_based_split(df: pd.DataFrame, test_years: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[~df.year.isin(test_years)].copy()
    test = df[df.year.isin(test_years)].copy()
    return train, test


@dataclass
class TrainedModels:
    scaler: StandardScaler
    playoffs_models: dict = field(default_factory=dict)
    rounds_models: dict = field(default_factory=dict)
    champion_model: XGBClassifier | None = None
    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))
    metrics: dict = field(default_factory=dict)

    def save(self, path=OUTPUT_DIR / "trained_models.joblib") -> None:
        joblib.dump(self, path)

    def predict_playoff_probability(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X[self.feature_cols])
        probs = [
            self.playoffs_models["logreg"].predict_proba(Xs)[:, 1],
            self.playoffs_models["rf"].predict_proba(X[self.feature_cols])[:, 1],
            self.playoffs_models["xgb"].predict_proba(X[self.feature_cols])[:, 1],
        ]
        return np.mean(probs, axis=0)

    def predict_rounds_won_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = [
            self.rounds_models["rf"].predict_proba(X[self.feature_cols]),
            self.rounds_models["xgb"].predict_proba(X[self.feature_cols]),
        ]
        avg = np.mean(probs, axis=0)
        return avg / avg.sum(axis=1, keepdims=True)  # renormalize floating-point drift

    def predict_champion_probability(self, X: pd.DataFrame) -> np.ndarray:
        return self.champion_model.predict_proba(X[self.feature_cols])[:, 1]


def _fit_all(X_train: pd.DataFrame, y_playoffs: pd.Series, y_rounds: pd.Series, y_champ: pd.Series,
             seed: int, n_estimators: int = 300) -> TrainedModels:
    """Fit the full ensemble (all three targets) on one training slice.
    Shared by train_and_evaluate (one fixed split) and walk_forward_predictions
    (many expanding-window splits, one per season)."""
    scaler = StandardScaler().fit(X_train)
    Xtr_s = scaler.transform(X_train)

    logreg = LogisticRegression(max_iter=2000).fit(Xtr_s, y_playoffs)
    rf = RandomForestClassifier(n_estimators=max(n_estimators, 100) + 100, max_depth=6,
                                 random_state=seed, n_jobs=-1).fit(X_train, y_playoffs)
    xgb = XGBClassifier(n_estimators=n_estimators, max_depth=3, learning_rate=0.05, random_state=seed,
                         eval_metric="logloss", n_jobs=-1).fit(X_train, y_playoffs)

    rf_rounds = RandomForestClassifier(n_estimators=max(n_estimators, 100) + 100, max_depth=6,
                                        random_state=seed, n_jobs=-1).fit(X_train, y_rounds)
    xgb_rounds = XGBClassifier(n_estimators=n_estimators, max_depth=3, learning_rate=0.05, random_state=seed,
                                objective="multi:softprob", num_class=N_ROUND_CLASSES, eval_metric="mlogloss",
                                n_jobs=-1).fit(X_train, y_rounds)

    champ_model = XGBClassifier(n_estimators=max(n_estimators - 100, 100), max_depth=3, learning_rate=0.05,
                                 random_state=seed, eval_metric="logloss", n_jobs=-1,
                                 scale_pos_weight=(y_champ == 0).sum() / max((y_champ == 1).sum(), 1)
                                 ).fit(X_train, y_champ)

    return TrainedModels(
        scaler=scaler,
        playoffs_models={"logreg": logreg, "rf": rf, "xgb": xgb},
        rounds_models={"rf": rf_rounds, "xgb": xgb_rounds},
        champion_model=champ_model,
        feature_cols=list(X_train.columns),
    )


def train_and_evaluate(ml_dataset: pd.DataFrame, test_years: tuple[int, ...] = (2022, 2023, 2024),
                        seed: int = RANDOM_SEED, feature_cols: list[str] = FEATURE_COLS) -> TrainedModels:
    train, test = time_based_split(ml_dataset, test_years)
    X_train, X_test = train[feature_cols], test[feature_cols]

    trained = _fit_all(X_train, train.made_playoffs.astype(int), train.rounds_won.astype(int),
                        train.champion.astype(int), seed=seed, n_estimators=300)

    y_te = test.made_playoffs.astype(int)
    ensemble_proba = trained.predict_playoff_probability(X_test)
    metrics: dict = {"made_playoffs": {
        "accuracy": accuracy_score(y_te, ensemble_proba > 0.5),
        "log_loss": log_loss(y_te, ensemble_proba),
        "roc_auc": roc_auc_score(y_te, ensemble_proba),
    }}

    y_te_r = test.rounds_won.astype(int)
    rounds_proba = trained.predict_rounds_won_proba(X_test)
    metrics["rounds_won"] = {
        "accuracy": accuracy_score(y_te_r, rounds_proba.argmax(axis=1)),
        "log_loss": log_loss(y_te_r, rounds_proba, labels=list(range(N_ROUND_CLASSES))),
    }

    y_te_c = test.champion.astype(int)
    champ_proba = trained.predict_champion_probability(X_test)
    metrics["champion"] = {
        "log_loss": log_loss(y_te_c, champ_proba, labels=[0, 1]),
        "roc_auc": roc_auc_score(y_te_c, champ_proba) if y_te_c.nunique() > 1 else float("nan"),
    }

    trained.metrics = metrics
    return trained


def walk_forward_predictions(ml_dataset: pd.DataFrame, min_train_seasons: int = MIN_TRAIN_SEASONS,
                              seed: int = RANDOM_SEED, feature_cols: list[str] = FEATURE_COLS,
                              n_estimators: int = 150, max_year: int | None = None) -> pd.DataFrame:
    """Genuine out-of-sample prediction for every eligible season: retrain on
    all strictly-earlier years and predict the target year, one year at a
    time (expanding window). This is what lets *any* season -- not just a
    fixed held-out tail -- be compared "model's pre-outcome prediction" vs
    "what actually happened," and it's the same mechanism used to project a
    future season (which simply has no "actual" to compare against yet).
    """
    years = sorted(ml_dataset.year.unique())
    if max_year is not None:
        years = [y for y in years if y <= max_year]

    rows = []
    for i, year in enumerate(years):
        if i < min_train_seasons:
            continue
        train = ml_dataset[ml_dataset.year < year]
        target = ml_dataset[ml_dataset.year == year]
        if train.empty or target.empty:
            continue

        X_train = train[feature_cols]
        trained = _fit_all(X_train, train.made_playoffs.astype(int), train.rounds_won.astype(int),
                            train.champion.astype(int), seed=seed, n_estimators=n_estimators)

        X_target = target[feature_cols]
        made_playoffs_prob = trained.predict_playoff_probability(X_target)
        rounds_proba = trained.predict_rounds_won_proba(X_target)
        champion_prob = trained.predict_champion_probability(X_target)

        out = target[["season", "year", "team", "made_playoffs", "rounds_won", "champion"]].copy()
        out["made_playoffs_prob"] = made_playoffs_prob
        out["expected_rounds_won"] = rounds_proba @ np.arange(N_ROUND_CLASSES)
        out["champion_prob"] = champion_prob
        rows.append(out)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def year_accuracy_summary(walk_forward_df: pd.DataFrame) -> pd.DataFrame:
    """One retrospective "how good was the prediction" score per season,
    blending three components into a single 0-1 accuracy: made-playoffs hit
    rate (easy -- win totals nearly determine seeding, so this is a sanity
    floor), how close the predicted expected rounds-won was to the actual
    (the harder, more informative part), and the probability the model
    assigned to the team that actually won it all (the single hardest call).
    """
    rows = []
    for year, g in walk_forward_df.groupby("year"):
        playoff_hit_rate = accuracy_score(g.made_playoffs.astype(int), g.made_playoffs_prob > 0.5)
        rounds_mae = (g.rounds_won.astype(int) - g.expected_rounds_won).abs().mean()
        rounds_closeness = float(np.clip(1 - rounds_mae / (N_ROUND_CLASSES - 1), 0, 1))
        champ_row = g[g.champion]
        champ_prob_assigned = float(champ_row.champion_prob.iloc[0]) if len(champ_row) else 0.0
        blended = 0.3 * playoff_hit_rate + 0.4 * rounds_closeness + 0.3 * champ_prob_assigned
        champion_predicted = g.loc[g.champion_prob.idxmax(), "team"]
        rows.append({
            "year": year,
            "playoff_hit_rate": round(playoff_hit_rate, 3),
            "rounds_closeness": round(rounds_closeness, 3),
            "champion_prob_assigned_to_actual": round(champ_prob_assigned, 3),
            "accuracy_score": round(blended, 3),
            "champion_actual": champ_row.team.iloc[0] if len(champ_row) else None,
            "champion_predicted": champion_predicted,
            "champion_correct": bool(len(champ_row) and champion_predicted == champ_row.team.iloc[0]),
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def future_confidence(year: int, baseline_accuracy: float, present_year: int = PRESENT_YEAR,
                       decay_rate: float = CONFIDENCE_DECAY_RATE, floor: float = CONFIDENCE_FLOOR) -> float:
    """Estimated (not measured) confidence for a season beyond present_year:
    the measured historical baseline, decayed per year further out. This is
    explicitly a modeled estimate of compounding uncertainty (unknown
    rookies, injuries, trades, aging), never presented as a measured
    accuracy the way year_accuracy_summary's score is for history.
    """
    years_out = max(year - present_year, 0)
    return float(max(baseline_accuracy * ((1 - decay_rate) ** years_out), floor))


def feature_importance_table(trained: TrainedModels) -> pd.DataFrame:
    """Average feature importance across the tree-based made_playoffs models."""
    rf_imp = trained.playoffs_models["rf"].feature_importances_
    xgb_imp = trained.playoffs_models["xgb"].feature_importances_
    return pd.DataFrame({
        "feature": trained.feature_cols,
        "rf_importance": rf_imp,
        "xgb_importance": xgb_imp,
        "avg_importance": (rf_imp + xgb_imp) / 2,
    }).sort_values("avg_importance", ascending=False).reset_index(drop=True)
