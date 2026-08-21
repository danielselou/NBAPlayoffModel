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

from config import N_ROUND_CLASSES, OUTPUT_DIR, RANDOM_SEED

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
        return np.mean(probs, axis=0)

    def predict_champion_probability(self, X: pd.DataFrame) -> np.ndarray:
        return self.champion_model.predict_proba(X[self.feature_cols])[:, 1]


def train_and_evaluate(ml_dataset: pd.DataFrame, test_years: tuple[int, ...] = (2022, 2023, 2024),
                        seed: int = RANDOM_SEED, feature_cols: list[str] = FEATURE_COLS) -> TrainedModels:
    train, test = time_based_split(ml_dataset, test_years)
    X_train, X_test = train[feature_cols], test[feature_cols]

    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

    metrics: dict = {}

    # --- made_playoffs: logreg + random forest + gradient boosting ---
    y_tr, y_te = train.made_playoffs.astype(int), test.made_playoffs.astype(int)
    logreg = LogisticRegression(max_iter=2000).fit(Xtr_s, y_tr)
    rf = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=seed, n_jobs=-1).fit(X_train, y_tr)
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed,
                         eval_metric="logloss", n_jobs=-1).fit(X_train, y_tr)

    ensemble_proba = np.mean([
        logreg.predict_proba(Xte_s)[:, 1], rf.predict_proba(X_test)[:, 1], xgb.predict_proba(X_test)[:, 1],
    ], axis=0)
    metrics["made_playoffs"] = {
        "accuracy": accuracy_score(y_te, ensemble_proba > 0.5),
        "log_loss": log_loss(y_te, ensemble_proba),
        "roc_auc": roc_auc_score(y_te, ensemble_proba),
    }

    # --- rounds_won: ordinal 0-5, modeled as multiclass ---
    y_tr_r, y_te_r = train.rounds_won.astype(int), test.rounds_won.astype(int)
    rf_rounds = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=seed, n_jobs=-1).fit(X_train, y_tr_r)
    xgb_rounds = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed,
                                objective="multi:softprob", num_class=N_ROUND_CLASSES, eval_metric="mlogloss",
                                n_jobs=-1).fit(X_train, y_tr_r)
    rounds_proba = np.mean([rf_rounds.predict_proba(X_test), xgb_rounds.predict_proba(X_test)], axis=0)
    rounds_proba = rounds_proba / rounds_proba.sum(axis=1, keepdims=True)  # renormalize fp drift
    metrics["rounds_won"] = {
        "accuracy": accuracy_score(y_te_r, rounds_proba.argmax(axis=1)),
        "log_loss": log_loss(y_te_r, rounds_proba, labels=list(range(N_ROUND_CLASSES))),
    }

    # --- champion: rare binary event, gradient boosting only ---
    y_tr_c, y_te_c = train.champion.astype(int), test.champion.astype(int)
    champ_model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed,
                                 eval_metric="logloss", n_jobs=-1,
                                 scale_pos_weight=(y_tr_c == 0).sum() / max((y_tr_c == 1).sum(), 1)).fit(X_train, y_tr_c)
    champ_proba = champ_model.predict_proba(X_test)[:, 1]
    metrics["champion"] = {
        "log_loss": log_loss(y_te_c, champ_proba, labels=[0, 1]),
        "roc_auc": roc_auc_score(y_te_c, champ_proba) if y_te_c.nunique() > 1 else float("nan"),
    }

    return TrainedModels(
        scaler=scaler,
        playoffs_models={"logreg": logreg, "rf": rf, "xgb": xgb},
        rounds_models={"rf": rf_rounds, "xgb": xgb_rounds},
        champion_model=champ_model,
        feature_cols=list(feature_cols),
        metrics=metrics,
    )


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
