"""Machine Learning Engine for Multi-Regime SAR Classification & Anomaly Detection.

Combines LightGBM gradient boosted decision trees on physical InSAR/PolSAR descriptors
with IsolationForest unsupervised geophysical anomaly scoring.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sar_motionlens.synthetic import REGIME_CLASSES

SAR_FEATURE_COLUMNS: list[str] = [
    "incidence_angle_deg",
    "los_velocity_mm_yr",
    "los_seasonal_amp_mm",
    "los_harmonic_r2",
    "sigma0_co_mean_db",
    "sigma0_co_std_db",
    "sigma0_cross_mean_db",
    "sigma0_cross_std_db",
    "pol_ratio_db_mean",
    "rvi_mean",
    "rvi_seasonal_amp",
    "pol_entropy_mean",
    "coherence_mean",
    "coherence_min",
]


@dataclass
class SARModelEvaluation:
    """Container for trained SAR regime classifier, CV metrics, and enriched predictions."""
    model: LGBMClassifier
    accuracy: float
    macro_f1: float
    conf_matrix: np.ndarray
    feature_importance_df: pl.DataFrame
    enriched_site_df: pl.DataFrame


def train_and_evaluate_sar_classifier(
    site_df: pl.DataFrame,
    seed: int = 42,
) -> SARModelEvaluation:
    """Train LightGBM on physical SAR time-series descriptors with 5-fold Stratified CV
    and compute IsolationForest geophysical anomaly scores.
    """
    x_matrix = site_df.select(SAR_FEATURE_COLUMNS).to_numpy()
    y_labels = site_df["regime_label"].to_numpy()

    clf = LGBMClassifier(
        n_estimators=120,
        learning_rate=0.06,
        num_leaves=15,
        min_child_samples=8,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_preds = cross_val_predict(clf, x_matrix, y_labels, cv=cv, method="predict")
    oof_probs = cross_val_predict(clf, x_matrix, y_labels, cv=cv, method="predict_proba")

    acc = float(accuracy_score(y_labels, oof_preds))
    macro_f1 = float(f1_score(y_labels, oof_preds, average="macro"))
    c_mat = confusion_matrix(y_labels, oof_preds)

    # Fit final model on full dataset for real-time interactive inference & feature importances
    clf.fit(x_matrix, y_labels)
    importances = clf.feature_importances_.astype(np.float64)
    imp_pct = (importances / np.maximum(np.sum(importances), 1e-9)) * 100.0

    fi_df = (
        pl.DataFrame({
            "feature": SAR_FEATURE_COLUMNS,
            "importance_gain_pct": imp_pct,
            "importance_pct": imp_pct,
        })
        .sort("importance_gain_pct", descending=True)
    )

    # Unsupervised anomaly detector for extreme kinematics or sudden coherence collapse
    iso = IsolationForest(contamination=0.08, random_state=seed)
    iso.fit(x_matrix)
    raw_anom = -iso.score_samples(x_matrix)
    anom_norm = (raw_anom - np.min(raw_anom)) / (np.max(raw_anom) - np.min(raw_anom) + 1e-9) * 100.0

    pred_regime_names = [REGIME_CLASSES[int(idx)] for idx in oof_preds]
    pred_confidence = np.max(oof_probs, axis=1) * 100.0

    enriched_df = site_df.with_columns([
        pl.Series("predicted_regime", pred_regime_names),
        pl.Series("prediction_confidence_pct", pred_confidence),
        pl.Series("ml_anomaly_score", anom_norm),
        pl.Series("anomaly_score", anom_norm / 100.0),
        (0.65 * pl.col("hazard_index") + 0.35 * pl.Series(anom_norm)).alias("combined_risk_score"),
    ])

    return SARModelEvaluation(
        model=clf,
        accuracy=acc,
        macro_f1=macro_f1,
        conf_matrix=c_mat,
        feature_importance_df=fi_df,
        enriched_site_df=enriched_df,
    )