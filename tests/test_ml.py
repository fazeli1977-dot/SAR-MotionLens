"""Unit tests for LightGBM SAR Regime Classifier and Anomaly Detection Engine."""

from sar_motionlens.ml_classifier import SAR_FEATURE_COLUMNS, train_and_evaluate_sar_classifier
from sar_motionlens.synthetic import generate_sar_spatiotemporal_cube


def test_lightgbm_sar_regime_classification_accuracy() -> None:
    """Verify LightGBM achieves >90% 5-fold cross-validated accuracy on multi-regime SAR signatures."""
    _, site_df = generate_sar_spatiotemporal_cube(n_sites_per_regime=40, n_epochs=24, seed=42)
    eval_res = train_and_evaluate_sar_classifier(site_df, seed=42)

    assert eval_res.accuracy >= 0.90
    assert eval_res.macro_f1 >= 0.90
    assert eval_res.conf_matrix.shape == (4, 4)
    assert eval_res.feature_importance_df.height == len(SAR_FEATURE_COLUMNS)
    assert "predicted_regime" in eval_res.enriched_site_df.columns
    assert "combined_risk_score" in eval_res.enriched_site_df.columns