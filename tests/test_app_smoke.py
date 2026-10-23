"""Smoke tests for the Streamlit application pipeline and data export buffers."""

import io
import polars as pl
from sar_motionlens.ml_classifier import train_and_evaluate_sar_classifier
from sar_motionlens.physics import RADAR_BANDS
from sar_motionlens.synthetic import generate_2d_interferogram_grid, generate_sar_spatiotemporal_cube


def test_dashboard_pipeline_and_parquet_export() -> None:
    """Verify end-to-end data cube, ML evaluation, 2D interferogram, and Parquet serialization."""
    first_band_key = next(iter(RADAR_BANDS.keys()))
    first_band_spec = RADAR_BANDS[first_band_key]

    ts_df, site_df = generate_sar_spatiotemporal_cube(
        n_sites_per_regime=15, n_epochs=18, band_name=first_band_key, seed=101
    )
    eval_res = train_and_evaluate_sar_classifier(site_df, seed=101)
    grid = generate_2d_interferogram_grid(
        grid_size=32, wavelength_m=first_band_spec.wavelength_m
    )

    buf = io.BytesIO()
    eval_res.enriched_site_df.write_parquet(buf)
    buf.seek(0)
    reloaded = pl.read_parquet(buf)

    assert reloaded.height == 60
    assert ts_df.height == 60 * 18
    assert grid["wrapped_phase_rad"].shape == (32, 32)
    assert grid["displacement_mm"].shape == (32, 32)