"""Unit tests for SAR Physics, InSAR Phase Wrapping, Polarimetry, and Synthetic Twin."""

import numpy as np
from sar_motionlens.physics import (
    RADAR_BANDS,
    compute_dual_pol_rvi,
    displacement_mm_to_wrapped_phase,
    fit_harmonic_motion_model,
    unwrap_1d_phase_series,
    unwrapped_phase_to_displacement_mm,
)
from sar_motionlens.synthetic import generate_2d_interferogram_grid, generate_sar_spatiotemporal_cube


def test_insar_phase_displacement_roundtrip() -> None:
    """Verify that gradual LOS displacement wrapped and unwrapped recovers exact mm motion."""
    wavelength_l_band = RADAR_BANDS["L-Band (NISAR 24cm)"].wavelength_m
    days = np.linspace(0, 365, 60)
    true_disp_mm = -60.0 * (days / 365.0)  # -60 mm/yr steady subsidence

    wrapped = displacement_mm_to_wrapped_phase(true_disp_mm, wavelength_m=wavelength_l_band)
    assert np.all(wrapped >= -np.pi) and np.all(wrapped <= np.pi)

    unwrapped = unwrap_1d_phase_series(wrapped)
    recovered_disp_mm = unwrapped_phase_to_displacement_mm(unwrapped, wavelength_m=wavelength_l_band)
    np.testing.assert_allclose(recovered_disp_mm, true_disp_mm, atol=1e-6)


def test_dual_pol_rvi_physical_bounds() -> None:
    """Verify Dual-Pol Radar Vegetation Index increases with cross-pol volume scattering."""
    sigma_co = np.array([-8.0, -8.0])
    sigma_cross = np.array([-18.0, -10.0])  # Bare soil vs dense vegetation canopy
    rvi = compute_dual_pol_rvi(sigma_co, sigma_cross)
    assert rvi.shape == (2,)
    assert 0.0 < rvi[0] < rvi[1] < 2.0


def test_harmonic_motion_fitting() -> None:
    """Verify harmonic least-squares accurately extracts linear rate and seasonal amplitude."""
    days = np.arange(0, 730, 12, dtype=np.float64)
    t_yr = days / 365.25
    signal = 10.0 - 45.0 * t_yr + 12.0 * np.cos(2.0 * np.pi * t_yr - 0.8)

    res = fit_harmonic_motion_model(days, signal)
    assert np.isclose(res["linear_rate_per_yr"], -45.0, atol=1e-3)
    assert np.isclose(res["seasonal_amplitude"], 12.0, atol=1e-3)
    assert res["harmonic_r2"] > 0.999


def test_synthetic_spatiotemporal_cube_and_grid() -> None:
    """Verify Polars DataFrame generation and 2D interferogram simulation."""
    ts_df, site_df = generate_sar_spatiotemporal_cube(n_sites_per_regime=10, n_epochs=16)
    assert site_df.height == 40
    assert ts_df.height == 40 * 16
    assert "rvi_mean" in site_df.columns
    assert "los_velocity_mm_yr" in site_df.columns

    grid = generate_2d_interferogram_grid(grid_size=32)
    assert grid["wrapped_phase_rad"].shape == (32, 32)
    assert grid["coherence"].shape == (32, 32)