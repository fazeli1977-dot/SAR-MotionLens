"""Multi-Band SAR Spatio-Temporal Digital Twin Generator.

Generates physically grounded time-series and 2D spatial interferograms for four
canonical Earth surface regimes ('Dancing with the SARs'):
1. Subsidence_Aquifer: Groundwater depletion + seasonal poroelastic oscillation
2. Glacier_Creep: Fast kinematic ice flow + summer melt coherence drop
3. Floodplain_Wetland: Specular inundation drops in Sigma0 + rapid water-level phase shifts
4. Crop_Phenology: High Dual-Pol RVI & depolarization entropy cycles during canopy growth
"""

from __future__ import annotations
import numpy as np
import polars as pl
from sar_motionlens.physics import (
    RADAR_BANDS,
    compute_dual_pol_rvi,
    compute_polarimetric_entropy_proxy,
    displacement_mm_to_wrapped_phase,
    fit_harmonic_motion_model,
)

REGIME_CLASSES: list[str] = [
    "Subsidence_Aquifer",
    "Glacier_Creep",
    "Floodplain_Wetland",
    "Crop_Phenology",
]


def generate_sar_spatiotemporal_cube(
    n_sites_per_regime: int = 90,
    n_epochs: int = 32,
    cadence_days: int = 12,
    band_name: str = "L-Band (NISAR 24cm)",
    seed: int = 42,
    base_lat: float = 35.68,
    base_lon: float = 51.42,
    deformation_scale: float = 1.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate both a full multi-epoch time-series Polars DataFrame and a site-level feature summary DataFrame."""
    rng = np.random.default_rng(seed)
    band = RADAR_BANDS.get(band_name, RADAR_BANDS["L-Band (NISAR 24cm)"])
    wavelength_m = band.wavelength_m

    # Base coordinates centered around a diverse tectonic/agricultural basin (e.g., Central Valley / Zagros / Alpine)
    base_lat, base_lon = float(base_lat), float(base_lon)
    days = np.arange(n_epochs, dtype=np.float64) * cadence_days
    t_yr = days / 365.25

    ts_records: list[dict] = []
    site_records: list[dict] = []

    site_id_counter = 1000
    is_l_band = wavelength_m > 0.15
    coherence_band_boost = 0.14 if is_l_band else -0.05

    for regime_idx, regime in enumerate(REGIME_CLASSES):
        # Distinct spatial clusters around the region for realistic 3D map visualization
        cluster_lat_offset = (regime_idx - 1.5) * 0.08
        cluster_lon_offset = ((regime_idx % 2) - 0.5) * 0.12

        for _ in range(n_sites_per_regime):
            site_id = f"SAR-{site_id_counter}"
            site_id_counter += 1

            lat = float(base_lat + cluster_lat_offset + rng.normal(0, 0.025))
            lon = float(base_lon + cluster_lon_offset + rng.normal(0, 0.025))
            inc_angle_deg = float(rng.uniform(32.0, 44.0))

            if regime == "Subsidence_Aquifer":
                # Strong negative linear LOS rate (-35 to -85 mm/yr) + seasonal aquifer recharge cycle
                vel_mm_yr = rng.uniform(-85.0, -35.0)
                seasonal_amp_mm = rng.uniform(6.0, 16.0)
                co_base_db = rng.uniform(-9.5, -5.5)
                cross_base_db = co_base_db - rng.uniform(7.5, 10.5)
                co_season_db = rng.uniform(0.5, 1.5)
                cross_season_db = rng.uniform(0.4, 1.2)
                base_coh = rng.uniform(0.68, 0.86) + coherence_band_boost

            elif regime == "Glacier_Creep":
                # Very strong displacement (-140 to -75 mm/yr) + strong summer acceleration & melt decorrelation
                vel_mm_yr = rng.uniform(-145.0, -75.0)
                seasonal_amp_mm = rng.uniform(18.0, 38.0)
                co_base_db = rng.uniform(-13.5, -8.5)
                cross_base_db = co_base_db - rng.uniform(5.5, 8.0)
                co_season_db = rng.uniform(2.0, 4.0)
                cross_season_db = rng.uniform(1.5, 3.0)
                base_coh = rng.uniform(0.38, 0.62) + coherence_band_boost

            elif regime == "Floodplain_Wetland":
                # Moderate displacement + massive backscatter drops during seasonal inundation
                vel_mm_yr = rng.uniform(-14.0, 8.0)
                seasonal_amp_mm = rng.uniform(10.0, 26.0)
                co_base_db = rng.uniform(-18.5, -12.5)
                cross_base_db = co_base_db - rng.uniform(6.5, 9.5)
                co_season_db = rng.uniform(4.5, 7.5)
                cross_season_db = rng.uniform(3.5, 6.0)
                base_coh = rng.uniform(0.30, 0.54) + coherence_band_boost

            else:  # Crop_Phenology
                # Stable ground (near-zero LOS displacement) + strong cross-pol HV volume scattering surge
                vel_mm_yr = rng.uniform(-6.0, 6.0)
                seasonal_amp_mm = rng.uniform(1.5, 5.5)
                co_base_db = rng.uniform(-11.5, -7.5)
                cross_base_db = co_base_db - rng.uniform(3.2, 5.5)
                co_season_db = rng.uniform(2.2, 4.2)
                cross_season_db = rng.uniform(4.0, 7.0)
                base_coh = rng.uniform(0.48, 0.72) + coherence_band_boost

            vel_mm_yr *= deformation_scale
            seasonal_amp_mm *= deformation_scale
            phase_shift = rng.uniform(0, 2.0 * np.pi)
            disp_los_mm = (
                vel_mm_yr * t_yr
                + seasonal_amp_mm * np.sin(2.0 * np.pi * t_yr + phase_shift)
                + rng.normal(0.0, 1.8, size=n_epochs)
            )

            sigma0_co_db = (
                co_base_db
                + co_season_db * np.sin(2.0 * np.pi * t_yr + phase_shift)
                + rng.normal(0.0, 0.65, size=n_epochs)
            )
            sigma0_cross_db = (
                cross_base_db
                + cross_season_db * np.sin(2.0 * np.pi * t_yr + phase_shift - 0.3)
                + rng.normal(0.0, 0.65, size=n_epochs)
            )

            rvi_series = compute_dual_pol_rvi(sigma0_co_db, sigma0_cross_db)
            entropy_series = compute_polarimetric_entropy_proxy(sigma0_co_db, sigma0_cross_db)
            wrapped_phase = displacement_mm_to_wrapped_phase(disp_los_mm, wavelength_m=wavelength_m)

            coherence_series = np.clip(
                base_coh - 0.12 * np.abs(np.sin(np.pi * t_yr)) + rng.normal(0.0, 0.04, size=n_epochs),
                0.10,
                0.99,
            )

            harm_disp = fit_harmonic_motion_model(days, disp_los_mm)
            harm_rvi = fit_harmonic_motion_model(days, rvi_series)

            # Risk / Anomaly Severity Score [0, 100] combining deformation speed and decorrelation hazard
            hazard_score = float(
                np.clip(
                    0.55 * min(abs(harm_disp["linear_rate_per_yr"]) / 1.2, 100.0)
                    + 0.25 * min(harm_disp["seasonal_amplitude"] * 2.2, 100.0)
                    + 0.20 * (1.0 - float(np.mean(coherence_series))) * 100.0,
                    0.0,
                    100.0,
                )
            )

            site_records.append({
                "site_id": site_id,
                "regime": regime,
                "regime_label": regime_idx,
                "latitude": lat,
                "longitude": lon,
                "incidence_angle_deg": inc_angle_deg,
                "los_velocity_mm_yr": harm_disp["linear_rate_per_yr"],
                "los_seasonal_amp_mm": harm_disp["seasonal_amplitude"],
                "los_harmonic_r2": harm_disp["harmonic_r2"],
                "sigma0_co_mean_db": float(np.mean(sigma0_co_db)),
                "sigma0_co_std_db": float(np.std(sigma0_co_db)),
                "sigma0_cross_mean_db": float(np.mean(sigma0_cross_db)),
                "sigma0_cross_std_db": float(np.std(sigma0_cross_db)),
                "pol_ratio_db_mean": float(np.mean(sigma0_cross_db - sigma0_co_db)),
                "rvi_mean": float(np.mean(rvi_series)),
                "rvi_seasonal_amp": harm_rvi["seasonal_amplitude"],
                "pol_entropy_mean": float(np.mean(entropy_series)),
                "coherence_mean": float(np.mean(coherence_series)),
                "coherence_min": float(np.min(coherence_series)),
                "hazard_index": hazard_score,
                "lat": lat,
                "lon": lon,
                "seasonal_amp_mm": harm_disp["seasonal_amplitude"],
                "mean_coherence": float(np.mean(coherence_series)),
                "mean_rvi": float(np.mean(rvi_series)),
                "pol_entropy": float(np.mean(entropy_series)),
                "backscatter_vv_db": float(np.mean(sigma0_co_db)),
                "backscatter_vh_db": float(np.mean(sigma0_cross_db)),
                "vv_db": float(np.mean(sigma0_co_db)),
                "vh_db": float(np.mean(sigma0_cross_db)),
                "anomaly_score": hazard_score / 100.0,
            })

            for ep_idx in range(n_epochs):
                ts_records.append({
                    "site_id": site_id,
                    "regime": regime,
                    "epoch_idx": int(ep_idx),
                    "day": float(days[ep_idx]),
                    "disp_los_mm": float(disp_los_mm[ep_idx]),
                    "wrapped_phase_rad": float(wrapped_phase[ep_idx]),
                    "sigma0_co_db": float(sigma0_co_db[ep_idx]),
                    "sigma0_cross_db": float(sigma0_cross_db[ep_idx]),
                    "rvi": float(rvi_series[ep_idx]),
                    "pol_entropy": float(entropy_series[ep_idx]),
                    "coherence": float(coherence_series[ep_idx]),
                    "backscatter_vv_db": float(sigma0_co_db[ep_idx]),
                    "backscatter_vh_db": float(sigma0_cross_db[ep_idx]),
                    "vv_db": float(sigma0_co_db[ep_idx]),
                    "vh_db": float(sigma0_cross_db[ep_idx]),
                    "los_disp_mm": float(disp_los_mm[ep_idx]),
                    "displacement_mm": float(disp_los_mm[ep_idx]),
                    "wrapped_phase": float(wrapped_phase[ep_idx]),
                })

    ts_df = pl.DataFrame(ts_records).with_columns([
        pl.col('sigma0_co_db').alias('backscatter_vv_db'),
        pl.col('sigma0_cross_db').alias('backscatter_vh_db'),
        pl.col('sigma0_co_db').alias('vv_db'),
        pl.col('sigma0_cross_db').alias('vh_db'),
        pl.col('disp_los_mm').alias('los_disp_mm'),
        pl.col('disp_los_mm').alias('displacement_mm'),
        pl.col('disp_los_mm').alias('los_displacement_mm'),
        pl.col('day').alias('day_of_year'),
        pl.col('wrapped_phase_rad').alias('wrapped_phase'),
    ])
    site_df = pl.DataFrame(site_records).with_columns([
        pl.col('latitude').alias('lat'),
        pl.col('longitude').alias('lon'),
        pl.col('los_seasonal_amp_mm').alias('seasonal_amp_mm'),
        pl.col('coherence_mean').alias('mean_coherence'),
        pl.col('rvi_mean').alias('mean_rvi'),
        pl.col('pol_entropy_mean').alias('pol_entropy'),
        pl.col('sigma0_co_mean_db').alias('backscatter_vv_db'),
        pl.col('sigma0_cross_mean_db').alias('backscatter_vh_db'),
        pl.col('sigma0_co_mean_db').alias('vv_db'),
        pl.col('sigma0_cross_mean_db').alias('vh_db'),
        (pl.col('hazard_index') / 100.0).alias('anomaly_score'),
    ])
    return ts_df, site_df


def generate_2d_interferogram_grid(
    grid_size: int = 120,
    wavelength_m: float = 0.2384,
    peak_subsidence_mm: float = -75.0,
    time_years: float = 1.0,
    noise_level: float = 0.15,
    seed: int = 42,
    bowl_angle_deg: float = 0.0,
    fault_creep: bool = False,
) -> dict[str, np.ndarray]:
    """Simulate a 2D spatial SAR deformation bowl, wrapped interferometric fringes, and coherence map."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-3.0, 3.0, grid_size)
    y = np.linspace(-3.0, 3.0, grid_size)
    xx, yy = np.meshgrid(x, y)
    theta = np.radians(bowl_angle_deg)
    xx, yy = xx * np.cos(theta) + yy * np.sin(theta), -xx * np.sin(theta) + yy * np.cos(theta)

    # Double-lobed deformation bowl (e.g., subsiding aquifer + secondary fault creep)
    bowl_1 = np.exp(-((xx + 0.55) ** 2 + (yy - 0.35) ** 2) / 1.35)
    bowl_2 = 0.55 * np.exp(-((xx - 0.85) ** 2 + (yy + 0.65) ** 2) / 0.85)
    disp_mm = peak_subsidence_mm * time_years * (bowl_1 + bowl_2)
    if fault_creep:
        # Right-lateral creep: smooth step in LOS across the fault trace (x' = 0).
        disp_mm = disp_mm + 0.5 * peak_subsidence_mm * time_years * np.arctan(xx / 0.15) / (np.pi / 2)

    wrapped_clean = displacement_mm_to_wrapped_phase(disp_mm, wavelength_m=wavelength_m)
    phase_noise = rng.normal(0.0, noise_level * np.pi, size=(grid_size, grid_size))
    wrapped_noisy = (wrapped_clean + phase_noise + np.pi) % (2.0 * np.pi) - np.pi

    # Coherence drops in high phase-gradient zones and noisy patches
    grad_y, grad_x = np.gradient(disp_mm)
    strain_mag = np.hypot(grad_x, grad_y)
    coherence_2d = np.clip(0.92 - 0.035 * strain_mag - 0.25 * noise_level + rng.normal(0, 0.03, disp_mm.shape), 0.1, 0.99)

    return {
        "x_km": x * 10.0,
        "y_km": y * 10.0,
        "displacement_mm": disp_mm,
        "wrapped_phase_rad": wrapped_noisy,
        "coherence": coherence_2d,
    }
