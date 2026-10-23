"""Core Radar Physics, Polarimetry (PolSAR), and Interferometry (InSAR) Engine.

Designed for NASA Space Apps 2026: 'Dancing with the SARs'.
Supports multi-frequency radar bands:
- NISAR L-Band (1.257 GHz, lambda = 23.84 cm)
- NISAR S-Band (3.200 GHz, lambda = 9.37 cm)
- Sentinel-1 C-Band (5.405 GHz, lambda = 5.55 cm)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RadarBandSpec:
    """Physical specifications of a Spaceborne SAR frequency band."""
    name: str
    frequency_ghz: float
    wavelength_m: float
    critical_baseline_m: float
    penetration_depth_desc: str

    @property
    def mission(self) -> str:
        return self.name

    @property
    def penetration_desc(self) -> str:
        return self.penetration_depth_desc


RADAR_BANDS: dict[str, RadarBandSpec] = {
    "L-Band (NISAR 24cm)": RadarBandSpec(
        name="L-Band (NISAR 24cm)",
        frequency_ghz=1.257,
        wavelength_m=0.2384,
        critical_baseline_m=6500.0,
        penetration_depth_desc="Deep canopy & dry soil penetration (~15-25 cm); high temporal coherence over vegetated terrain.",
    ),
    "S-Band (NISAR 9.4cm)": RadarBandSpec(
        name="S-Band (NISAR 9.4cm)",
        frequency_ghz=3.200,
        wavelength_m=0.0937,
        critical_baseline_m=3200.0,
        penetration_depth_desc="Moderate canopy interaction; ideal for agricultural structure and snow wetness.",
    ),
    "C-Band (Sentinel-1 5.6cm)": RadarBandSpec(
        name="C-Band (Sentinel-1 5.6cm)",
        frequency_ghz=5.405,
        wavelength_m=0.05546,
        critical_baseline_m=1800.0,
        penetration_depth_desc="High sensitivity to millimeter-scale surface motion and top-canopy leaves.",
    ),
}


def db_to_linear(db_values: np.ndarray | float) -> np.ndarray | float:
    """Convert radar backscatter Sigma0 from decibels (dB) to linear power scale."""
    return np.power(10.0, np.asarray(db_values, dtype=np.float64) / 10.0)


def linear_to_db(linear_values: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    """Convert linear radar power to decibels (dB) safely."""
    arr = np.maximum(np.asarray(linear_values, dtype=np.float64), eps)
    return 10.0 * np.log10(arr)


def compute_dual_pol_rvi(sigma0_co_db: np.ndarray, sigma0_cross_db: np.ndarray) -> np.ndarray:
    """Compute Dual-Polarization Radar Vegetation Index (RVI).

    Formula:
        RVI = (4 * sigma0_cross_linear) / (sigma0_co_linear + sigma0_cross_linear)

    Ranges from ~0 (smooth bare surface / specular water) to ~1+ (dense random volume canopy).
    """
    co_lin = db_to_linear(sigma0_co_db)
    cross_lin = db_to_linear(sigma0_cross_db)
    denom = np.maximum(co_lin + cross_lin, 1e-12)
    return (4.0 * cross_lin) / denom


def compute_polarimetric_entropy_proxy(sigma0_co_db: np.ndarray, sigma0_cross_db: np.ndarray) -> np.ndarray:
    """Compute pseudo-Shannon polarimetric depolarization entropy H_p in [0, 1]."""
    co_lin = db_to_linear(sigma0_co_db)
    cross_lin = db_to_linear(sigma0_cross_db)
    total = np.maximum(co_lin + cross_lin, 1e-12)
    p1 = np.clip(co_lin / total, 1e-9, 1.0 - 1e-9)
    p2 = np.clip(cross_lin / total, 1e-9, 1.0 - 1e-9)
    entropy = -(p1 * np.log2(p1) + p2 * np.log2(p2))
    return np.clip(entropy, 0.0, 1.0)


def displacement_mm_to_wrapped_phase(
    disp_los_mm: np.ndarray,
    wavelength_m: float = 0.2384,
) -> np.ndarray:
    """Convert Line-of-Sight (LOS) displacement in mm to wrapped interferometric phase [-pi, pi).

    Delta_phi = - (4 * pi / lambda) * Delta_r_LOS
    """
    disp_m = np.asarray(disp_los_mm, dtype=np.float64) * 1e-3
    unwrapped_phase = -(4.0 * np.pi / wavelength_m) * disp_m
    wrapped = (unwrapped_phase + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped


def unwrapped_phase_to_displacement_mm(
    unwrapped_phase_rad: np.ndarray,
    wavelength_m: float = 0.2384,
) -> np.ndarray:
    """Convert unwrapped interferometric phase (radians) back to LOS displacement (mm)."""
    disp_m = -(wavelength_m / (4.0 * np.pi)) * np.asarray(unwrapped_phase_rad, dtype=np.float64)
    return disp_m * 1e3


def unwrap_1d_phase_series(wrapped_phase: np.ndarray) -> np.ndarray:
    """Temporal 1D phase unwrapping along time axis using Itoh's continuity condition."""
    return np.unwrap(np.asarray(wrapped_phase, dtype=np.float64), axis=-1)


def fit_harmonic_motion_model(
    days: np.ndarray,
    signal: np.ndarray,
) -> dict[str, float]:
    """Fit a joint Linear + Annual Sinusoidal model to a SAR time-series:

    y(t) = c0 + v * t_yr + A * cos(2*pi*t_yr) + B * sin(2*pi*t_yr)

    Returns linear rate (per year), seasonal amplitude, phase peak day, and R2 score.
    """
    t_yr = np.asarray(days, dtype=np.float64) / 365.25
    y = np.asarray(signal, dtype=np.float64)

    design_matrix = np.column_stack([
        np.ones_like(t_yr),
        t_yr,
        np.cos(2.0 * np.pi * t_yr),
        np.sin(2.0 * np.pi * t_yr),
    ])

    coeffs, _, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)
    c0, velocity_yr, a_cos, b_sin = coeffs

    amplitude = float(np.hypot(a_cos, b_sin))
    phase_rad = float(np.arctan2(b_sin, a_cos))
    peak_day_of_year = float(((phase_rad % (2.0 * np.pi)) / (2.0 * np.pi)) * 365.25)

    y_pred = design_matrix @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))

    return {
        "intercept": float(c0),
        "linear_rate_per_yr": float(velocity_yr),
        "seasonal_amplitude": amplitude,
        "peak_day_of_year": peak_day_of_year,
        "harmonic_r2": r2,
    }