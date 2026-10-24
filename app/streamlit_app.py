"""SAR-MotionLens: Multi-Band InSAR & PolSAR Spatiotemporal Earth Dynamics Explorer.

Interactive Streamlit Mission Control Dashboard built for the NASA Space Apps Challenge
("Through the Radar Looking Glass: Revealing Earth Processes with SAR").
"""

from __future__ import annotations
import io
import sys
from pathlib import Path

# Streamlit only watches the main script, so edited modules under src/ can stay stale in memory
# after a redeploy; always re-import them from the repo.
_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))
for _name in [n for n in sys.modules if n == "sar_motionlens" or n.startswith("sar_motionlens.")]:
    del sys.modules[_name]

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import pydeck as pdk
import streamlit as st

from sar_motionlens.geo import (
    PRESETS,
    SCENARIOS,
    LocationProfile,
    analyze_location,
    build_geojson,
    build_pdf_report,
    explain_risk,
    fringes_for_displacement,
)
from sar_motionlens.ml_classifier import train_and_evaluate_sar_classifier
from sar_motionlens.nisar import NisarGranule, search_nisar
from sar_motionlens.physics import RADAR_BANDS
from sar_motionlens.synthetic import (
    REGIME_CLASSES,
    generate_2d_interferogram_grid,
    generate_sar_spatiotemporal_cube,
)

try:
    import folium
    from streamlit_folium import st_folium
except ImportError:  # optional dependency
    folium = None
    st_folium = None

st.set_page_config(
    page_title="SAR-MotionLens | NASA SAR Earth Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NASA_CSS = """
<style>
.stApp { background: radial-gradient(circle at 20% 0%, #0B1B3A 0%, #050B1A 55%, #02050D 100%); }
h1 { letter-spacing: .5px; border-bottom: 3px solid #FC3D21; padding-bottom: .4rem; }
h2, h3 { color: #FFFFFF; text-transform: uppercase; letter-spacing: 1px; font-size: 1.15rem !important; }
[data-testid="stSidebar"] { background: #081330; border-right: 1px solid #0B3D91; }
[data-testid="stMetric"] {
  background: linear-gradient(145deg, #0B1B3A, #0A2552);
  border: 1px solid #0B3D91; border-left: 4px solid #FC3D21;
  border-radius: 6px; padding: .7rem .9rem;
}
[data-testid="stMetricLabel"] { text-transform: uppercase; letter-spacing: .8px; color: #9DB4D8; }
button[kind="secondary"], .stDownloadButton button {
  border: 1px solid #0B3D91; background: #0B3D91; color: #fff; border-radius: 4px;
}
button[kind="secondary"]:hover, .stDownloadButton button:hover { border-color: #FC3D21; color: #fff; }
.dev-footer {
  margin-top: 1rem; padding: 1.2rem 1.5rem; border-top: 3px solid #FC3D21;
  background: #081330; font-size: 1.25rem; line-height: 1.9; color: #E8EEF8;
}
.dev-footer a { color: #5FA8FF; text-decoration: none; font-weight: 600; }
</style>
"""
st.markdown(NASA_CSS, unsafe_allow_html=True)

REGIME_RGB_MAP: dict[str, list[int]] = {
    "Subsidence_Aquifer": [239, 68, 68, 210],
    "Glacier_Creep": [56, 189, 248, 210],
    "Floodplain_Wetland": [59, 130, 246, 210],
    "Crop_Phenology": [16, 185, 129, 210],
}


@st.cache_data(show_spinner=False)
def load_mission_cube_and_ml(
    n_sites_per_regime: int,
    n_epochs: int,
    band_name: str,
    seed: int,
    base_lat: float = 35.68,
    base_lon: float = 51.42,
    deformation_scale: float = 1.0,
):
    """Cached generator and LightGBM trainer for rapid interactive exploration."""
    ts_df, site_df = generate_sar_spatiotemporal_cube(
        n_sites_per_regime=n_sites_per_regime,
        n_epochs=n_epochs,
        band_name=band_name,
        seed=seed,
        base_lat=base_lat,
        base_lon=base_lon,
        deformation_scale=deformation_scale,
    )
    eval_res = train_and_evaluate_sar_classifier(site_df, seed=seed)
    return ts_df, eval_res


@st.cache_data(show_spinner=False, ttl=3600)
def cached_nisar_search(lat: float, lon: float) -> list[NisarGranule]:
    return search_nisar(lat, lon)


@st.cache_data(show_spinner=False)
def cached_location_profile(lat: float, lon: float) -> LocationProfile:
    return analyze_location(lat, lon)


@st.cache_data(show_spinner=False)
def cached_interferogram(
    peak_mm: float, wavelength_m: float, seed: int, angle_deg: float, fault_creep: bool, time_years: float = 1.0
) -> dict[str, np.ndarray]:
    return generate_2d_interferogram_grid(
        grid_size=110,
        peak_subsidence_mm=peak_mm,
        time_years=time_years,
        wavelength_m=wavelength_m,
        seed=seed,
        bowl_angle_deg=angle_deg,
        fault_creep=fault_creep,
    )


def _set_point(lat: float, lon: float) -> None:
    st.session_state.selected_point = {
        "lat": float(np.clip(lat, -85.0, 85.0)),
        "lon": float(((lon + 180.0) % 360.0) - 180.0),
    }


def _apply_preset() -> None:
    _set_point(*PRESETS[st.session_state.preset_choice])


def _apply_scenario() -> None:
    sc = next(s for s in SCENARIOS if s["title"] == st.session_state.scenario_choice)
    st.session_state.preset_choice = sc["preset"]
    st.session_state.band_choice = sc["band"]
    st.session_state.active_story = sc["story"]
    _set_point(*PRESETS[sc["preset"]])


def render_dashboard() -> None:
    """Main entry point for the SAR-MotionLens Streamlit application."""
    st.title("🛰️ SAR-MotionLens: Multi-Band InSAR & PolSAR Earth Dynamics Intelligence")
    st.markdown(
        "**Through the Radar Looking Glass: Revealing Earth Processes with SAR** — "
        "Fusing **Sentinel-1 (C-Band)** and **NISAR (L/S-Band)** Interferometric Phase ($\\Delta\\phi$) "
        "and Dual-Polarization Backscatter ($\\sigma^0_{VV}, \\sigma^0_{VH}$, RVI, Entropy $H$) with "
        "**LightGBM** & **IsolationForest** to classify and monitor critical Earth hazards day and night."
    )

    if "selected_point" not in st.session_state:
        st.session_state.selected_point = {"lat": 35.6892, "lon": 51.3890}
    point = st.session_state.selected_point
    base_lat, base_lon = point["lat"], point["lon"]
    profile = cached_location_profile(round(base_lat, 4), round(base_lon, 4))

    with st.sidebar:
        st.subheader("Guided Scenarios")
        st.selectbox("Story Mode", [s["title"] for s in SCENARIOS], key="scenario_choice")
        st.button("Load Scenario", on_click=_apply_scenario, width="stretch")
        st.divider()
        st.subheader("Global Study Point")
        st.selectbox("Strategic Preset", list(PRESETS.keys()), key="preset_choice")
        st.button("Apply Preset", on_click=_apply_preset, width="stretch")
        # Key embeds the point so inputs refresh after map clicks or presets.
        key_sfx = f"{base_lat:.4f}_{base_lon:.4f}"
        c_lat, c_lon = st.columns(2)
        in_lat = c_lat.number_input("Latitude", -85.0, 85.0, base_lat, 0.01, format="%.4f", key=f"lat_{key_sfx}")
        in_lon = c_lon.number_input("Longitude", -180.0, 180.0, base_lon, 0.01, format="%.4f", key=f"lon_{key_sfx}")
        if st.button("Apply Coordinates", width="stretch"):
            _set_point(in_lat, in_lon)
            st.rerun()
        st.caption(f"Selected: Lat {base_lat:.4f}, Lon {base_lon:.4f}. Or click the map in the first tab.")
        st.divider()
        st.header("🎛️ SAR Sensor & Mission Controls")
        band_name = st.selectbox(
            "Select SAR Sensor Band",
            options=list(RADAR_BANDS.keys()),
            index=0,
            key="band_choice",
            format_func=lambda k: f"{k} — {RADAR_BANDS[k].mission} (λ={RADAR_BANDS[k].wavelength_m*100:.2f} cm)",
        )
        band_spec = RADAR_BANDS[band_name]
        st.caption(
            f"**Frequency:** {band_spec.frequency_ghz:.2f} GHz | "
            f"**Fringe Cycle (λ/2):** {band_spec.wavelength_m*500:.1f} mm/cycle"
        )

        n_sites_per_regime = st.slider(
            "Sites per Regime Class", min_value=20, max_value=80, value=40, step=10
        )
        n_epochs = st.slider(
            "Temporal Revisits (Epochs / Year)", min_value=16, max_value=48, value=30, step=2
        )
        seed = st.number_input("Simulation Random Seed", min_value=1, max_value=9999, value=42)

        st.divider()
        st.subheader("🌐 2D Interferogram Simulator")
        peak_deform_mm = st.slider(
            "Peak Deformation Bowl (mm)",
            min_value=-180.0,
            max_value=-10.0,
            value=float(np.clip(round(profile.peak_deform_mm / 5.0) * 5.0, -180.0, -10.0)),
            step=5.0,
            key=f"peak_{key_sfx}",
        )
        obs_years = st.slider("Observation Span (years)", 0.25, 2.0, 1.0, 0.25)

    ts_df, eval_res = load_mission_cube_and_ml(
        n_sites_per_regime=n_sites_per_regime,
        n_epochs=n_epochs,
        band_name=band_name,
        seed=(int(seed) + profile.seed_offset) % 1_000_000,
        base_lat=float(base_lat),
        base_lon=float(base_lon),
        deformation_scale=profile.deformation_scale,
    )
    enriched_df = eval_res.enriched_site_df
    mean_ml_risk = float(enriched_df["combined_risk_score"].mean())

    # Top KPI Telemetry Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Monitored SAR Targets", f"{enriched_df.height}", f"{len(REGIME_CLASSES)} Regimes")
    col2.metric("Spatiotemporal Observations", f"{ts_df.height:,}", f"{n_epochs} Revisits/Site")
    col3.metric("LightGBM 5-Fold CV Accuracy", f"{eval_res.accuracy * 100:.2f}%", "Stratified OOF")
    col4.metric("Macro F1-Score", f"{eval_res.macro_f1:.3f}", "Multi-Regime")
    high_risk_count = enriched_df.filter(pl.col("combined_risk_score") >= 65.0).height
    col5.metric("High-Hazard Anomaly Sites", f"{high_risk_count}", "Risk ≥ 65/100")

    tab_map, tab_ts, tab_ml = st.tabs([
        "🌍 3D Geospatial Deck & 2D InSAR Fringe Simulator",
        "📈 Site Time-Series Kinematics & PolSAR Decomposition",
        "🧠 LightGBM Regime Classifier & Data Export",
    ])

    with tab_map:
        if st.session_state.get("active_story"):
            st.info(f"**Story:** {st.session_state.active_story}")
        try:
            with st.spinner("Querying NASA/ASF DAAC for real NISAR granules..."):
                granules = cached_nisar_search(round(base_lat, 3), round(base_lon, 3))
            nisar_error = None
        except Exception as exc:  # network/API failure must not break the dashboard
            granules, nisar_error = [], str(exc)

        st.subheader("Point Selection: click anywhere on the globe")
        if st_folium is None:
            st.warning("Install `streamlit-folium` to enable click-to-select; use the sidebar coordinates meanwhile.")
        else:
            fmap = folium.Map(location=[base_lat, base_lon], zoom_start=5, tiles="OpenStreetMap")
            for g in granules[:10]:
                if g.footprint:
                    folium.Polygon(
                        [(y, x) for x, y in g.footprint],
                        color="#FC3D21", weight=1, fill=True, fill_opacity=0.05,
                        tooltip=f"NISAR {g.level} | {g.start_time[:10]}",
                    ).add_to(fmap)
            folium.Marker([base_lat, base_lon], tooltip=f"{base_lat:.4f}, {base_lon:.4f}").add_to(fmap)
            map_state = st_folium(fmap, height=320, use_container_width=True, returned_objects=["last_clicked"])
            clicked = (map_state or {}).get("last_clicked")
            if clicked and st.session_state.get("last_click") != (clicked["lat"], clicked["lng"]):
                st.session_state.last_click = (clicked["lat"], clicked["lng"])
                _set_point(clicked["lat"], clicked["lng"])
                st.rerun()

        st.subheader("Real NISAR Data at This Point (NASA ASF DAAC)")
        if nisar_error:
            st.warning(f"NISAR catalog unavailable right now: {nisar_error}")
        elif not granules:
            st.info("No NISAR granules found at this point yet (mission coverage is still expanding).")
        else:
            levels = sorted({g.level for g in granules})
            n1, n2, n3 = st.columns(3)
            n1.metric("Granules Found", len(granules), ", ".join(levels))
            n2.metric("Latest Acquisition", max(g.start_time for g in granules)[:10])
            n3.metric("Orbit", granules[0].flight_direction.title(), f"Path {granules[0].path_number}")
            st.dataframe(
                [
                    {"Product": g.level, "Start (UTC)": g.start_time, "Direction": g.flight_direction,
                     "Path": g.path_number, "Download (Earthdata login)": g.url}
                    for g in granules
                ],
                width="stretch",
                column_config={"Download (Earthdata login)": st.column_config.LinkColumn()},
            )
        st.caption(
            "Granule metadata above is **real NISAR data**. The deformation, interferogram, time-series "
            "and ML results below are a **physically motivated simulation** for the selected point; "
            "NISAR HDF5 products (e.g. GUNW) can be downloaded with a free NASA Earthdata account."
        )

        st.subheader("Simulated Deformation Analysis (Demonstration)")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Nearest Feature", profile.nearest_feature, f"{profile.nearest_km:.0f} km")
        p2.metric("Displacement Rate", f"{-profile.rate_mm_yr:.1f} mm/yr", profile.nearest_kind)
        p3.metric("Terrain (synthetic DEM)", f"{profile.elevation_m:.0f} m", f"slope {profile.slope_deg:.1f} deg")
        p4.metric("Subsidence Risk", profile.risk_class, f"ML mean risk {mean_ml_risk:.1f}/100")
        st.caption(explain_risk(profile))

        with st.expander("Compare with a second location"):
            other_name = st.selectbox("Second location", list(PRESETS.keys()), key="compare_choice")
            other = cached_location_profile(*[round(v, 4) for v in PRESETS[other_name]])
            st.dataframe(
                [
                    {"Metric": "Location", "Selected point": f"{base_lat:.3f}, {base_lon:.3f}", other_name: "preset"},
                    {"Metric": "Displacement rate (mm/yr)", "Selected point": f"{-profile.rate_mm_yr:.1f}", other_name: f"{-other.rate_mm_yr:.1f}"},
                    {"Metric": "Risk class", "Selected point": profile.risk_class, other_name: other.risk_class},
                    {"Metric": "Nearest feature", "Selected point": profile.nearest_feature, other_name: other.nearest_feature},
                    {"Metric": "Terrain (m / slope deg)", "Selected point": f"{profile.elevation_m:.0f} / {profile.slope_deg:.1f}", other_name: f"{other.elevation_m:.0f} / {other.slope_deg:.1f}"},
                ],
                width="stretch",
                hide_index=True,
            )

        left_col, right_col = st.columns([1.15, 1.0])

        with left_col:
            st.subheader("3D SAR Deformation & Hazard Telemetry Map")
            selected_regimes = st.multiselect(
                "Filter Earth Process Regimes",
                options=REGIME_CLASSES,
                default=REGIME_CLASSES,
            )
            filtered_sites = enriched_df.filter(pl.col("regime").is_in(selected_regimes))
            map_records = filtered_sites.to_dicts()
            for row in map_records:
                row["color"] = REGIME_RGB_MAP.get(row["regime"], [200, 200, 200, 200])
                row["elevation_m"] = float(row["combined_risk_score"]) * 450.0

            view_state = pdk.ViewState(
                latitude=float(filtered_sites["lat"].mean() or 36.2),
                longitude=float(filtered_sites["lon"].mean() or -119.5),
                zoom=5.2,
                pitch=45,
                bearing=-12,
            )

            col_layer = pdk.Layer(
                "ColumnLayer",
                data=map_records,
                get_position="[lon, lat]",
                get_elevation="elevation_m",
                elevation_scale=1,
                radius=2800,
                get_fill_color="color",
                pickable=True,
                auto_highlight=True,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[col_layer],
                    initial_view_state=view_state,
                    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                    tooltip={
                        "html": (
                            "<b>Site:</b> {site_id}<br/>"
                            "<b>True Regime:</b> {regime}<br/>"
                            "<b>Predicted:</b> {predicted_regime} ({prediction_confidence_pct:.1f}%)<br/>"
                            "<b>LOS Velocity:</b> {los_velocity_mm_yr:.1f} mm/yr<br/>"
                            "<b>Mean RVI:</b> {rvi_mean:.2f} | <b>Coherence:</b> {coherence_mean:.2f}<br/>"
                            "<b>Combined Risk Score:</b> {combined_risk_score:.1f}/100"
                        )
                    },
                )
            )
            st.caption(
                "Column height represents the **Combined Risk Score** (fusion of physical subsidence/inundation hazard "
                "and IsolationForest anomaly score). Color denotes physical regime."
            )

        with right_col:
            st.subheader(f"2D InSAR Wrapped Phase Fringe Simulator ({band_name})")
            grid_data = cached_interferogram(
                float(peak_deform_mm),
                band_spec.wavelength_m,
                (int(seed) + profile.seed_offset) % 1_000_000,
                profile.bowl_angle_deg,
                profile.fault_creep,
                float(obs_years),
            )

            fig_insar = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    "Wrapped Interferogram Δφ ∈ [-π, +π] (rad)",
                    "Unwrapped LOS Deformation (mm)",
                ),
                horizontal_spacing=0.12,
            )
            fig_insar.add_trace(
                go.Heatmap(
                    z=grid_data["wrapped_phase_rad"],
                    colorscale="Twilight",
                    zmin=-np.pi,
                    zmax=np.pi,
                    colorbar=dict(title="rad", x=0.45, len=0.85),
                ),
                row=1,
                col=1,
            )
            fig_insar.add_trace(
                go.Heatmap(
                    z=grid_data["displacement_mm"],
                    colorscale="RdBu_r",
                    colorbar=dict(title="mm", x=1.02, len=0.85),
                ),
                row=1,
                col=2,
            )
            fig_insar.update_layout(
                height=430,
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark",
            )
            st.plotly_chart(fig_insar, width="stretch")
            st.info(
                f"**Physical Insight ({band_name}):** Each full color cycle ($2\\pi$ radians) corresponds to "
                f"**{band_spec.wavelength_m * 500:.2f} mm** of Line-of-Sight motion ($\\lambda/2$). "
                "Switch between **C-Band** and **L-Band** in the sidebar to see how shorter wavelengths produce "
                "denser interferometric fringes for the exact same ground deformation!"
            )
            shown_mm = float(peak_deform_mm) * float(obs_years)
            st.caption(f"Peak LOS motion over {obs_years:g} yr: {shown_mm:.0f} mm. Fringes this produces per band:")
            st.dataframe(
                [
                    {"Band": name, "Wavelength (cm)": round(spec.wavelength_m * 100, 2),
                     "Fringes": round(fringes_for_displacement(shown_mm, spec.wavelength_m), 1)}
                    for name, spec in RADAR_BANDS.items()
                ],
                width="stretch",
                hide_index=True,
            )

    with tab_ts:
        st.subheader("Multi-Parameter SAR Time-Series Inspector (InSAR Kinematics + PolSAR Scattering)")
        site_options = enriched_df["site_id"].to_list()
        sel_col1, sel_col2 = st.columns([1, 2])
        with sel_col1:
            chosen_site = st.selectbox("Select Monitoring Site ID", options=site_options, index=0)
        site_meta = enriched_df.filter(pl.col("site_id") == chosen_site).row(0, named=True)
        with sel_col2:
            st.markdown(
                f"**True Regime:** `{site_meta['regime']}` | "
                f"**LightGBM Prediction:** `{site_meta['predicted_regime']}` "
                f"({site_meta['prediction_confidence_pct']:.1f}% confidence) | "
                f"**Harmonic Fit $R^2$:** `{site_meta['los_harmonic_r2']:.3f}` | "
                f"**Risk Score:** `{site_meta['combined_risk_score']:.1f}/100`"
            )

        site_ts = ts_df.filter(pl.col("site_id") == chosen_site).sort("epoch_idx")
        days = site_ts["day_of_year"].to_numpy()
        t_yr = days / 365.25

        vel = float(site_meta["los_velocity_mm_yr"])
        omega = 2.0 * np.pi
        design = np.column_stack([np.ones_like(t_yr), t_yr, np.sin(omega * t_yr), np.cos(omega * t_yr)])
        coeffs, _, _, _ = np.linalg.lstsq(design, site_ts["los_displacement_mm"].to_numpy(), rcond=None)
        fitted_los = design @ coeffs
        resid_sigma = float(np.std(site_ts["los_displacement_mm"].to_numpy() - fitted_los))

        fig_ts = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"Unwrapped InSAR LOS Displacement (v = {vel:.1f} mm/yr)",
                "Dual-Pol Backscatter Intensity σ⁰ Co-Pol vs Cross-Pol (dB)",
                "Radar Vegetation Index (RVI) & Polarimetric Entropy (H)",
                "Interferometric Coherence γ ∈ [0, 1]",
            ),
            vertical_spacing=0.14,
            horizontal_spacing=0.10,
        )

        fig_ts.add_trace(
            go.Scatter(
                x=days,
                y=site_ts["los_displacement_mm"].to_list(),
                mode="markers+lines",
                name="Observed LOS (mm)",
                line=dict(color="#38BDF8", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=np.concatenate([days, days[::-1]]),
                y=np.concatenate([fitted_los + resid_sigma, (fitted_los - resid_sigma)[::-1]]).tolist(),
                fill="toself",
                fillcolor="rgba(244,63,94,0.15)",
                line=dict(width=0),
                name="Fit ±1σ uncertainty",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=days,
                y=fitted_los.tolist(),
                mode="lines",
                name="Harmonic Model Fit",
                line=dict(color="#F43F5E", width=2.5, dash="dash"),
            ),
            row=1,
            col=1,
        )

        fig_ts.add_trace(
            go.Scatter(
                x=days,
                y=site_ts["sigma0_co_db"].to_list(),
                mode="lines+markers",
                name="σ⁰ Co-Pol (VV/HH dB)",
                line=dict(color="#FBBF24"),
            ),
            row=1,
            col=2,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=days,
                y=site_ts["sigma0_cross_db"].to_list(),
                mode="lines+markers",
                name="σ⁰ Cross-Pol (VH/HV dB)",
                line=dict(color="#A855F7"),
            ),
            row=1,
            col=2,
        )

        fig_ts.add_trace(
            go.Scatter(
                x=days,
                y=site_ts["rvi"].to_list(),
                mode="lines+markers",
                name="Dual-Pol RVI",
                line=dict(color="#10B981", width=2),
            ),
            row=2,
            col=1,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=days,
                y=site_ts["pol_entropy"].to_list(),
                mode="lines",
                name="Pol Entropy H",
                line=dict(color="#6EE7B7", dash="dot"),
            ),
            row=2,
            col=1,
        )

        fig_ts.add_trace(
            go.Scatter(
                x=days,
                y=site_ts["coherence"].to_list(),
                mode="lines+markers",
                name="Coherence γ",
                line=dict(color="#F97316", width=2),
            ),
            row=2,
            col=2,
        )

        fig_ts.update_xaxes(title_text="Day of Year (DOY)")
        fig_ts.update_layout(height=620, template="plotly_dark", margin=dict(l=25, r=25, t=45, b=25))
        st.plotly_chart(fig_ts, width="stretch")

    with tab_ml:
        st.subheader("LightGBM Physical Feature Importance & Multi-Regime Confusion Matrix")
        ml_col1, ml_col2 = st.columns([1.1, 1.0])

        with ml_col1:
            fi_pd = eval_res.feature_importance_df.to_pandas()
            fig_fi = px.bar(
                fi_pd,
                x="importance_gain_pct",
                y="feature",
                orientation="h",
                title="LightGBM Split Feature Importance (%)",
                labels={"importance_gain_pct": "Relative Importance (%)", "feature": "SAR Physical Descriptor"},
                color="importance_gain_pct",
                color_continuous_scale="Viridis",
            )
            fig_fi.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=440,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=45, b=20),
            )
            st.plotly_chart(fig_fi, width="stretch")

        with ml_col2:
            fig_cm = px.imshow(
                eval_res.conf_matrix,
                x=REGIME_CLASSES,
                y=REGIME_CLASSES,
                text_auto=True,
                color_continuous_scale="Blues",
                title="5-Fold Stratified Cross-Validation Confusion Matrix",
                labels=dict(x="Predicted SAR Regime", y="True Ground Regime", color="Sites"),
            )
            fig_cm.update_layout(height=440, template="plotly_dark", margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig_cm, width="stretch")

        st.subheader("🚨 Priority Geophysical Hazard & Anomaly Watchlist")
        top_hazards = (
            enriched_df.sort("combined_risk_score", descending=True)
            .select([
                "site_id",
                "regime",
                "predicted_regime",
                "prediction_confidence_pct",
                "los_velocity_mm_yr",
                "los_seasonal_amp_mm",
                "rvi_mean",
                "coherence_mean",
                "ml_anomaly_score",
                "combined_risk_score",
            ])
            .head(25)
        )
        st.dataframe(top_hazards.to_pandas(), width="stretch")

        report_extra = {
            "band": band_name,
            "peak_deform_mm_input": float(peak_deform_mm),
            "ml_mean_risk_score": mean_ml_risk,
            "ml_accuracy": float(eval_res.accuracy),
        }
        pt_col1, pt_col2 = st.columns(2)
        pt_col1.download_button(
            "Export Point Report (GeoJSON)",
            data=build_geojson(profile, report_extra),
            file_name=f"sar_motionlens_{base_lat:.3f}_{base_lon:.3f}.geojson",
            mime="application/geo+json",
        )
        pt_col2.download_button(
            "Export Point Report (PDF)",
            data=build_pdf_report(
                "SAR-MotionLens Point Report",
                [
                    f"Location: {base_lat:.4f}, {base_lon:.4f}",
                    f"Sensor band: {band_name}",
                    f"Nearest feature: {profile.nearest_feature} ({profile.nearest_km:.0f} km)",
                    f"Displacement rate: {-profile.rate_mm_yr:.1f} mm/yr (LOS, simulated)",
                    f"Peak deformation bowl: {peak_deform_mm:.0f} mm",
                    f"Terrain (synthetic DEM): {profile.elevation_m:.0f} m, slope {profile.slope_deg:.1f} deg",
                    f"Subsidence risk class: {profile.risk_class}",
                    f"ML mean combined risk: {mean_ml_risk:.1f}/100",
                    "Note: values are simulated for demonstration, not measured InSAR products.",
                ],
            ),
            file_name=f"sar_motionlens_{base_lat:.3f}_{base_lon:.3f}.pdf",
            mime="application/pdf",
        )

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            parquet_buf = io.BytesIO()
            enriched_df.write_parquet(parquet_buf)
            st.download_button(
                label="⬇️ Download Enriched Site Catalog (Apache Parquet)",
                data=parquet_buf.getvalue(),
                file_name=f"sar_motionlens_sites_{band_name}.parquet",
                mime="application/octet-stream",
            )
        with exp_col2:
            csv_bytes = enriched_df.write_csv().encode("utf-8")
            st.download_button(
                label="⬇️ Download Enriched Site Catalog (CSV)",
                data=csv_bytes,
                file_name=f"sar_motionlens_sites_{band_name}.csv",
                mime="text/csv",
            )

    st.divider()
    st.markdown(
        '<div class="dev-footer"><b>Developer:</b> Alireza Sani<br>'
        '<b>Email:</b> <a href="mailto:alirezafazeli@live.com">alirezafazeli@live.com</a><br>'
        '<b>GitHub:</b> <a href="https://github.com/fazeli1977-dot" target="_blank">fazeli1977-dot</a></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_dashboard()
