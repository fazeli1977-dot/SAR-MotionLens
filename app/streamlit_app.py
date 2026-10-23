"""SAR-MotionLens: Multi-Band InSAR & PolSAR Spatiotemporal Earth Dynamics Explorer.

Interactive Streamlit Mission Control Dashboard built for the NASA Space Apps Challenge
("Through the Radar Looking Glass: Revealing Earth Processes with SAR").
"""

from __future__ import annotations
import io
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import pydeck as pdk
import streamlit as st

from sar_motionlens.ml_classifier import train_and_evaluate_sar_classifier
from sar_motionlens.physics import RADAR_BANDS
from sar_motionlens.synthetic import (
    REGIME_CLASSES,
    generate_2d_interferogram_grid,
    generate_sar_spatiotemporal_cube,
)

st.set_page_config(
    page_title="SAR-MotionLens | NASA SAR Earth Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
):
    """Cached generator and LightGBM trainer for rapid interactive exploration."""
    ts_df, site_df = generate_sar_spatiotemporal_cube(
        n_sites_per_regime=n_sites_per_regime,
        n_epochs=n_epochs,
        band_name=band_name,
        seed=seed,
        base_lat=base_lat,
        base_lon=base_lon,
    )
    eval_res = train_and_evaluate_sar_classifier(site_df, seed=seed)
    return ts_df, eval_res


def render_dashboard() -> None:
    """Main entry point for the SAR-MotionLens Streamlit application."""
    st.title("🛰️ SAR-MotionLens: Multi-Band InSAR & PolSAR Earth Dynamics Intelligence")
    st.markdown(
        "**Through the Radar Looking Glass: Revealing Earth Processes with SAR** — "
        "Fusing **Sentinel-1 (C-Band)** and **NISAR (L/S-Band)** Interferometric Phase ($\\Delta\\phi$) "
        "and Dual-Polarization Backscatter ($\\sigma^0_{VV}, \\sigma^0_{VH}$, RVI, Entropy $H$) with "
        "**LightGBM** & **IsolationForest** to classify and monitor critical Earth hazards day and night."
    )

    with st.sidebar:
        st.subheader("Global Study Region")
        REGIONS = {
            "San Andreas & Central Valley (California, USA)": (36.60, -119.85),
            "Jakobshavn Glacier (Greenland)": (69.16, -49.83),
            "Mount Etna Volcano (Sicily, Italy)": (37.75, 14.99),
            "Mekong Delta Floodplain (Vietnam)": (10.03, 105.78),
            "Tokyo Bay & Kanto Plain (Japan)": (35.68, 139.76),
            "Tehran Plain & Alborz (Iran)": (35.68, 51.42),
            "Custom Coordinates (Enter Lat/Lon)": None,
        }
        selected_region = st.selectbox("Target Geospatial Hotspot", list(REGIONS.keys()), index=0)
        if REGIONS[selected_region] is None:
            c_lat, c_lon = st.columns(2)
            base_lat = c_lat.number_input("Latitude", -85.0, 85.0, 36.60, 0.1)
            base_lon = c_lon.number_input("Longitude", -180.0, 180.0, -119.85, 0.1)
        else:
            base_lat, base_lon = REGIONS[selected_region]
            st.caption(f"Center: Lat {base_lat:.2f} deg, Lon {base_lon:.2f} deg")
        st.divider()
        st.header("🎛️ SAR Sensor & Mission Controls")
        band_name = st.selectbox(
            "Select SAR Sensor Band",
            options=list(RADAR_BANDS.keys()),
            index=0,
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
            "Peak Deformation Bowl (mm)", min_value=-180.0, max_value=-10.0, value=-75.0, step=5.0
        )

    ts_df, eval_res = load_mission_cube_and_ml(
        n_sites_per_regime=n_sites_per_regime,
        n_epochs=n_epochs,
        band_name=band_name,
        seed=int(seed),
        base_lat=float(base_lat),
        base_lon=float(base_lon),
    )
    enriched_df = eval_res.enriched_site_df

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
            grid_data = generate_2d_interferogram_grid(
                grid_size=110,
                peak_subsidence_mm=peak_deform_mm,
                wavelength_m=band_spec.wavelength_m,
                seed=int(seed),
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
            st.plotly_chart(fig_insar, use_container_width=True)
            st.info(
                f"**Physical Insight ({band_name}):** Each full color cycle ($2\\pi$ radians) corresponds to "
                f"**{band_spec.wavelength_m * 500:.2f} mm** of Line-of-Sight motion ($\\lambda/2$). "
                "Switch between **C-Band** and **L-Band** in the sidebar to see how shorter wavelengths produce "
                "denser interferometric fringes for the exact same ground deformation!"
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
        st.plotly_chart(fig_ts, use_container_width=True)

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
            st.plotly_chart(fig_fi, use_container_width=True)

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
            st.plotly_chart(fig_cm, use_container_width=True)

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
        st.dataframe(top_hazards.to_pandas(), use_container_width=True)

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


if __name__ == "__main__":
    render_dashboard()