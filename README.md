# SAR-MotionLens: Multi-Frequency InSAR & Polarimetric SAR Earth Surface Dynamics Lab

[![NASA Space Apps Challenge 2026](https://img.shields.io/badge/NASA%20Space%20Apps-2026%20Challenge-0B3D91?style=for-the-badge&logo=nasa)](https://www.spaceappschallenge.org/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Lab-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Regime%20Classifier-2E7D32?style=for-the-badge)](https://lightgbm.readthedocs.io/)

> **Challenge:** *Through the Radar Looking Glass: Revealing Earth Processes with SAR*
> **Author:** Alireza Sani (azeli1977-dot)

---

## 1. Executive Summary

**SAR-MotionLens** is an interactive scientific laboratory and machine learning pipeline designed to reveal, quantify, and classify dynamic processes on Earth's surface using **Synthetic Aperture Radar (SAR)** across **X-Band, C-Band (Sentinel-1), and L-Band (NISAR / ALOS-4)** frequencies.

1. **Aquifer Subsidence (quifer_subsidence):** Secular inelastic compaction driven by groundwater overdraft, combined with seasonal poroelastic oscillation.
2. **Permafrost Thaw & Active Layer Breathing (permafrost_thaw):** High-amplitude seasonal frost heave and thaw settlement accompanied by gradual thermokarst subsidence.
3. **Wetland & Floodplain Inundation (loodplain_inundation):** Specular water-surface scattering dips in VV backscatter, coherence loss, and rapid hydrological phase shifts.
4. **Forest Canopy & Stable Bedrock (orest_canopy_stable):** High cross-polarized volume scattering (VH), elevated Radar Vegetation Index (RVI), and near-zero crustal deformation.

---

## 2. System Architecture

```
SAR-MotionLens/
|-- app/
|   -- streamlit_app.py           # 4-Tab Interactive Mission Dashboard (PyDeck 3D + Plotly + Exports)
|-- src/sar_motionlens/
|   |-- __init__.py                # Package metadata
|   |-- physics.py                 # DInSAR phase wrapping, coherence decorrelation, RVI, harmonic OLS
|   |-- synthetic.py               # Multi-band spatiotemporal SAR cube & 2D interferogram generator
|   -- ml_classifier.py           # Stratified 5-Fold LightGBM classifier + IsolationForest anomaly engine
|-- tests/
|   |-- test_physics.py            # Scientific unit tests for radar equations & bounds
|   |-- test_ml.py                 # ML accuracy & anomaly detection verification
|   -- test_app_smoke.py          # End-to-end pipeline & Parquet serialization smoke test
|-- data/                          # Generated sample Parquet spatiotemporal cubes
-- pyproject.toml                 # Dependencies & pytest/ruff configuration
```

---

## 3. Quickstart (Windows PowerShell)

```powershell
cd D:\NASA2026\SAR-MotionLens
.\.venv\Scripts\python.exe -m pytest tests -v
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py
```

---

## 🛰️ NASA Space Apps Challenge — Earth Observation Suite by Alireza Sani

Explore all three Earth Science & Remote Sensing platforms developed for the NASA Space Apps Challenge:

| Project | Domain & Sensors | Live Repository |
| :--- | :--- | :--- |
| **1. SAR-MotionLens** | Synthetic Aperture Radar (Sentinel-1 / NISAR) InSAR Deformation, Flood & Landslide Intelligence | [github.com/fazeli1977-dot/SAR-MotionLens](https://github.com/fazeli1977-dot/SAR-MotionLens) |
| **2. FireSync-MODIS-VIIRS** | Multi-Sensor Active Wildfire Detection, FRP Fusion & Smoke Plume Tracking (MODIS + VIIRS) | [github.com/fazeli1977-dot/FireSync-MODIS-VIIRS](https://github.com/fazeli1977-dot/FireSync-MODIS-VIIRS) |
| **3. ClimateLens** | Global Climate Anomaly Analytics, Earth Observation Telemetry & Interactive Forecasting | [github.com/fazeli1977-dot/ClimateLens](https://github.com/fazeli1977-dot/ClimateLens) |
