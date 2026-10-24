<div align="center">

# SAR-MotionLens

### Multi-Frequency InSAR & Polarimetric SAR Earth Surface Dynamics Lab

*Through the Radar Looking Glass: Revealing Earth Processes with SAR*

[![NASA Space Apps 2026](https://img.shields.io/badge/NASA%20Space%20Apps-2026-0B3D91?style=for-the-badge&logo=nasa)](https://www.spaceappschallenge.org/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FC3D21?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Classifier-2E7D32?style=for-the-badge)](https://lightgbm.readthedocs.io/)

[**Run Locally**](#run-the-app) | [**Deploy on Streamlit Cloud**](https://share.streamlit.io/deploy?repository=fazeli1977-dot/SAR-MotionLens&branch=main&mainModule=app/streamlit_app.py) | [**Developer**](#developer)

</div>

---

## Overview

SAR-MotionLens is an interactive lab that simulates, analyzes and classifies Earth-surface dynamics with **Synthetic Aperture Radar** across **X-, C- (Sentinel-1) and L-Band (NISAR / ALOS-4)**.

Click anywhere on the world map, enter latitude/longitude, or pick a strategic preset. The platform then generates for that point:

- an InSAR wrapped-phase interferogram and unwrapped LOS deformation map
- a LOS displacement and PolSAR time series (backscatter, RVI, entropy, coherence)
- a LightGBM regime classification with IsolationForest anomaly scoring
- a subsidence risk class: **Stable**, **Moderate** or **Critical**
- a downloadable point report in **GeoJSON** and **PDF**

> Data are physically motivated **simulations** for demonstration; deformation rates near presets are illustrative, and the terrain model is a synthetic DEM proxy, not SRTM.

## Earth-process regimes

| Regime | Signature |
| :--- | :--- |
| `Subsidence_Aquifer` | Secular LOS compaction plus seasonal poroelastic cycle |
| `Glacier_Creep` | Fast ice flow, summer coherence loss |
| `Floodplain_Wetland` | Specular backscatter drops, rapid phase shifts |
| `Crop_Phenology` | High cross-pol volume scattering, near-zero deformation |

## Run the app

```powershell
git clone https://github.com/fazeli1977-dot/SAR-MotionLens.git
cd SAR-MotionLens
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\streamlit run app/streamlit_app.py
```

Then open **http://localhost:8501**. Run the tests with `.\.venv\Scripts\python -m pytest`.

## Project structure

```
SAR-MotionLens/
|-- app/streamlit_app.py          # Dashboard (point selection, maps, charts, exports)
|-- src/sar_motionlens/
|   |-- geo.py                    # Coordinate -> deformation profile, GeoJSON/PDF export
|   |-- physics.py                # Phase wrapping, coherence, RVI, harmonic fit
|   |-- synthetic.py              # Spatiotemporal cube and 2D interferogram generator
|   `-- ml_classifier.py          # LightGBM classifier + IsolationForest
|-- tests/                        # Unit and smoke tests
`-- .streamlit/config.toml        # NASA-style dark theme
```

## NASA Space Apps Earth Observation Suite

| Project | Domain | Repository |
| :--- | :--- | :--- |
| **SAR-MotionLens** (2026) | InSAR deformation and SAR regime classification | [fazeli1977-dot/SAR-MotionLens](https://github.com/fazeli1977-dot/SAR-MotionLens) |
| **FireSync-MODIS-VIIRS** (2025) | Multi-sensor active wildfire detection and FRP fusion | [fazeli1977-dot/FireSync-MODIS-VIIRS](https://github.com/fazeli1977-dot/FireSync-MODIS-VIIRS) |
| **ClimateLens** (2024) | Climate anomaly analytics and forecasting | [fazeli1977-dot/ClimateLens](https://github.com/fazeli1977-dot/ClimateLens) |

## Developer

**Alireza Fazeli** | [alirezafazeli@live.com](mailto:alirezafazeli@live.com) | [github.com/fazeli1977-dot](https://github.com/fazeli1977-dot)
