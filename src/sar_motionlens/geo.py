"""Location-driven deformation context: turns any (lat, lon) into simulation parameters.

The terrain model is a deterministic synthetic DEM proxy (not real SRTM); hotspot rates are
illustrative values for demonstration, not measured velocities.
"""

from __future__ import annotations
import json
import zlib
from dataclasses import asdict, dataclass
import numpy as np

# name: (lat, lon, kind, peak LOS rate mm/yr, influence radius km, strike deg)
HOTSPOTS: dict[str, tuple[float, float, str, float, float, float]] = {
    "Tehran-Shahriar Plain (Iran)": (35.60, 51.05, "aquifer", 160.0, 45.0, 0.0),
    "San Andreas Fault (California, USA)": (35.80, -120.40, "fault", 35.0, 60.0, 135.0),
    "Central Valley Aquifer (California, USA)": (36.30, -119.60, "aquifer", 110.0, 80.0, 0.0),
    "Mount Etna (Sicily, Italy)": (37.75, 14.99, "volcano", 60.0, 30.0, 0.0),
    "Mexico City Basin (Mexico)": (19.43, -99.13, "aquifer", 170.0, 40.0, 0.0),
    "Tokyo Bay & Kanto Plain (Japan)": (35.68, 139.76, "aquifer", 30.0, 40.0, 0.0),
    "Mekong Delta (Vietnam)": (10.03, 105.78, "delta", 45.0, 90.0, 0.0),
    "Jakobshavn Glacier (Greenland)": (69.16, -49.83, "glacier", 175.0, 60.0, 90.0),
    "North Anatolian Fault (Turkey)": (40.75, 30.40, "fault", 25.0, 60.0, 90.0),
    "Jakarta Coast (Indonesia)": (-6.12, 106.83, "delta", 120.0, 35.0, 0.0),
}

PRESETS: dict[str, tuple[float, float]] = {k: (v[0], v[1]) for k, v in HOTSPOTS.items()}


@dataclass(frozen=True)
class LocationProfile:
    lat: float
    lon: float
    elevation_m: float
    slope_deg: float
    nearest_feature: str
    nearest_kind: str
    nearest_km: float
    rate_mm_yr: float
    peak_deform_mm: float
    deformation_scale: float
    bowl_angle_deg: float
    fault_creep: bool
    seed_offset: int
    risk_class: str


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return float(6371.0 * 2.0 * np.arcsin(np.sqrt(a)))


def _synthetic_elevation(lat: float, lon: float) -> float:
    e = 900.0 + 700.0 * np.sin(lat * 0.37) * np.cos(lon * 0.29) + 400.0 * np.sin(lat * 1.3 + lon * 0.9)
    e += 25.0 * abs(lat) if abs(lat) > 60 else 0.0
    return float(max(e, 0.0))


def _synthetic_slope_deg(lat: float, lon: float) -> float:
    d = 0.05
    dy = (_synthetic_elevation(lat + d, lon) - _synthetic_elevation(lat - d, lon)) / (2 * d * 111_000.0)
    dx = (_synthetic_elevation(lat, lon + d) - _synthetic_elevation(lat, lon - d)) / (
        2 * d * 111_000.0 * max(np.cos(np.radians(lat)), 0.05)
    )
    return float(np.degrees(np.arctan(np.hypot(dx, dy))))


def classify_risk(rate_mm_yr: float) -> str:
    r = abs(rate_mm_yr)
    return "Stable" if r < 15.0 else "Moderate" if r < 50.0 else "Critical"


def analyze_location(lat: float, lon: float) -> LocationProfile:
    """Derive displacement rate and geometry for a point from hotspot proximity and terrain."""
    lat, lon = float(lat), float(lon)
    elev = _synthetic_elevation(lat, lon)
    slope = _synthetic_slope_deg(lat, lon)

    # Flat low-lying terrain is more susceptible to compaction-driven subsidence.
    susceptibility = float(np.exp(-slope / 3.0) * np.exp(-elev / 1500.0))
    rate = 3.0 + 12.0 * susceptibility

    nearest = ("Open terrain", "background", float("inf"), 0.0)
    best_contrib = 0.0
    nearest_d = float("inf")
    for name, (hlat, hlon, kind, peak, radius, strike) in HOTSPOTS.items():
        d = haversine_km(lat, lon, hlat, hlon)
        contrib = peak * float(np.exp(-((d / radius) ** 2)))
        rate += contrib
        if d < nearest_d:
            nearest_d = d
            nearest = (name, kind, d, strike)
        best_contrib = max(best_contrib, contrib)
    if best_contrib < 5.0:
        nearest = ("Open terrain", "background", nearest_d, 0.0)

    rate = float(np.clip(rate, 2.0, 180.0))
    kind = nearest[1]
    return LocationProfile(
        lat=lat,
        lon=lon,
        elevation_m=elev,
        slope_deg=slope,
        nearest_feature=nearest[0],
        nearest_kind=kind,
        nearest_km=float(nearest[2]),
        rate_mm_yr=rate,
        peak_deform_mm=-float(np.clip(rate, 10.0, 180.0)),
        deformation_scale=float(np.clip(rate / 60.0, 0.15, 2.5)),
        bowl_angle_deg=float(nearest[3]) if kind == "fault" else float(zlib.crc32(f"{lat:.3f},{lon:.3f}".encode()) % 180),
        fault_creep=kind == "fault" and best_contrib >= 5.0,
        seed_offset=int(zlib.crc32(f"{lat:.2f},{lon:.2f}".encode()) % 10_000),
        risk_class=classify_risk(rate),
    )


SCENARIOS: list[dict[str, str]] = [
    {
        "title": "Sinking Capital: Tehran Plain",
        "preset": "Tehran-Shahriar Plain (Iran)",
        "band": "L-Band (NISAR 24cm)",
        "story": "Decades of groundwater pumping compact the aquifer beneath the Tehran-Shahriar plain. "
        "The ground slowly settles, and radar sees it as a bowl of concentric fringes.",
    },
    {
        "title": "A Fault That Creeps: San Andreas",
        "preset": "San Andreas Fault (California, USA)",
        "band": "C-Band (Sentinel-1 5.6cm)",
        "story": "Along the creeping section of the San Andreas, the two sides of the fault slide past each other. "
        "In an interferogram this appears as a sharp phase step across the fault trace.",
    },
    {
        "title": "Sinking City: Mexico City",
        "preset": "Mexico City Basin (Mexico)",
        "band": "L-Band (NISAR 24cm)",
        "story": "Mexico City sits on an ancient lake bed. Heavy water extraction makes it one of the fastest "
        "subsiding major cities on Earth.",
    },
    {
        "title": "A Breathing Volcano: Mount Etna",
        "preset": "Mount Etna (Sicily, Italy)",
        "band": "S-Band (NISAR 9.4cm)",
        "story": "Magma moving beneath Etna inflates and deflates the volcano. SAR measures these millimetre-scale "
        "changes from orbit, day and night, through clouds.",
    },
    {
        "title": "Ice on the Move: Jakobshavn Glacier",
        "preset": "Jakobshavn Glacier (Greenland)",
        "band": "L-Band (NISAR 24cm)",
        "story": "Jakobshavn is one of the fastest flowing glaciers. Long-wavelength L-band keeps coherence "
        "over moving ice where shorter wavelengths decorrelate.",
    },
]


def explain_risk(profile: LocationProfile) -> str:
    """One-sentence plain-language reason for the point's risk class."""
    kind_text = {
        "aquifer": "an aquifer-compaction zone",
        "fault": "an active fault",
        "volcano": "a volcanic edifice",
        "delta": "a subsiding delta or coastal plain",
        "glacier": "a fast-flowing glacier",
        "background": "open terrain with no mapped deformation source nearby",
    }[profile.nearest_kind]
    terrain = "flat, low-lying terrain that compacts easily" if profile.slope_deg < 1.0 else "sloped terrain"
    return (
        f"Classified {profile.risk_class}: simulated LOS rate of {profile.rate_mm_yr:.0f} mm/yr, "
        f"{profile.nearest_km:.0f} km from {profile.nearest_feature} ({kind_text}), on {terrain}."
    )


def fringes_for_displacement(displacement_mm: float, wavelength_m: float) -> float:
    """Number of full 2*pi fringes produced by a LOS displacement (lambda/2 per fringe)."""
    return abs(displacement_mm) / (wavelength_m * 500.0)


def build_geojson(profile: LocationProfile, extra: dict | None = None) -> str:
    props = asdict(profile) | (extra or {})
    feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [profile.lon, profile.lat]},
        "properties": props,
    }
    return json.dumps({"type": "FeatureCollection", "features": [feature]}, indent=2)


def build_pdf_report(title: str, lines: list[str]) -> bytes:
    """Minimal single-page text PDF (no external dependency)."""

    def esc(s: str) -> str:
        s = s.encode("latin-1", "replace").decode("latin-1")
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content = ["BT /F1 16 Tf 50 790 Td (" + esc(title) + ") Tj ET"]
    y = 760
    for line in lines:
        content.append(f"BT /F1 10 Tf 50 {y} Td ({esc(line)}) Tj ET")
        y -= 16
    stream = "\n".join(content).encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)
