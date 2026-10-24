"""Real NISAR granule discovery through the public ASF DAAC Search API (no login needed)."""

from __future__ import annotations
from dataclasses import dataclass
import requests

ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"


@dataclass(frozen=True)
class NisarGranule:
    file_id: str
    level: str
    start_time: str
    flight_direction: str
    path_number: int | None
    frame_number: int | None
    url: str
    footprint: list[list[float]]  # [[lon, lat], ...] outer ring


def search_nisar(lat: float, lon: float, max_results: int = 25, timeout: float = 30.0) -> list[NisarGranule]:
    """Return the newest NISAR granules intersecting a point. Raises requests.RequestException on failure."""
    params = {
        "platform": "NISAR",
        "intersectsWith": f"POINT({lon:.5f} {lat:.5f})",
        "output": "geojson",
        "maxResults": max_results,
    }
    resp = requests.get(ASF_SEARCH_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    out: list[NisarGranule] = []
    for feat in resp.json().get("features", []):
        p = feat.get("properties", {})
        geom = feat.get("geometry") or {}
        ring = (geom.get("coordinates") or [[]])[0] if geom.get("type") == "Polygon" else []
        out.append(
            NisarGranule(
                file_id=p.get("fileID", ""),
                level=p.get("processingLevel", ""),
                start_time=p.get("startTime", ""),
                flight_direction=p.get("flightDirection", ""),
                path_number=p.get("pathNumber"),
                frame_number=p.get("frameNumber"),
                url=p.get("url", ""),
                footprint=[[float(x), float(y)] for x, y in ring],
            )
        )
    return out
