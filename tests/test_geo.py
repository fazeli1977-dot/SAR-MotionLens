"""Tests for coordinate-driven analysis and exports."""

import json
from sar_motionlens.geo import PRESETS, analyze_location, build_geojson, build_pdf_report


def test_profiles_differ_by_location() -> None:
    mexico = analyze_location(*PRESETS["Mexico City Basin (Mexico)"])
    open_land = analyze_location(-25.0, 135.0)
    assert mexico.rate_mm_yr > open_land.rate_mm_yr
    assert mexico.risk_class == "Critical"


def test_exports() -> None:
    p = analyze_location(35.6892, 51.3890)
    gj = json.loads(build_geojson(p))
    assert gj["features"][0]["geometry"]["coordinates"] == [p.lon, p.lat]
    assert build_pdf_report("t", ["a (b)"]).startswith(b"%PDF")


def test_scenarios_and_helpers() -> None:
    from sar_motionlens.geo import SCENARIOS, explain_risk, fringes_for_displacement
    assert all(s['preset'] in PRESETS for s in SCENARIOS)
    assert 'Critical' in explain_risk(analyze_location(*PRESETS['Mexico City Basin (Mexico)']))
    assert abs(fringes_for_displacement(-119.2, 0.2384) - 1.0) < 1e-6
