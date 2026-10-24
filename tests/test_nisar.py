"""Tests for NISAR catalog parsing (network mocked)."""

from sar_motionlens import nisar


class _Resp:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {
            "features": [
                {
                    "geometry": {"type": "Polygon", "coordinates": [[[50.0, 35.0], [52.0, 35.0], [52.0, 36.0], [50.0, 35.0]]]},
                    "properties": {
                        "fileID": "NISAR_L2_X",
                        "processingLevel": "GUNW",
                        "startTime": "2026-10-09T15:26:20Z",
                        "flightDirection": "DESCENDING",
                        "pathNumber": 136,
                        "url": "https://example.org/x.h5",
                    },
                }
            ]
        }


def test_search_nisar_parses_granules(monkeypatch) -> None:
    monkeypatch.setattr(nisar.requests, "get", lambda *a, **k: _Resp())
    g = nisar.search_nisar(35.7, 51.4)
    assert len(g) == 1 and g[0].level == "GUNW" and g[0].path_number == 136
    assert g[0].footprint[0] == [50.0, 35.0]
