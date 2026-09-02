from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAVER_MAP_URL = "https://map.naver.com/p/search/%EC%95%88%EA%B3%BC"
TEMPLATES = (
    "web/templates/result.html",
    "web/templates/m_result.html",
)


def test_nearby_clinic_link_opens_naver_map_outside_iframe():
    for relative_path in TEMPLATES:
        template = (ROOT / relative_path).read_text(encoding="utf-8")
        assert f'href="{NAVER_MAP_URL}"' in template
        assert 'target="_blank"' in template
        assert 'rel="noopener noreferrer"' in template
        assert 'onclick="openNearbyClinicMap()"' not in template
        assert "map.naver.com/v5/search" not in template
