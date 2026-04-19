import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "fetch_histdata", ROOT / "tools" / "fetch_histdata.py"
)
fetch_histdata = importlib.util.module_from_spec(spec)
sys.modules["fetch_histdata"] = fetch_histdata
spec.loader.exec_module(fetch_histdata)


SAMPLE_FORM = """
<html><body>
<form id="file_down" method="POST" action="get.php">
  <input type="hidden" id="tk" name="tk" value="ABC123def">
  <input type="hidden" id="date" name="date" value="2024">
  <input type="hidden" id="datemonth" name="datemonth" value="2024">
  <input type="hidden" id="platform" name="platform" value="ASCII">
  <input type="hidden" id="timeframe" name="timeframe" value="M1">
  <input type="hidden" id="fxpair" name="fxpair" value="USDJPY">
</form>
</body></html>
"""


def test_parse_form_extracts_all_fields():
    fields = fetch_histdata._parse_form(SAMPLE_FORM)
    assert fields == {
        "tk": "ABC123def",
        "date": "2024",
        "datemonth": "2024",
        "platform": "ASCII",
        "timeframe": "M1",
        "fxpair": "USDJPY",
    }


def test_parse_form_handles_attribute_order():
    html = (
        '<input value="XYZ" name="tk">'
        '<input name="date" value="2024">'
        '<input value="2024" name="datemonth">'
        '<input name="platform" value="ASCII">'
        '<input name="timeframe" value="M1">'
        '<input name="fxpair" value="USDJPY">'
    )
    fields = fetch_histdata._parse_form(html)
    assert fields["tk"] == "XYZ"


def test_parse_form_missing_raises():
    with pytest.raises(RuntimeError):
        fetch_histdata._parse_form("<html>nothing here</html>")


def test_detail_url_year_only():
    u = fetch_histdata._detail_url("USDJPY", 2024)
    assert u.endswith("/1-minute-bar-quotes/usdjpy/2024")


def test_detail_url_with_month():
    u = fetch_histdata._detail_url("EURUSD", 2026, 3)
    assert u.endswith("/1-minute-bar-quotes/eurusd/2026/3")
