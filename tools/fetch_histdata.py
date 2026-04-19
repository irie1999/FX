"""Download HistData.com Generic ASCII M1 archives automatically.

HistData requires a form POST with a session token parsed from the HTML of
the download page. This script:

  1. Fetches the year (or month) page
  2. Extracts hidden form fields including the `tk` token
  3. POSTs to /get.php with a Referer header
  4. Extracts the CSV out of the returned ZIP to the output directory

Stdlib only; no extra pip installs required.

Usage:
    python tools/fetch_histdata.py --pair USDJPY --years 2023 2024 2025
    python tools/fetch_histdata.py --pair EURUSD --months 2026-01 2026-02
    python tools/fetch_histdata.py --pair USDJPY --years 2024 --out data/raw

Notes:
    - Completed past years are downloaded as a single yearly ZIP.
    - The current year must be fetched month-by-month via --months.
    - HistData terms permit personal/research use. Respect their bandwidth
      by not re-downloading data you already have (--skip-existing, default).
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://www.histdata.com"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Fields we need to echo back in the POST to get.php
FORM_FIELDS = ("tk", "date", "datemonth", "platform", "timeframe", "fxpair")


def _http_get(url: str, referer: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if referer:
        req.add_header("Referer", referer)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _http_post(url: str, data: dict[str, str], referer: str) -> bytes:
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={
            "User-Agent": UA,
            "Referer": referer,
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": BASE,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _parse_form(html: str) -> dict[str, str]:
    """Pull required hidden inputs out of the download-page form."""
    fields: dict[str, str] = {}
    for name in FORM_FIELDS:
        # <input ... name="tk" ... value="abc123">  (attribute order varies)
        pattern = rf'<input[^>]*\bname=["\']{name}["\'][^>]*\bvalue=["\']([^"\']*)["\']'
        m = re.search(pattern, html, re.IGNORECASE)
        if not m:
            # value-before-name variant
            pattern2 = rf'<input[^>]*\bvalue=["\']([^"\']*)["\'][^>]*\bname=["\']{name}["\']'
            m = re.search(pattern2, html, re.IGNORECASE)
        if not m:
            raise RuntimeError(f"Could not find form field '{name}' on download page")
        fields[name] = m.group(1)
    return fields


def _detail_url(pair: str, year: int, month: int | None = None) -> str:
    path = f"/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair.lower()}/{year}"
    if month is not None:
        path += f"/{month}"
    return BASE + path


def _download_one(pair: str, year: int, month: int | None, out_dir: Path, skip_existing: bool) -> Path | None:
    """Download and extract a single year or month archive. Returns extracted CSV path."""
    label = f"{pair} {year}" + (f"-{month:02d}" if month else "")
    # Predict extracted filename pattern: DAT_ASCII_{PAIR}_M1_{YYYY}[MM].csv
    expected = f"DAT_ASCII_{pair.upper()}_M1_{year}" + (f"{month:02d}" if month else "") + ".csv"
    target = out_dir / expected
    if skip_existing and target.exists() and target.stat().st_size > 0:
        print(f"[skip] {label} -> {target} already exists")
        return target

    detail = _detail_url(pair, year, month)
    print(f"[fetch] {label} page: {detail}")
    html = _http_get(detail).decode("utf-8", errors="replace")
    fields = _parse_form(html)

    print(f"[post ] {label} get.php")
    try:
        blob = _http_post(f"{BASE}/get.php", fields, referer=detail)
    except urllib.error.HTTPError as e:
        print(f"[error] {label}: HTTP {e.code} {e.reason}", file=sys.stderr)
        return None

    if not blob[:2] == b"PK":
        snippet = blob[:200].decode("utf-8", errors="replace")
        raise RuntimeError(f"{label}: server did not return a ZIP. First bytes: {snippet!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: Path | None = None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            dest = out_dir / Path(name).name
            with zf.open(name) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            print(f"[save ] {dest} ({dest.stat().st_size:,} bytes)")
            extracted = dest
    if extracted is None:
        raise RuntimeError(f"{label}: no CSV found inside ZIP")
    return extracted


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download HistData Generic ASCII M1 data")
    ap.add_argument("--pair", default="USDJPY", help="Currency pair, e.g. USDJPY, EURUSD")
    ap.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=[],
        help="Completed years to fetch as annual ZIPs, e.g. --years 2023 2024",
    )
    ap.add_argument(
        "--months",
        nargs="*",
        default=[],
        help="Specific months in YYYY-MM, e.g. --months 2026-01 2026-02 (for current year)",
    )
    ap.add_argument("--out", type=Path, default=Path("data/raw"), help="Output directory")
    ap.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download even if the target CSV already exists",
    )
    args = ap.parse_args(argv)

    if not args.years and not args.months:
        ap.error("Provide --years and/or --months")

    skip_existing = not args.no_skip_existing
    saved: list[Path] = []

    for y in args.years:
        try:
            p = _download_one(args.pair, y, None, args.out, skip_existing)
        except Exception as e:
            print(f"[error] {args.pair} {y}: {e}", file=sys.stderr)
            continue
        if p:
            saved.append(p)

    for ym in args.months:
        try:
            year_str, mon_str = ym.split("-")
            y = int(year_str)
            m = int(mon_str)
        except ValueError:
            print(f"[error] bad --months value {ym!r}, expected YYYY-MM", file=sys.stderr)
            continue
        try:
            p = _download_one(args.pair, y, m, args.out, skip_existing)
        except Exception as e:
            print(f"[error] {args.pair} {y}-{m:02d}: {e}", file=sys.stderr)
            continue
        if p:
            saved.append(p)

    if saved:
        print(f"\nDone. {len(saved)} file(s) in {args.out.resolve()}")
        print("\nNext step:")
        glob_path = str(args.out / f"DAT_ASCII_{args.pair.upper()}_M1_*.csv")
        print(f'  python -m fx.main --histdata "{glob_path}" --resample 1h')
        return 0
    print("No files downloaded.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
