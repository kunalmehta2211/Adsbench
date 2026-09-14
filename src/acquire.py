"""Stage 1: acquire raw public data. Run on your own machine.

What it automates and what it cannot:
  - NHTSA SGO ADS incident CSV: attempts direct download; NHTSA occasionally
    moves the file, so on failure it prints the portal URL to fetch manually.
  - CPUC quarterly reports: scrapes the CPUC AV quarterly reporting page for
    spreadsheet links and downloads them all into data/raw/cpuc/.
  - CA DMV OL316 collision reports: index page listing only (reports are
    individual PDFs; we use them for dedup spot-checks, not bulk parsing).
  - SWITRS: cannot be automated, needs a free TIMS account. Instructions
    printed.

Every download is recorded in data/SNAPSHOT.txt with a timestamp and SHA256,
which becomes the frozen-snapshot statement in the paper.
"""

import datetime as dt
import hashlib
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SNAPSHOT = ROOT / "data" / "SNAPSHOT.txt"

# Known-good as of mid-2026; if a URL 404s the script tells you where to look.
SGO_PORTAL = "https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting"
SGO_CSV_CANDIDATES = [
    # NHTSA has hosted the ADS incident file at static paths like these;
    # they refresh monthly and occasionally rename.
    "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADS.csv",
]
CPUC_QUARTERLY_PAGE = (
    "https://www.cpuc.ca.gov/regulatory-services/licensing/"
    "transportation-licensing-and-analysis-branch/autonomous-vehicle-programs/"
    "quarterly-reporting"
)
DMV_COLLISION_PAGE = (
    "https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/"
    "autonomous-vehicle-collision-reports/"
)

UA = {"User-Agent": "adsbench-research/0.1 (academic use)"}


def log_snapshot(path: Path, url: str) -> None:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    with SNAPSHOT.open("a") as f:
        f.write(f"{dt.datetime.now().isoformat()}\t{path.name}\t{sha}\t{url}\n")


def download(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, headers=UA, timeout=120)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  FAILED {url}: {e}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    log_snapshot(dest, url)
    print(f"  OK {dest.relative_to(ROOT)} ({len(r.content) / 1e6:.1f} MB)")
    return True


def acquire_sgo() -> None:
    print("[1/4] NHTSA SGO ADS incident reports")
    for url in SGO_CSV_CANDIDATES:
        if download(url, RAW / "sgo" / "SGO_Incident_Reports_ADS.csv"):
            return
    print(
        "  Could not fetch automatically. Open the portal, download the\n"
        f"  'Incident Reports - ADS' CSV, save as data/raw/sgo/SGO_Incident_Reports_ADS.csv\n"
        f"  Portal: {SGO_PORTAL}"
    )


def acquire_cpuc() -> None:
    print("[2/4] CPUC quarterly reports (VMT)")
    try:
        page = requests.get(CPUC_QUARTERLY_PAGE, headers=UA, timeout=120).text
    except requests.RequestException as e:
        print(f"  Could not load CPUC page ({e}). Open it manually:\n  {CPUC_QUARTERLY_PAGE}")
        return
    links = set(re.findall(r'href="([^"]+\.(?:xlsx|xls|csv))"', page, flags=re.I))
    if not links:
        print(f"  No spreadsheet links found; page layout may have changed:\n  {CPUC_QUARTERLY_PAGE}")
        return
    for href in sorted(links):
        url = href if href.startswith("http") else "https://www.cpuc.ca.gov" + href
        name = re.sub(r"[^A-Za-z0-9._-]", "_", url.split("/")[-1])
        download(url, RAW / "cpuc" / name)


def acquire_dmv_index() -> None:
    print("[3/4] CA DMV OL316 collision report index (for dedup spot-checks)")
    try:
        page = requests.get(DMV_COLLISION_PAGE, headers=UA, timeout=120).text
        (RAW / "dmv").mkdir(parents=True, exist_ok=True)
        (RAW / "dmv" / "collision_index.html").write_text(page)
        print("  saved data/raw/dmv/collision_index.html")
    except requests.RequestException as e:
        print(f"  Could not load DMV page ({e}): {DMV_COLLISION_PAGE}")


def switrs_instructions() -> None:
    print("[4/4] SWITRS human crash data (manual, ~10 minutes)")
    print(
        "  1. Create a free account at https://tims.berkeley.edu\n"
        "  2. SWITRS Query: counties San Francisco, San Mateo, Los Angeles;\n"
        "     years 2019-2025 (pre-pandemic year included for sensitivity);\n"
        "     all severities including PDO.\n"
        "  3. Export Crashes CSV to data/raw/switrs/switrs_crashes.csv\n"
        "  4. Human VMT: Caltrans annual VMT by county,\n"
        "     save as data/raw/human_vmt/caltrans_vmt.csv (county, year, avmt)"
    )


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    acquire_sgo()
    acquire_cpuc()
    acquire_dmv_index()
    switrs_instructions()
    print("\nDone. Verify data/SNAPSHOT.txt, then run: python src/ingest_sgo.py")
