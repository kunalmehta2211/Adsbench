"""Generate realistic fixtures and dry-run the full pipeline end to end.

Usage: python tests/dryrun.py
Creates a fake SGO CSV (realistic headers, report versions, cross-entity
duplicates), fake CPUC-derived VMT tables, and a fake benchmark, then runs
ingest -> dedup -> run_real with small sampler settings. Verifies the code
paths, not the science; the science check is src/synthetic.py.
"""

import sys
from pathlib import Path as _P
_b = _P(__file__).resolve().parents[1] / 'data/processed/benchmark.csv'
if _b.exists() and 'YES_' in _b.read_text():
    sys.exit('REFUSING: real benchmark data present. Dry run would overwrite '
             'data/processed and data/interim with fixtures. Move real data '
             'aside or run in a fresh copy of the repo.')
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(11)

# ---- fake SGO CSV with realistic headers ----
sgo_dir = ROOT / "data/raw/sgo"
sgo_dir.mkdir(parents=True, exist_ok=True)
rows = []
rid = 0
for op, n, inj_p in [("Waymo LLC", 160, 0.15), ("Zoox, Inc.", 18, 0.2)]:
    for _ in range(n):
        rid += 1
        date = pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 1600)))
        inj = rng.random() < inj_p
        base = dict(**{
            "Report ID": f"FIX-{rid:05d}",
            "Report Version": 1,
            "Reporting Entity": op,
            "Report Type": "ADS",
            "Incident Date": date.strftime("%m/%d/%Y"),
            "Incident Time (24:00)": "14:00",
            "City": rng.choice(["San Francisco", "Daly City", "San Mateo"]),
            "State": "CA",
            "Highest Injury Severity Alleged": "Minor" if inj else "No Injuries Reported",
            "SV Any Air Bags Deployed?": "Yes" if rng.random() < 0.05 else "No",
            "CP Any Air Bags Deployed?": "No",
            "SV Was Vehicle Towed?": "Yes" if rng.random() < 0.1 else "No",
            "Automation System Engaged?": "ADS Engaged",
            "VIN": f"5YJ{rng.integers(10**14, 10**15)}",
            "Incident Time (24:00)": f"{rng.integers(0,24):02d}:00",
            "Driver / Operator Type": "None" if rng.random() < 0.9 else "In-Vehicle",
            "Narrative": "[REDACTED]" if rng.random() < 0.3 else ("Low speed contact during passenger trip." if rng.random() < 0.6 else "Low speed contact."),
        })
        rows.append(base)
        if rng.random() < 0.15:  # version churn
            v2 = dict(base); v2["Report Version"] = 2
            rows.append(v2)
        if rng.random() < 0.08:  # cross-entity duplicate, same day/city
            dup = dict(base); rid += 1
            dup["Report ID"] = f"FIX-{rid:05d}"
            rows.append(dup)
pd.DataFrame(rows).to_csv(sgo_dir / "SGO_Incident_Reports_ADS.csv", index=False)
print(f"fixture SGO rows: {len(rows)}")

# ---- ingest + dedup ----
for script in ("ingest_sgo.py", "dedup.py"):
    r = subprocess.run([sys.executable, str(ROOT / "src" / script)],
                       capture_output=True, text=True)
    print(f"--- {script} ---\n{r.stdout}")
    if r.returncode != 0:
        sys.exit(f"{script} failed:\n{r.stderr}")

# ---- fake processed exposure + benchmark ----
proc = ROOT / "data/processed"
proc.mkdir(parents=True, exist_ok=True)
qs = [str(q) for q in pd.period_range("2022Q1", "2026Q2", freq="Q")]
vrows = []
for op, base in [("waymo", 2.5e6), ("zoox", 4e4)]:
    for q in qs:
        if op == "zoox" and q < "2024Q1":
            continue
        vrows.append(dict(operator=op, quarter=q,
                          vmt_reported=base * float(rng.lognormal(0, 0.05)),
                          filing="fixture.xlsx", is_latest=True))
pd.DataFrame(vrows).to_csv(proc / "vmt.csv", index=False)
pd.DataFrame([dict(operator="waymo", tau=0.05, source="fixture"),
              dict(operator="zoox", tau=0.15, source="fixture")]
             ).to_csv(proc / "tau.csv", index=False)
pd.DataFrame([dict(metro="california", severity="any_injury",
                   H=5200, V=2.0e9, r_alpha=88.0, r_beta=12.0,
                   verified="FIXTURE")]).to_csv(proc / "benchmark.csv", index=False)
print("fixture exposure + benchmark written")

# ---- fit with small sampler settings ----
import yaml
cfg_path = ROOT / "config.yaml"
cfg = yaml.safe_load(cfg_path.read_text())
orig = dict(cfg["model"])
cfg["model"].update(draws=400, tune=400, chains=2)
cfg_path.write_text(yaml.dump(cfg))
try:
    r = subprocess.run([sys.executable, str(ROOT / "src/run_real.py")],
                       capture_output=True, text=True)
    print(f"--- run_real.py ---\n{r.stdout}")
    if r.returncode != 0:
        sys.exit(f"run_real failed:\n{r.stderr[-3000:]}")
finally:
    cfg["model"].update(orig)
    cfg_path.write_text(yaml.dump(cfg))
print("DRY RUN COMPLETE")
