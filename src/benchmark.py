"""Stage 4: human benchmark per metro and severity (paper 4.4, P5).

Inputs you provide (from acquire step):
  data/raw/switrs/switrs_crashes.csv   TIMS export (crash-level)
  data/raw/human_vmt/caltrans_vmt.csv  county, year, avmt (annual VMT, miles)
  data/verify/blincoe_corrections.csv  severity, r_mean, r_sd, verified
  data/verify/umtri_benchmark.csv      severity, rate_per_mile, verified

The two files in data/verify/ ship as templates with verified=NO and NaN
values. Fill them from the cited sources (Blincoe et al. 2023 DOT HS 813 403;
Flannagan et al. 2023) and set verified=YES. The script REFUSES to emit a
benchmark from unverified values unless --allow-unverified, which exists only
for pipeline dry-runs and stamps the output accordingly.

Output: data/processed/benchmark.csv
  metro, severity, H (crash count), V (miles), r_alpha, r_beta
where (r_alpha, r_beta) is the Beta prior matching (r_mean, r_sd).
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
SWITRS = ROOT / "data" / "raw" / "switrs" / "switrs_crashes.csv"
HVMT = ROOT / "data" / "raw" / "human_vmt" / "caltrans_vmt.csv"
BLINCOE = ROOT / "data" / "verify" / "blincoe_corrections.csv"
UMTRI = ROOT / "data" / "verify" / "umtri_benchmark.csv"
OUT = ROOT / "data" / "processed" / "benchmark.csv"

# TIMS/SWITRS severity codes: 1 fatal, 2 severe injury, 3 other visible,
# 4 complaint of pain, 0 PDO
SEV_MAP = {
    "any_injury": [1, 2, 3, 4],
    "serious_plus": [1, 2],
    "police_reportable_proxy": [0, 1, 2, 3, 4],
    "airbag_or_tow": None,  # not derivable from SWITRS; UMTRI covers it
    "sgo_any": None,        # below any police floor by construction
}


def beta_from_moments(mean: float, sd: float) -> tuple[float, float]:
    k = mean * (1 - mean) / sd**2 - 1
    if k <= 0:
        sys.exit(f"Invalid Beta moments mean={mean} sd={sd}")
    return mean * k, (1 - mean) * k


def main(allow_unverified: bool = False) -> None:
    for f in (SWITRS, HVMT, BLINCOE, UMTRI):
        if not f.exists():
            sys.exit(f"Missing {f.relative_to(ROOT)}; see module docstring.")
    corr = pd.read_csv(BLINCOE)
    if (corr["verified"].str.upper() != "YES").any():
        msg = "blincoe_corrections.csv contains unverified rows."
        if not allow_unverified:
            sys.exit(msg + " Verify against DOT HS 813 403 or pass --allow-unverified.")
        print("WARNING: " + msg + " Output stamped unverified.")

    crashes = pd.read_csv(SWITRS, dtype=str)
    # TIMS exports name these COUNTY and COLLISION_SEVERITY; adjust if needed
    ccol = next((c for c in crashes.columns if "county" in c.lower()), None)
    scol = next((c for c in crashes.columns if "severity" in c.lower()), None)
    ycol = next((c for c in crashes.columns if "year" in c.lower()), None)
    if not all([ccol, scol, ycol]):
        sys.exit(f"SWITRS columns not recognized; found {list(crashes.columns)[:20]}")
    crashes["sev"] = pd.to_numeric(crashes[scol], errors="coerce")
    crashes["year"] = pd.to_numeric(crashes[ycol], errors="coerce")

    hvmt = pd.read_csv(HVMT)
    years = range(int(CFG["study"]["start_quarter"][:4]),
                  int(CFG["study"]["end_quarter"][:4]) + 1)

    rows = []
    for metro, spec in CFG["study"]["metros"].items():
        counties = spec["switrs_counties"]
        in_metro = crashes[crashes[ccol].str.title().isin(counties)
                           & crashes["year"].isin(years)]
        V = float(hvmt[hvmt["county"].str.title().isin(counties)
                       & hvmt["year"].isin(years)]["avmt"].sum())
        for sev_name, codes in SEV_MAP.items():
            if codes is None:
                continue
            H = int(in_metro["sev"].isin(codes).sum())
            row = corr[corr["severity"] == sev_name]
            if row.empty:
                continue
            a, b = beta_from_moments(float(row.r_mean.iloc[0]),
                                     float(row.r_sd.iloc[0]))
            rows.append(dict(metro=metro, severity=sev_name, H=H, V=V,
                             r_alpha=a, r_beta=b,
                             verified=str(row.verified.iloc[0]).upper()))
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(allow_unverified="--allow-unverified" in sys.argv)
