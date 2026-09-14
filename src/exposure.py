"""Stage 3: operator VMT per quarter from CPUC filings (paper 4.3, P1).

CPUC templates vary across program years and carriers, so this stage works in
two passes:
  pass 1 (survey): open every spreadsheet in data/raw/cpuc/, list its sheets
     and any columns that look like mileage, and write a mapping skeleton to
     data/interim/cpuc_mapping.csv for you to confirm or correct (one row per
     file: operator, quarter, sheet, vmt_column, row_filter). ~15 minutes of
     human eyeballing, once.
  pass 2 (extract): apply the confirmed mapping, sum VMT per operator-quarter,
     and, when the same operator-quarter appears in multiple filings (original
     plus corrected), keep every value with its filing date. The relative
     spread of revisions calibrates the measurement-noise prior tau per
     operator; operators with no revision history get the config default.

Output:
  data/processed/vmt.csv: operator, quarter, vmt_reported, filing, is_latest
  data/processed/tau.csv: operator, tau  (revision-calibrated or default)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RAW = ROOT / "data" / "raw" / "cpuc"
MAPPING = ROOT / "data" / "interim" / "cpuc_mapping.csv"
OUT_VMT = ROOT / "data" / "processed" / "vmt.csv"
OUT_TAU = ROOT / "data" / "processed" / "tau.csv"

MILEAGE_HINTS = ("vmt", "vehicle miles", "miles traveled", "total miles")


def survey() -> None:
    rows = []
    for f in sorted(RAW.glob("*.xls*")) + sorted(RAW.glob("*.csv")):
        try:
            sheets = (pd.read_excel(f, sheet_name=None, nrows=5, dtype=str)
                      if f.suffix.startswith(".xls")
                      else {"csv": pd.read_csv(f, nrows=5, dtype=str)})
        except Exception as e:
            rows.append(dict(file=f.name, sheet="UNREADABLE", vmt_column=str(e)))
            continue
        for name, head in sheets.items():
            cand = [c for c in head.columns
                    if any(h in str(c).lower() for h in MILEAGE_HINTS)]
            rows.append(dict(file=f.name, sheet=name,
                             vmt_column=";".join(map(str, cand)) or "NONE_FOUND",
                             operator="", quarter="", row_filter=""))
    MAPPING.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(MAPPING, index=False)
    print(f"Survey written to {MAPPING.relative_to(ROOT)}.")
    print("Fill in operator, quarter, and confirm vmt_column for the rows that "
          "matter, delete the rest, then rerun: python src/exposure.py extract")


def extract() -> None:
    m = pd.read_csv(MAPPING, dtype=str).dropna(subset=["operator", "quarter"])
    if m.empty:
        sys.exit(f"No confirmed rows in {MAPPING}. Run the survey pass first.")
    recs = []
    for _, r in m.iterrows():
        f = RAW / r["file"]
        df = (pd.read_excel(f, sheet_name=r["sheet"])
              if f.suffix.startswith(".xls") else pd.read_csv(f))
        col = r["vmt_column"]
        if col not in df.columns:
            sys.exit(f"{f.name}: column '{col}' not found. Fix mapping.")
        vals = pd.to_numeric(df[col], errors="coerce")
        if r.get("row_filter") and not pd.isna(r["row_filter"]):
            vals = vals[df.eval(r["row_filter"])]
        recs.append(dict(operator=r["operator"], quarter=r["quarter"],
                         vmt_reported=float(vals.sum()), filing=r["file"]))
    vmt = pd.DataFrame(recs)
    vmt["is_latest"] = ~vmt.duplicated(["operator", "quarter"], keep="last")
    OUT_VMT.parent.mkdir(parents=True, exist_ok=True)
    vmt.to_csv(OUT_VMT, index=False)

    # tau calibration from revisions: sd of log(reported) across filings of
    # the same operator-quarter, pooled per operator
    taus = []
    for op, g in vmt.groupby("operator"):
        rev = g.groupby("quarter")["vmt_reported"].apply(
            lambda v: np.std(np.log(v)) if len(v) > 1 else np.nan).dropna()
        # calibrate from revisions only when at least 2 quarters show a
        # genuine refiling, and clamp: sheet-to-sheet noise on tiny mileage
        # (e.g. sub-1k-mile quarters) is not a documented revision and can
        # produce absurd tau (observed: 2.2). Bounds [0.02, 0.30].
        if len(rev) >= 2:
            tau = float(np.clip(rev.mean(), 0.02, 0.30))
            src = "revisions"
        else:
            tau, src = CFG["model"]["tau_default"], "default"
        taus.append(dict(operator=op, tau=tau, source=src))
    pd.DataFrame(taus).to_csv(OUT_TAU, index=False)
    print(vmt.groupby("operator")["vmt_reported"].sum().to_string())
    print(f"wrote {OUT_VMT.relative_to(ROOT)} and {OUT_TAU.relative_to(ROOT)}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "survey"
    survey() if mode == "survey" else extract()
