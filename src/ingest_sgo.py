"""Stage 2a: parse the NHTSA SGO ADS incident CSV into a tidy crash table.

Defensive by design: the exact column headers in the public CSV have changed
across SGO amendments, so this script (a) fuzzy-matches the columns it needs,
(b) prints a schema report of what it found, and (c) refuses to continue if a
required concept is missing, telling you which header to map in COLUMN_HINTS.

Output: data/interim/sgo_reports.csv, one row per (report, version), with
normalized fields:
  report_id, version, operator, incident_date, quarter, city, metro, state,
  injury_severity, any_injury, airbag, towed, narrative_len
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
SRC_DIR = ROOT / "data" / "raw" / "sgo"
OUT = ROOT / "data" / "interim" / "sgo_reports.csv"

# concept -> list of header fragments to fuzzy-match (case/space-insensitive).
# If the schema report shows a miss, add the real header fragment here.
COLUMN_HINTS = {
    "report_id": ["report id"],
    "version": ["report version"],
    "entity": ["reporting entity"],
    "incident_date": ["incident date"],
    "city": ["city"],
    "state": ["state"],
    "injury_severity": ["highest injury severity"],
    "airbag_sv": ["sv any air bags", "sv air bags", "any air bags"],
    "airbag_cp": ["cp any air bags", "cp air bags"],
    "vin": ["vin"],
    "time": ["incident time"],
    "driver_type": ["driver / operator type", "operator type"],
    "same_incident": ["same incident id"],
    "same_vehicle": ["same vehicle id"],
    "towed": ["sv was vehicle towed", "was any vehicle towed", "towed"],
    "narrative": ["narrative"],
    "ads_engaged": ["automation system engaged"],
}
REQUIRED = ["report_id", "version", "entity", "incident_date", "city", "state",
            "injury_severity"]


def norm(s: str) -> str:
    return " ".join(str(s).lower().replace("?", " ").split())


def match_columns(cols) -> dict:
    """Prefer an exact normalized match, fall back to substring, so that
    e.g. 'State' is never confused with 'Source - State'."""
    found = {}
    ncols = {norm(c): c for c in cols}
    for concept, hints in COLUMN_HINTS.items():
        for h in hints:
            if h in ncols:
                found[concept] = ncols[h]
                break
            hit = next((orig for n, orig in ncols.items() if h in n), None)
            if hit:
                found[concept] = hit
                break
    return found


def yes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["yes", "y", "true"])


def ingest_one(SRC: Path) -> "pd.DataFrame":
    print(f"\n### {SRC.name}")
    df = pd.read_csv(SRC, dtype=str, low_memory=False)
    cols = match_columns(df.columns)

    print("=== schema report ===")
    for concept in COLUMN_HINTS:
        print(f"  {concept:16s} -> {cols.get(concept, 'MISSING')}")
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        sys.exit(
            f"Required concepts not found: {missing}. Open the CSV, find the "
            "real header, add a fragment of it to COLUMN_HINTS, rerun."
        )

    out = pd.DataFrame({
        "report_id": df[cols["report_id"]].str.strip(),
        "version": pd.to_numeric(df[cols["version"]], errors="coerce"),
        "entity": df[cols["entity"]].str.strip(),
        "city": df[cols["city"]].str.strip().str.title(),
        "state": df[cols["state"]].str.strip().str.upper(),
        "injury_severity": df[cols["injury_severity"]].str.strip(),
    })
    raw_date = df[cols["incident_date"]].astype(str).str.strip()
    # archive files use MMM-YYYY (month precision); try those formats
    # FIRST, since pandas mixed-format inference mangles them.
    out["incident_date"] = pd.to_datetime(raw_date, format="%b-%Y",
                                          errors="coerce")
    mask = out["incident_date"].isna()
    if mask.any():
        out.loc[mask, "incident_date"] = pd.to_datetime(
            raw_date[mask], format="%b-%y", errors="coerce")
    mask = out["incident_date"].isna()
    if mask.any():
        out.loc[mask, "incident_date"] = pd.to_datetime(
            raw_date[mask], errors="coerce", format="mixed")
    out["airbag"] = (
        yes(df[cols["airbag_sv"]]) if "airbag_sv" in cols else False
    ) | (yes(df[cols["airbag_cp"]]) if "airbag_cp" in cols else False)
    out["towed"] = yes(df[cols["towed"]]) if "towed" in cols else False
    out["narrative_len"] = (
        df[cols["narrative"]].fillna("").str.len() if "narrative" in cols else 0
    )
    out["vin"] = df[cols["vin"]].str.strip().str.upper() if "vin" in cols else None
    out["incident_time"] = df[cols["time"]].str.strip() if "time" in cols else None
    out["driver_type"] = (df[cols["driver_type"]].str.strip()
                          if "driver_type" in cols else None)
    out["same_incident_id"] = (df[cols["same_incident"]].str.strip()
                               if "same_incident" in cols else None)
    out["same_vehicle_id"] = (df[cols["same_vehicle"]].str.strip()
                              if "same_vehicle" in cols else None)
    # passenger-service context flag for the numerator-scope robustness refit
    # (paper IV-C). Conservative by construction: redacted or silent
    # narratives count as NOT service, so the filtered numerator is a lower
    # bound on service crashes. Refine keywords against real narratives.
    if "narrative" in cols:
        nar = df[cols["narrative"]].fillna("").str.lower()
        out["service_context"] = nar.str.contains(
            "passenger|rider|trip|pick-up|pickup|drop-off|dropoff|fare",
            regex=True)
    else:
        out["service_context"] = False
    if "ads_engaged" in cols:
        engaged = df[cols["ads_engaged"]].astype(str).str.lower()
        out = out[~engaged.str.contains("not engaged", na=False)]

    # rider-only filter: CPUC exposure covers passenger service, so crashes
    # with an in-vehicle safety driver (testing) must be excluded or the
    # numerator and denominator describe different fleets. First order for
    # operators whose testing mileage dwarfs service mileage (e.g. Zoox).
    if CFG["study"].get("rider_only", True) and out["driver_type"].notna().any():
        dt = out["driver_type"].astype(str).str.lower()
        n_before = len(out)
        out = out[~dt.str.contains("in-vehicle", na=False)]
        print(f"rider-only filter: {n_before} -> {len(out)} "
              "(dropped reports with an in-vehicle driver)")
    elif CFG["study"].get("rider_only", True):
        print("WARNING: driver/operator type column not found; rider-only "
              "filter NOT applied. Check COLUMN_HINTS against the CSV.")

    # operator mapping; print unmapped entities so config can be extended
    ent2op = {norm(e): op for op, ents in CFG["operators"].items() for e in ents}
    out["operator"] = out["entity"].map(lambda e: ent2op.get(norm(e)))
    unmapped = sorted(out.loc[out["operator"].isna(), "entity"].dropna().unique())
    if unmapped:
        print(f"NOTE: unmapped reporting entities (add to config if in scope): {unmapped}")

    # severity normalization, robust to the third-amendment label change:
    # archive era: "No Injuries Reported", "Minor", "Moderate", "Serious",
    #   "Fatality", "Unknown"
    # current era: "Property Damage. No Injured Reported",
    #   "No Injured Reported", "Minor W/O Hospitalization",
    #   "Minor W/ Hospitalization", "Moderate W/O Hospitalization",
    #   "Moderate W/ Hospitalization", "Serious", "Fatality", "Unknown"
    sev = out["injury_severity"].str.lower().fillna("")
    non_injury = (sev.eq("") | sev.str.contains("no injur", na=False)
                  | sev.isin(["none", "unknown"])
                  | sev.str.contains("property damage", na=False))
    out["any_injury"] = ~non_injury
    # KSI-aligned: Serious (hospitalization-required, archive) or Fatality.
    # Current-era "Minor/Moderate W/ Hospitalization" maps closer to ER
    # treatment than to KSI-severe and is excluded; noted as a limitation.
    out["serious_plus"] = sev.str.contains("serious|fatal", na=False)

    # metro assignment + study filters
    city2metro = {c.title(): m for m, spec in CFG["study"]["metros"].items()
                  for c in spec["cities"]}
    out["metro"] = out["city"].map(city2metro)
    n0 = len(out)
    out = out[(out["state"] == CFG["study"]["state"]) & out["operator"].notna()
              & out["metro"].notna() & out["incident_date"].notna()]
    out["quarter"] = out["incident_date"].dt.to_period("Q").astype(str)
    out = out[(out["quarter"] >= CFG["study"]["start_quarter"])
              & (out["quarter"] <= CFG["study"]["end_quarter"])]

    print(f"  raw report rows: {n0} | in scope: {len(out)}")
    return out


def main() -> None:
    files = sorted(SRC_DIR.glob("*.csv"))
    if not files:
        sys.exit(f"No CSVs in {SRC_DIR}. Run src/acquire.py first.")
    parts = [ingest_one(f) for f in files]
    out = pd.concat(parts, ignore_index=True)
    # a report can appear in both archive and current files; keep one row
    # per (report_id, version), preferring the later (current) file
    n0 = len(out)
    out = out.drop_duplicates(subset=["report_id", "version"], keep="last")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\n=== combined ingest summary ===")
    print(f"  rows: {n0} -> {len(out)} after cross-file report dedup")
    print(out.groupby(["operator", "metro"]).size().to_string())
    print(f"  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
