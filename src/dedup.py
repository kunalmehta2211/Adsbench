"""Stage 2b: deduplicate SGO reports into physical crashes (paper 4.5, P3).

Two layers:
  1. Version collapse: keep the highest report version per report_id.
  2. Crash linkage: reports from different entities (or duplicate filings)
     describing the same physical crash are clustered on
     (operator, incident_date, city). Same-day same-city pairs from one
     operator are genuinely ambiguous when narratives are redacted, so
     instead of forcing a choice we emit BOTH a conservative count (cluster
     everything, fewer crashes) and a liberal count (nothing clustered, more
     crashes). The model mixes over these, making redaction cost visible.

Output:
  data/interim/crashes_by_cell.csv with columns
    operator, metro, quarter, severity, n_conservative, n_liberal
  where severity in config severity_levels. 'police_reportable_proxy' is the
  flag-based cap used for the pi bounds (injury OR airbag OR towed), NOT a
  claim that these are exactly the police-reportable events.
"""

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
SRC = ROOT / "data" / "interim" / "sgo_reports.csv"
OUT = ROOT / "data" / "interim" / "crashes_by_cell.csv"


def main() -> None:
    df = pd.read_csv(SRC, parse_dates=["incident_date"])

    # layer 1: latest version per report
    df = df.sort_values("version").groupby("report_id", as_index=False).last()
    print(f"after version collapse: {len(df)} reports")

    # layer 2: cluster into physical crashes.
    # Primary key: NHTSA's own Same Incident ID (field 55), generated so
    # that "data users [can] determine whether other reports regarding the
    # same incident were filed by another Reporting Entity". Where absent,
    # fall back to VIN + date + hour; rows lacking all identifiers stay
    # distinct in the liberal count.
    sid = df.get("same_incident_id", pd.Series(index=df.index, dtype=str))
    sid = sid.astype(str).replace({"nan": "", "None": ""}).fillna("")
    df["hour"] = (df.get("incident_time", pd.Series(dtype=str))
                  .astype(str).str.extract(r"^(\d{1,2})")[0].fillna("NA"))
    vin = df.get("vin", pd.Series(index=df.index, dtype=str))
    vin = vin.astype(str).replace({"nan": "", "None": ""}).fillna("")
    fallback = ("V_" + vin + "_" + df["incident_date"].astype(str)
                + "_" + df["hour"])
    fallback[vin.eq("")] = "RID_" + df.loc[vin.eq(""), "report_id"].astype(str)
    df["crash_key"] = sid.where(sid.ne(""), fallback)
    n_sid = int(sid.ne("").sum())
    key = ["operator", "metro", "crash_key"]
    df["cluster_size"] = df.groupby(key)["report_id"].transform("size")
    print(f"reports with NHTSA Same Incident ID: {n_sid}/{len(df)}")
    print(f"reports merged into multi-report crashes: "
          f"{int((df['cluster_size'] > 1).sum())}")

    # severity flags per report
    df["sev_any_injury"] = df["any_injury"].astype(bool)
    df["sev_airbag_or_tow"] = df["airbag"].astype(bool) | df["towed"].astype(bool) | df["sev_any_injury"]
    df["sev_serious_plus"] = df["serious_plus"].astype(bool)
    df["sev_police_proxy"] = df["sev_airbag_or_tow"]  # flag-based cap for pi
    df["sev_sgo_any"] = True

    sev_cols = {
        "sgo_any": "sev_sgo_any",
        "police_reportable_proxy": "sev_police_proxy",
        "any_injury": "sev_any_injury",
        "airbag_or_tow": "sev_airbag_or_tow",
        "serious_plus": "sev_serious_plus",
    }

    rows = []
    svc = (df["service_context"].astype(bool)
           if "service_context" in df else pd.Series(False, index=df.index))
    for (op, metro, q), g in df.groupby(["operator", "metro", "quarter"]):
        clusters = g.groupby(key)
        gs = g[svc.loc[g.index]]
        clusters_s = gs.groupby(key) if len(gs) else None
        for sev_name, col in sev_cols.items():
            # liberal: every report is its own crash
            n_lib = int(g[col].sum())
            # conservative: a cluster counts once, at its max severity
            n_con = int(clusters[col].max().sum())
            rows.append(dict(
                operator=op, metro=metro, quarter=q, severity=sev_name,
                n_conservative=n_con, n_liberal=n_lib,
                n_conservative_svc=(int(clusters_s[col].max().sum())
                                    if clusters_s is not None else 0),
                n_liberal_svc=int(gs[col].sum()) if len(gs) else 0))
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    tot = out[out.severity == "sgo_any"][["n_conservative", "n_liberal"]].sum()
    print(f"crashes (sgo_any): conservative={tot.n_conservative}, liberal={tot.n_liberal}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print("Spot-check tip: sample 10 clusters and compare against DMV OL316 PDFs.")


if __name__ == "__main__":
    main()
