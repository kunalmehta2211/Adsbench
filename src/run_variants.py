"""Run the full sensitivity suite behind the paper's Section VI robustness
paragraph. One command, unattended:

    python src/run_variants.py

Variants (each a full refit of every severity cell, conservative dedup):
  main         config as-is (this is the paper-of-record run)
  tau_half     all tau priors x0.5
  tau_double   all tau priors x2
  delta_zero   delta priors centered at 0 (no scope correction)
  delta_wide   delta prior sds x2
  overdisp     period-level rate drift enabled (Sec IV-B)

Outputs:
  results/variants/<name>/..._summary.csv     per-variant summaries
  results/variants_comparison.csv             medians side by side
  results/variants_report.txt                 max shift per headline cell

Expected runtime at full settings (2000/2000, 4 chains): several hours.
Progress prints per fit; safe to leave overnight. Rerun skips finished
variants.
"""

import copy
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config.yaml"
RES = ROOT / "results"
VAR = RES / "variants"
VAR.mkdir(parents=True, exist_ok=True)

BASE = yaml.safe_load(CFG_PATH.read_text())


def make_cfg(name: str) -> dict:
    c = copy.deepcopy(BASE)
    m = c["model"]
    if name == "tau_half":
        m["tau_scale"] = 0.5
    elif name == "tau_double":
        m["tau_scale"] = 2.0
    elif name == "delta_zero":
        for op in m.get("delta_by_operator", {}):
            m["delta_by_operator"][op]["mean"] = 0.0
        m["delta_prior_mean_log"] = 0.0
    elif name == "delta_wide":
        for op in m.get("delta_by_operator", {}):
            m["delta_by_operator"][op]["sd"] *= 2
        m["delta_prior_sd"] *= 2
    elif name == "overdisp":
        m["overdispersed"] = True
    return c


VARIANTS = ["main", "tau_half", "tau_double", "delta_zero", "delta_wide",
            "overdisp"]


def run_variant(name: str) -> None:
    outdir = VAR / name
    if (outdir / "DONE").exists():
        print(f"[{name}] already done, skipping")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] running ...")
    backup = CFG_PATH.read_text()
    try:
        yaml.safe_dump(make_cfg(name), CFG_PATH.open("w"), sort_keys=False)
        r = subprocess.run([sys.executable, str(ROOT / "src" / "run_real.py")],
                           capture_output=True, text=True)
        (outdir / "run.log").write_text(r.stdout[-8000:] + "\n" + r.stderr[-4000:])
        if r.returncode != 0:
            print(f"[{name}] FAILED, see {outdir/'run.log'}")
            return
        copied = 0
        for f in RES.glob("california_*_summary.csv"):
            shutil.copy(f, outdir / f.name); copied += 1
        if copied >= 4:
            (outdir / "DONE").write_text("ok")
            print(f"[{name}] done ({copied} tables)")
        else:
            print(f"[{name}] INCOMPLETE ({copied} tables), not marked done")
    finally:
        CFG_PATH.write_text(backup)


def report() -> None:
    rows = []
    for name in VARIANTS:
        for f in (VAR / name).glob("california_*_conservative_summary.csv"):
            d = pd.read_csv(f)
            d["variant"] = name
            rows.append(d)
    if not rows:
        return
    all_ = pd.concat(rows)
    piv = all_.pivot_table(index=["severity", "operator"],
                           columns="variant", values="rho_median")
    piv.to_csv(RES / "variants_comparison.csv")
    lines = ["Max |log-shift| of rho median vs main, per cell:"]
    for (sev, op), row in piv.iterrows():
        if "main" not in row or pd.isna(row["main"]):
            continue
        import numpy as np
        shifts = {v: abs(np.log(row[v] / row["main"]))
                  for v in row.index if v != "main" and not pd.isna(row[v])}
        if shifts:
            worst = max(shifts, key=shifts.get)
            lines.append(f"  {sev:22s} {op:7s} worst={worst:11s} "
                         f"|dlog rho|={shifts[worst]:.3f}")
    (RES / "variants_report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    todo = sys.argv[1:] or VARIANTS
    for v in todo:
        run_variant(v)
    report()
    print(f"\nSend results/variants_comparison.csv and variants_report.txt")
