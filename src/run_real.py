"""Stage 5: assemble model inputs, fit, report (paper Sections 5-6).

Reads the processed outputs of the earlier stages:
  data/interim/crashes_by_cell.csv
  data/processed/vmt.csv, tau.csv, benchmark.csv

For each metro x severity cell it builds a BenchData, fits the Section 4
model, and writes:
  results/{metro}_{severity}_summary.csv    posterior summaries per operator
  results/{metro}_{severity}_idata.nc       full posterior (arviz)
  results/forest_{metro}.png                rate-ratio forest plot
  results/bounds_{metro}.csv                4.6 threshold bounds (sgo_any only)

Dedup ambiguity (P3) is handled by fitting both the conservative and liberal
crash counts and reporting both, an interval over deduplication rather than a
mixture; the full mixture is a v1 item.
"""

import sys
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from model import BenchData, build_model, fit, threshold_bounds

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RES = ROOT / "results"


def quarters_index() -> list[str]:
    qs = pd.period_range(CFG["study"]["start_quarter"],
                         CFG["study"]["end_quarter"], freq="Q")
    return [str(q) for q in qs]


def build_cell(metro: str, severity: str, dedup_mode: str,
               crashes: pd.DataFrame, vmt: pd.DataFrame,
               tau: pd.DataFrame, bench: pd.DataFrame,
               numerator_scope: str = "all"):
    # Third amended SGO (Jun 2025 onward) removed SV-only airbag/tow fields;
    # combined Any-Vehicle flags are not comparable, so flag-based severities
    # are fit on the SV-flag era only.
    SEVERITY_END = {"airbag_or_tow": "2025Q1",
                    "police_reportable_proxy": "2025Q1"}
    # numerator_scope="service" restricts crash counts to reports whose
    # narrative indicates passenger service: the IV-C robustness refit that
    # matches the numerator to the CPUC denominator's scope.
    qs = quarters_index()
    if severity in SEVERITY_END:
        qs = [q for q in qs if q <= SEVERITY_END[severity]]
    ops = sorted(set(vmt.operator) & set(crashes.operator))
    if not ops:
        return None, None
    I, T = len(ops), len(qs)
    C = np.zeros((I, T), dtype=int)
    M = np.full((I, T), np.nan)
    ncol = "n_conservative" if dedup_mode == "conservative" else "n_liberal"
    if numerator_scope == "service":
        ncol += "_svc"
    csub = crashes[(crashes.metro == metro) & (crashes.severity == severity)]
    vlatest = vmt[vmt.is_latest]
    for i, op in enumerate(ops):
        for t, q in enumerate(qs):
            c = csub[(csub.operator == op) & (csub.quarter == q)][ncol]
            C[i, t] = int(c.iloc[0]) if len(c) else 0
            m = vlatest[(vlatest.operator == op) & (vlatest.quarter == q)]
            if len(m):
                M[i, t] = float(m.vmt_reported.iloc[0])
    # NOTE: CPUC VMT is not metro-split in older templates. v0 assigns the
    # operator's statewide VMT to its primary metro and warns; refine with
    # trip-level tables where available.
    b = bench[(bench.metro == metro) & (bench.severity == severity)]
    if b.empty:
        return None, None
    b = b.iloc[0]
    tau_map = dict(zip(tau.operator, tau.tau))
    dmap = CFG["model"].get("delta_by_operator", {})
    d_mean = np.array([dmap.get(o, {}).get("mean", CFG["model"]["delta_prior_mean_log"]) for o in ops])
    d_sd = np.array([dmap.get(o, {}).get("sd", CFG["model"]["delta_prior_sd"]) for o in ops])
    d = BenchData(
        C=C, M_hat=M,
        tau_prior_sd=np.array([tau_map.get(o, CFG["model"]["tau_default"]) for o in ops])
                     * float(CFG["model"].get("tau_scale", 1.0)),
        delta_prior_mean=d_mean,
        delta_prior_sd=d_sd,
        H=int(b.H), V=float(b.V),
        r_alpha=float(b.r_alpha), r_beta=float(b.r_beta),
        overdispersed=bool(CFG["model"].get("overdispersed", False)),
    )
    return d, ops


def main(numerator_scope: str = "all") -> None:
    crashes = pd.read_csv(ROOT / "data/interim/crashes_by_cell.csv")
    vmt = pd.read_csv(ROOT / "data/processed/vmt.csv")
    tau = pd.read_csv(ROOT / "data/processed/tau.csv")
    bench = pd.read_csv(ROOT / "data/processed/benchmark.csv")
    RES.mkdir(exist_ok=True)
    mc = CFG["model"]

    for metro in CFG["study"]["metros"]:
        forest = []
        for severity in bench[bench.metro == metro].severity.unique():
            for mode in ("conservative", "liberal"):
                d, ops = build_cell(metro, severity, mode, crashes, vmt,
                                    tau, bench, numerator_scope)
                if d is None:
                    continue
                idata = fit(build_model(d), draws=mc["draws"], tune=mc["tune"],
                            chains=mc["chains"])
                rho = idata.posterior["rho"].values.reshape(-1, len(ops))
                summ = pd.DataFrame({
                    "operator": ops,
                    "rho_median": np.median(rho, 0),
                    "rho_lo": np.percentile(rho, 2.5, 0),
                    "rho_hi": np.percentile(rho, 97.5, 0),
                    "p_safer": (rho < 1).mean(0),
                    "severity": severity, "dedup": mode,
                })
                summ.to_csv(RES / f"{metro}_{severity}_{mode}_summary.csv",
                            index=False)
                try:
                    idata.to_netcdf(RES / f"{metro}_{severity}_{mode}_idata.nc")
                except (ValueError, ImportError) as e:
                    print(f"  (posterior not saved to netcdf: {e}; "
                          "pip install h5netcdf to keep full posteriors)")
                forest.append(summ)
                print(f"[{metro}/{severity}/{mode}]")
                print(summ.round(3).to_string(index=False))

        if forest:
            fdf = pd.concat(forest)
            fig, ax = plt.subplots(figsize=(7, 0.5 * len(fdf) + 1.5))
            y = np.arange(len(fdf))
            ax.hlines(y, fdf.rho_lo, fdf.rho_hi)
            ax.plot(fdf.rho_median, y, "o")
            ax.axvline(1.0, ls="--", lw=1)
            ax.set_yticks(y)
            ax.set_yticklabels(fdf.operator + " / " + fdf.severity + " / " + fdf.dedup,
                               fontsize=7)
            ax.set_xlabel("rate ratio rho (ADS / human), 95% CI, log scale")
            ax.set_xscale("log")
            fig.tight_layout()
            fig.savefig(RES / f"forest_{metro}.png", dpi=200)
            print(f"wrote results/forest_{metro}.png")


if __name__ == "__main__":
    import sys
    main("service" if "--service-only" in sys.argv else "all")
