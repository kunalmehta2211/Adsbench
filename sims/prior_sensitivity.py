"""Robustness item 1+2: prior sensitivity table and prior predictive checks.

Sensitivity (paper IV-C/IV-H promise): refit the standard three-operator
synthetic scenario under systematic prior perturbations and report how each
operator's rate ratio posterior moves:
  baseline        calibrated tau, delta ~ N(log .9, .10), parity-centered alpha
  tau_half        all tau_i halved
  tau_double      all tau_i doubled
  delta_zero      delta prior centered at 0 (no definitional undercount)
  delta_wide      delta prior sd doubled
  alpha_diffuse   wide fixed-center prior instead of parity-centered

Prior predictive (paper IV-H promise): draw from the priors alone and check
that the implied distributions of reported VMT and crash counts comfortably
cover the observed data (reported as the quantile of each observation within
its prior predictive distribution; values hugging 0 or 1 indicate
prior-data conflict).

Outputs: results/sensitivity_table.csv, results/prior_predictive.csv
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pymc as pm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model import BenchData, build_model, fit

RES = ROOT / "results"; RES.mkdir(exist_ok=True)
OPS = ["large_op", "mid_op", "early_op"]

# ---- the standard synthetic scenario (same seed as synthetic.py) ----
rng = np.random.default_rng(7)
I, T = 3, 16
MU, LAM = 3.0e-6, np.array([0.6e-6, 3.0e-6, 4.5e-6])
RHO = LAM / MU
M = np.stack([np.full(T, 3.0e6), np.full(T, 1.5e5),
              np.concatenate([np.zeros(8), np.full(8, 1.5e4)])
              ]) * rng.lognormal(0, 0.05, (I, T))
TAU = np.array([0.05, 0.10, 0.20])
Mh = np.where(M > 0, np.exp(np.log(np.maximum(M, 1)) + np.log(0.90)
              + rng.normal(0, TAU[:, None], (I, T))), np.nan)
C = rng.poisson(LAM[:, None] * M)
H = int(rng.poisson(MU * 2.0e9 * 0.88))


def data(tau_scale=1.0, delta_mean=np.log(0.9), delta_sd=0.10):
    return BenchData(C=C, M_hat=Mh, tau_prior_sd=TAU * tau_scale,
                     delta_prior_mean=np.full(I, delta_mean),
                     delta_prior_sd=np.full(I, delta_sd),
                     H=H, V=2.0e9, r_alpha=88.0, r_beta=12.0)


VARIANTS = {
    "baseline":      (data(), "parity"),
    "tau_half":      (data(tau_scale=0.5), "parity"),
    "tau_double":    (data(tau_scale=2.0), "parity"),
    "delta_zero":    (data(delta_mean=0.0), "parity"),
    "delta_wide":    (data(delta_sd=0.20), "parity"),
    "alpha_diffuse": (data(), "diffuse"),
}

rows = []
for name, (d, ap) in VARIANTS.items():
    idata = fit(build_model(d, alpha_prior=ap), draws=1000, tune=1000, chains=2)
    rho = idata.posterior["rho"].values.reshape(-1, I)
    for i, op in enumerate(OPS):
        lo, med, hi = np.percentile(rho[:, i], [2.5, 50, 97.5])
        rows.append(dict(variant=name, operator=op, rho_true=RHO[i],
                         rho_lo=lo, rho_med=med, rho_hi=hi,
                         covered=int(lo <= RHO[i] <= hi)))
    print(f"{name:14s} " + "  ".join(
        f"{op}:{r['rho_med']:.2f}[{r['rho_lo']:.2f},{r['rho_hi']:.2f}]"
        for op, r in zip(OPS, rows[-3:])))

with open(RES / "sensitivity_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

# max shift of medians vs baseline, per operator
base = {r["operator"]: r["rho_med"] for r in rows if r["variant"] == "baseline"}
shifts = {}
for r in rows:
    if r["variant"] == "baseline":
        continue
    s = abs(np.log(r["rho_med"] / base[r["operator"]]))
    shifts[r["operator"]] = max(shifts.get(r["operator"], 0), s)
print("\nmax |log median shift| vs baseline:",
      {k: f"{v:.2f}" for k, v in shifts.items()})
print(f"coverage across variants: "
      f"{sum(r['covered'] for r in rows)}/{len(rows)}")

# ---- prior predictive checks (baseline priors) ----
d, _ = VARIANTS["baseline"]
with build_model(d):
    prior = pm.sample_prior_predictive(draws=2000, random_seed=5)
pp = prior.prior_predictive
obs_mask = ~np.isnan(Mh)
q_rows = []
Cpp = pp["C_obs"].values.reshape(-1, obs_mask.sum())
Mpp = pp["M_hat_obs"].values.reshape(-1, obs_mask.sum())
c_obs, m_obs = C[obs_mask], np.log(Mh[obs_mask])
op_idx = np.nonzero(obs_mask)[0]
for i, op in enumerate(OPS):
    sel = op_idx == i
    qc = float(np.mean(Cpp[:, sel].sum(1) <= c_obs[sel].sum()))
    qm = float(np.mean(Mpp[:, sel].mean(1) <= m_obs[sel].mean()))
    q_rows.append(dict(operator=op, q_total_crashes=qc, q_mean_log_vmt=qm))
    print(f"prior predictive {op}: crash-total quantile {qc:.2f}, "
          f"log-VMT quantile {qm:.2f}")
with open(RES / "prior_predictive.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=q_rows[0].keys())
    w.writeheader(); w.writerows(q_rows)
print("(quantiles in (0.05, 0.95) indicate no prior-data conflict)")
