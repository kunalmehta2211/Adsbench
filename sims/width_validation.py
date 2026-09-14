"""Simulation study 1: does the fitted model obey Proposition 1?

Prop 1 claims Var(log lambda | data) ~ 1/C + tau_eff^2 with
tau_eff^2 = tau^2 + sigma_delta^2. We simulate a single operator across a
grid of expected crash counts and tau values, fit the actual PyMC model
(not the approximation), and compare empirical posterior sd of log lambda
against the theoretical sqrt(1/C + tau_eff^2).

Outputs:
  results/width_validation.csv
  results/fig_width.pdf   (paper figure: posterior width vs C, by tau)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

rng = np.random.default_rng(3)

LAMBDA_TRUE = 3.0e-6
SIGMA_DELTA = 0.05
TAUS = [0.02, 0.10, 0.25]
C_TARGETS = [3, 10, 30, 100, 300, 1000]
T = 8


def fit_single(C_obs, M_hat, tau):
    """Single-operator reduction of the Section IV model (no pooling,
    benchmark not needed for lambda width)."""
    with pm.Model():
        log_lam = pm.Flat("log_lambda")
        delta = pm.Normal("delta", mu=0.0, sigma=SIGMA_DELTA)
        log_M = pm.Normal("log_M", mu=np.log(M_hat), sigma=2.0, shape=T)
        pm.Normal("M_hat_obs", mu=log_M + delta, sigma=tau,
                  observed=np.log(M_hat))
        pm.Poisson("C_obs", mu=pm.math.exp(log_lam + log_M), observed=C_obs)
        idata = pm.sample(draws=800, tune=800, chains=2, cores=1,
                          random_seed=17, target_accept=0.92,
                          progressbar=False)
    return float(idata.posterior["log_lambda"].std())


rows = []
for tau in TAUS:
    for C_target in C_TARGETS:
        M_true = np.full(T, C_target / (LAMBDA_TRUE * T))
        M_hat = np.exp(np.log(M_true) + rng.normal(0, tau, T))
        C_obs = rng.poisson(LAMBDA_TRUE * M_true)
        C_tot = int(C_obs.sum())
        if C_tot == 0:
            C_obs[0] = 1; C_tot = 1
        sd_emp = fit_single(C_obs, M_hat, tau)
        sd_theory = np.sqrt(1.0 / C_tot + tau**2 / T + SIGMA_DELTA**2)
        rows.append((tau, C_tot, sd_emp, sd_theory))
        print(f"tau={tau:.2f} C={C_tot:5d}  empirical={sd_emp:.4f} "
              f"theory={sd_theory:.4f}  ratio={sd_emp/sd_theory:.3f}")

import csv
with open(RES / "width_validation.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tau", "C", "posterior_sd_empirical", "prop1_theory"])
    w.writerows(rows)

# paper figure
fig, ax = plt.subplots(figsize=(4.2, 3.1))
markers = {0.02: "o", 0.10: "s", 0.25: "^"}
colors = {0.02: "#1b6ca8", 0.10: "#c05a2e", 0.25: "#3a7d44"}
Cgrid = np.logspace(np.log10(2), np.log10(1500), 200)
for tau in TAUS:
    sub = [(C, e) for t_, C, e, th in rows if t_ == tau]
    ax.plot(Cgrid, np.sqrt(1/Cgrid + tau**2/T + SIGMA_DELTA**2), "-",
            lw=1.1, color=colors[tau], alpha=0.8)
    ax.plot([c for c, _ in sub], [e for _, e in sub], markers[tau],
            ms=5, color=colors[tau],
            label=rf"$\tau={tau}$")
    
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("total observed crashes $C$")
ax.set_ylabel(r"posterior sd of $\log\lambda$")
ax.legend(fontsize=8, title="lines: Prop. 1\nmarkers: MCMC", title_fontsize=7)
fig.tight_layout()
fig.savefig(RES / "fig_width.pdf")
ratios = [e/th for _, _, e, th in rows]
print(f"\nmean empirical/theory ratio: {np.mean(ratios):.3f} "
      f"(min {min(ratios):.3f}, max {max(ratios):.3f})")
