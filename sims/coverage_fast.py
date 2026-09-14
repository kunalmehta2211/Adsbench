"""Coverage study, fast harness: build the PyMC model once with pm.Data
containers and swap observations per replication. Identical model semantics
to sims/coverage_study.py; appends to the same checkpoint CSV, so earlier
replications remain valid. Settings: 1000 draws, 1000 tune, 4 chains.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pymc as pm

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "coverage.csv"
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100

I, T = 3, 16
MU, LAM = 3.0e-6, np.array([0.6e-6, 3.0e-6, 4.5e-6])
RHO = LAM / MU
TAU = np.array([0.05, 0.10, 0.20])
V_H, R_H = 2.0e9, 0.88


def gen(rep):
    rng = np.random.default_rng(1000 + rep)
    M = np.stack([np.full(T, 3.0e6), np.full(T, 1.5e5),
                  np.concatenate([np.zeros(8), np.full(8, 1.5e4)])
                  ]) * rng.lognormal(0, 0.05, (I, T))
    Mh = np.where(M > 0, np.exp(np.log(np.maximum(M, 1)) + np.log(0.90)
                  + rng.normal(0, TAU[:, None], (I, T))), np.nan)
    C = rng.poisson(LAM[:, None] * M)
    H = int(rng.poisson(MU * V_H * R_H))
    return C, Mh, H


C0, Mh0, H0 = gen(0)
mask = ~np.isnan(Mh0)  # identical across reps by construction
op_idx = np.nonzero(mask)[0]
n_obs = int(mask.sum())

with pm.Model() as model:
    C_d = pm.Data("C_d", C0[mask])
    Ml_d = pm.Data("Ml_d", np.log(Mh0[mask]))
    H_d = pm.Data("H_d", np.array(H0, dtype="int64"))
    mu = pm.LogNormal("mu_human", mu=np.log(3e-6), sigma=1.5)
    r = pm.Beta("r_report", alpha=88.0, beta=12.0)
    pm.Poisson("H_obs", mu=mu * V_H * r, observed=H_d)
    sigma_op = pm.HalfNormal("sigma_op", sigma=1.0)
    alpha = pm.Normal("alpha", mu=pm.math.log(mu), sigma=1.0)
    beta_raw = pm.Normal("beta_raw", 0.0, 1.0, shape=I)
    log_lam = alpha + beta_raw * sigma_op
    pm.Deterministic("rho", pm.math.exp(log_lam) / mu)
    delta = pm.Normal("delta", mu=np.log(0.9), sigma=0.10, shape=I)
    log_M = pm.Normal("log_M", mu=Ml_d, sigma=2.0, shape=n_obs)
    pm.Normal("Mh_obs", mu=log_M + delta[op_idx], sigma=TAU[op_idx],
              observed=Ml_d)
    pm.Poisson("C_obs", mu=pm.math.exp(log_lam[op_idx] + log_M),
               observed=C_d)

start = sum(1 for _ in open(OUT)) - 1 if OUT.exists() else 0
if start == 0:
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(
            ["rep"] + [f"cover_rho_{i}" for i in range(I)]
            + [f"width_rho_{i}" for i in range(I)] + ["cover_mu"])
print(f"resuming at rep {start}", flush=True)

for rep in range(start, N_REPS):
    C, Mh, H = gen(rep)
    with model:
        pm.set_data({"C_d": C[mask], "Ml_d": np.log(Mh[mask]),
                     "H_d": np.array(H, dtype="int64")})
        idata = pm.sample(draws=1000, tune=1000, chains=4, cores=1,
                          random_seed=rep, target_accept=0.9,
                          progressbar=False,
                          compute_convergence_checks=False)
    rho = idata.posterior["rho"].values.reshape(-1, I)
    muv = idata.posterior["mu_human"].values.ravel()
    lo, hi = np.percentile(rho, [2.5, 97.5], axis=0)
    mlo, mhi = np.percentile(muv, [2.5, 97.5])
    row = ([rep] + [int(lo[i] <= RHO[i] <= hi[i]) for i in range(I)]
           + [float(np.log(hi[i] / lo[i])) for i in range(I)]
           + [int(mlo <= MU <= mhi)])
    with open(OUT, "a", newline="") as f:
        csv.writer(f).writerow(row)
    print(f"rep {rep}: cover={row[1:1+I]} mu={row[-1]}", flush=True)
