"""Simulation study 2: frequentist calibration of the Bayesian intervals.

Repeats the three-operator synthetic experiment (large / parity / early
operator) N times with fresh randomness, refits the Section IV model each
time, and reports coverage of the 95% credible intervals for each operator's
rate ratio plus the human rate. Nominal coverage should be near 0.95;
under-coverage would indicate overconfidence, the exact failure the framework
exists to avoid.

Checkpoints each replication to results/coverage.csv so partial runs are
usable. Rerun to append more replications.
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model import BenchData, build_model, fit

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
OUT = RES / "coverage.csv"

N_REPS = 100
I, T = 3, 16
MU = 3.0e-6
LAM = np.array([0.6e-6, 3.0e-6, 4.5e-6])
RHO = LAM / MU
TAU = np.array([0.05, 0.10, 0.20])
DELTA = np.log(0.90)
V_H, R_H = 2.0e9, 0.88

start = 0
if OUT.exists():
    start = sum(1 for _ in open(OUT)) - 1
    print(f"resuming at replication {start}")
else:
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(
            ["rep"] + [f"cover_rho_{i}" for i in range(I)]
            + [f"width_rho_{i}" for i in range(I)] + ["cover_mu"])

for rep in range(start, N_REPS):
    rng = np.random.default_rng(1000 + rep)
    M = np.stack([
        np.full(T, 3.0e6), np.full(T, 1.5e5),
        np.concatenate([np.zeros(8), np.full(T - 8, 1.5e4)]),
    ]) * rng.lognormal(0, 0.05, (I, T))
    Mh = np.where(M > 0, np.exp(np.log(np.maximum(M, 1)) + DELTA
                  + rng.normal(0, TAU[:, None], (I, T))), np.nan)
    C = rng.poisson(LAM[:, None] * M)
    H = int(rng.poisson(MU * V_H * R_H))
    d = BenchData(C=C, M_hat=Mh, tau_prior_sd=TAU,
                  delta_prior_mean=np.full(I, np.log(0.9)),
                  delta_prior_sd=np.full(I, 0.10),
                  H=H, V=V_H, r_alpha=88.0, r_beta=12.0)
    idata = fit(build_model(d), draws=1000, tune=1000, chains=4, seed=rep)
    rho = idata.posterior["rho"].values.reshape(-1, I)
    mu = idata.posterior["mu_human"].values.ravel()
    lo, hi = np.percentile(rho, [2.5, 97.5], axis=0)
    mlo, mhi = np.percentile(mu, [2.5, 97.5])
    row = ([rep] + [int(lo[i] <= RHO[i] <= hi[i]) for i in range(I)]
           + [float(np.log(hi[i] / lo[i])) for i in range(I)]
           + [int(mlo <= MU <= mhi)])
    with open(OUT, "a", newline="") as f:
        csv.writer(f).writerow(row)
    print(f"rep {rep}: cover={row[1:1+I]} mu={row[-1]}")

rows = list(csv.DictReader(open(OUT)))
n = len(rows)
print(f"\n=== coverage over {n} replications (nominal 0.95) ===")
for i, name in enumerate(["large_op", "mid_op", "early_op"]):
    c = np.mean([int(r[f"cover_rho_{i}"]) for r in rows])
    w = np.mean([float(r[f"width_rho_{i}"]) for r in rows])
    print(f"{name:9s} rho coverage {c:.2f}   mean log CI width {w:.2f}")
print(f"human mu coverage {np.mean([int(r['cover_mu']) for r in rows]):.2f}")
