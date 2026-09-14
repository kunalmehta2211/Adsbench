"""Model validation on synthetic data with known ground truth.

Simulates a California-like scenario: three operators whose fleets differ by
orders of magnitude (a Waymo-scale operator, a mid-scale one, an early
deployer), 16 quarters, injury-level severity, noisy VMT filings including a
systematic passenger-service undercount. Then fits the Section 4 model and
checks that 95% credible intervals cover the true rates and rate ratios.

The printed table is the model-validation evidence for paper Section 5.
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import BenchData, build_model, fit, threshold_bounds

rng = np.random.default_rng(7)

# ---------------- ground truth ----------------
I, T = 3, 16
OPERATORS = ["large_op", "mid_op", "early_op"]

# true injury-level crash rates per mile (large op safer than humans,
# mid op at parity, early op slightly worse: a hard, interesting case)
MU_HUMAN_TRUE = 3.0e-6
LAMBDA_TRUE = np.array([0.6e-6, 3.0e-6, 4.5e-6])
RHO_TRUE = LAMBDA_TRUE / MU_HUMAN_TRUE  # [0.2, 1.0, 1.5]

# true quarterly VMT: 3M, 150k, 15k miles per quarter (orders of magnitude apart)
M_TRUE = np.stack([
    np.full(T, 3.0e6) * rng.lognormal(0, 0.05, T),
    np.full(T, 1.5e5) * rng.lognormal(0, 0.05, T),
    np.concatenate([np.zeros(8), np.full(T - 8, 1.5e4)]) * rng.lognormal(0, 0.05, T),
])

# reported VMT: systematic 10% undercount (passenger-service definition),
# noise calibrated to documented CPUC revision magnitudes
DELTA_TRUE = np.log(0.90)
TAU_TRUE = np.array([0.05, 0.10, 0.20])  # early op has messier reporting
M_hat = np.where(
    M_TRUE > 0,
    np.exp(np.log(np.maximum(M_TRUE, 1)) + DELTA_TRUE
           + rng.normal(0, TAU_TRUE[:, None], (I, T))),
    np.nan,
)

# crashes from TRUE exposure
C = rng.poisson(LAMBDA_TRUE[:, None] * M_TRUE)

# human benchmark: 2e9 miles, reporting probability 0.88 at injury level
V_HUMAN, R_TRUE = 2.0e9, 0.88
H = int(rng.poisson(MU_HUMAN_TRUE * V_HUMAN * R_TRUE))

print("=== synthetic ground truth ===")
print(f"human rate {MU_HUMAN_TRUE:.2e}/mi, observed human crashes H={H}")
for i, name in enumerate(OPERATORS):
    print(f"{name:9s} lambda={LAMBDA_TRUE[i]:.2e} rho={RHO_TRUE[i]:.2f} "
          f"total true miles={M_TRUE[i].sum():.2e} total crashes={C[i].sum()}")

# ---------------- fit ----------------
data = BenchData(
    C=C,
    M_hat=M_hat,
    tau_prior_sd=np.array([0.05, 0.10, 0.20]),
    delta_prior_mean=np.full(I, np.log(0.9)),
    delta_prior_sd=np.full(I, 0.10),
    H=H,
    V=V_HUMAN,
    r_alpha=88.0,  # Beta(88,12): mean .88, sd ~.032, from Blincoe-style corrections
    r_beta=12.0,
)
idata = fit(build_model(data), draws=1000, tune=1000, chains=4)

# ---------------- recovery report ----------------
summ = az.summary(idata, var_names=["mu_human", "lambda", "rho", "sigma_op"])
print("\n=== posterior summary ===")
print(summ)

post = idata.posterior
covered = []
for i, name in enumerate(OPERATORS):
    lam_s = post["lambda"].values[..., i].ravel()
    rho_s = post["rho"].values[..., i].ravel()
    lam_ci = np.percentile(lam_s, [2.5, 97.5])
    rho_ci = np.percentile(rho_s, [2.5, 97.5])
    cov_l = lam_ci[0] <= LAMBDA_TRUE[i] <= lam_ci[1]
    cov_r = rho_ci[0] <= RHO_TRUE[i] <= rho_ci[1]
    covered += [cov_l, cov_r]
    p_safer = float((rho_s < 1).mean())
    print(f"\n{name}: true rho={RHO_TRUE[i]:.2f}  "
          f"posterior rho 95% CI [{rho_ci[0]:.2f}, {rho_ci[1]:.2f}]  "
          f"covered={cov_r}  P(safer than human)={p_safer:.2f}")

mu_ci = np.percentile(post["mu_human"].values.ravel(), [2.5, 97.5])
print(f"\nhuman rate: true {MU_HUMAN_TRUE:.2e}, "
      f"95% CI [{mu_ci[0]:.2e}, {mu_ci[1]:.2e}], "
      f"covered={mu_ci[0] <= MU_HUMAN_TRUE <= mu_ci[1]}")

# 4.6 bounds demo: suppose severity flags cap pi at [.05,.30] per operator
bounds = threshold_bounds(idata,
                          pi_min=np.array([0.05, 0.05, 0.05]),
                          pi_max=np.array([0.30, 0.30, 0.30]))
print("\n=== threshold-mismatch bounds (illustrative pi in [.05,.30]) ===")
for b, name in zip(bounds, OPERATORS):
    lo, hi = b["rho_lower_bound_ci"], b["rho_upper_bound_ci"]
    print(f"{name}: identified interval for rho_police, medians "
          f"[{lo[1]:.2f}, {hi[1]:.2f}]")

n_ok = sum(covered)
print(f"\nRECOVERY: {n_ok}/{len(covered)} rate and ratio CIs cover truth")
print("max r_hat:", float(summ["r_hat"].max()))
