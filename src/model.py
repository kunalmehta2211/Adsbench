"""Hierarchical Bayesian crash-rate benchmarking model (paper Section 4).

Implements, for one geographic stratum and one severity level:
  4.2 Poisson crash counts given latent true exposure
  4.3 lognormal measurement error on reported VMT (problem P1)
  4.4 human benchmark with uncertain reporting probability (problem P5)
  4.6 threshold-mismatch bounds, computed on posterior samples (problem P2)
  4.7 partial pooling of operator log-rates, centered on human parity (P4)

v1 roadmap (flagged, not yet implemented):
  - monotone binomial thinning chain across severity levels (4.2)
  - mixture over ambiguous deduplication configurations (4.5, problem P3)
  - multi-stratum (metro) hierarchy
"""

from dataclasses import dataclass

import numpy as np
import pymc as pm


@dataclass
class BenchData:
    """Inputs for one stratum x severity cell.

    Attributes:
        C: crash counts per operator per period, shape (I, T)
        M_hat: reported VMT (miles) per operator per period, shape (I, T);
            np.nan where an operator filed nothing that period
        tau_prior_sd: prior scale for exposure reporting noise per operator,
            shape (I,). Calibrate from documented filing corrections, e.g.
            the relative magnitude of the Zoox Feb-2026 and Waymo Jul-2022
            CPUC revisions. Operators with no revision history get the
            industry-level default.
        delta_prior_mean, delta_prior_sd: prior on systematic log-bias of
            reported vs crash-generating VMT (passenger-service miles
            undercount total autonomous miles, so mean <= 0), shape (I,)
        H: human benchmark crash count in the stratum (scalar)
        V: human benchmark VMT in miles (scalar; treated as known here,
            measurement model on V is a v1 item)
        r_alpha, r_beta: Beta prior on the human reporting probability at
            this severity, from Blincoe et al. corrections. At injury+
            severities these concentrate near 1.
        pi_max: upper bound on the fraction of ADS SGO events below the
            police floor, from SGO severity flags (injury/airbag/tow), per
            operator, shape (I,). Used post hoc for the 4.6 bounds.
        pi_min: lower bound, from jurisdiction damage thresholds and
            unredacted narratives, shape (I,).
    """

    C: np.ndarray
    M_hat: np.ndarray
    tau_prior_sd: np.ndarray
    delta_prior_mean: np.ndarray
    delta_prior_sd: np.ndarray
    H: int
    V: float
    r_alpha: float
    r_beta: float
    pi_max: np.ndarray | None = None
    pi_min: np.ndarray | None = None
    overdispersed: bool = False  # period-level rate drift (Sec IV-B variant)


def build_model(d: BenchData, alpha_prior: str = "parity") -> pm.Model:
    """alpha_prior: 'parity' centers the operator baseline on the human
    rate (paper 4.7 default); 'diffuse' uses a wide fixed-center prior,
    the alternative reported in the prior-sensitivity analysis."""
    I, T = d.C.shape
    observed_mask = ~np.isnan(d.M_hat)
    M_hat_filled = np.where(observed_mask, d.M_hat, 1.0)  # placeholder, masked out

    with pm.Model() as model:
        # ---- 4.4 human benchmark, estimated not assumed (P5) ----
        # true human rate per mile; weak lognormal prior spanning plausible
        # urban injury-crash rates
        mu = pm.LogNormal("mu_human", mu=np.log(3e-6), sigma=1.5)
        r = pm.Beta("r_report", alpha=d.r_alpha, beta=d.r_beta)
        pm.Poisson("H_obs", mu=mu * d.V * r, observed=d.H)

        # ---- 4.7 partial pooling centered on human parity (P4) ----
        # alpha centered on log(mu): prior expectation is parity with humans
        sigma_op = pm.HalfNormal("sigma_op", sigma=1.0)
        if alpha_prior == "parity":
            alpha = pm.Normal("alpha", mu=pm.math.log(mu), sigma=1.0)
        else:  # diffuse
            alpha = pm.Normal("alpha", mu=np.log(3e-6), sigma=3.0)
        beta_raw = pm.Normal("beta_raw", mu=0.0, sigma=1.0, shape=I)
        log_lam = pm.Deterministic("log_lambda", alpha + beta_raw * sigma_op)
        lam = pm.Deterministic("lambda", pm.math.exp(log_lam))

        # ---- 4.3 exposure measurement error (P1) ----
        # A period with no filing means no service: it contributes neither
        # exposure nor crash likelihood. Modeling it would create phantom
        # exposure and overconfident rates for young operators.
        delta = pm.Normal(
            "delta", mu=d.delta_prior_mean, sigma=d.delta_prior_sd, shape=I
        )
        op_idx, per_idx = np.nonzero(observed_mask)
        n_obs = op_idx.size
        # latent true log-VMT for service periods only, anchored near each
        # operator-period's own filing
        log_M = pm.Normal(
            "log_M",
            mu=np.log(M_hat_filled)[observed_mask],
            sigma=2.0,
            shape=n_obs,
        )
        # measurement: reported = true + bias + noise, on the log scale
        pm.Normal(
            "M_hat_obs",
            mu=log_M + delta[op_idx],
            sigma=d.tau_prior_sd[op_idx],
            observed=np.log(M_hat_filled)[observed_mask],
        )

        # ---- 4.2 Poisson counts given true exposure (service periods) ----
        if d.overdispersed:
            sigma_eps = pm.HalfNormal("sigma_eps", sigma=0.5)
            eps = pm.Normal("eps", 0.0, 1.0, shape=n_obs)
            log_rate = log_lam[op_idx] + eps * sigma_eps
        else:
            log_rate = log_lam[op_idx]
        pm.Poisson(
            "C_obs",
            mu=pm.math.exp(log_rate) * pm.math.exp(log_M),
            observed=d.C[observed_mask],
        )

        # ---- headline estimand ----
        pm.Deterministic("rho", lam / mu)

    return model


def threshold_bounds(idata, pi_min: np.ndarray, pi_max: np.ndarray):
    """4.6: police-comparable rate-ratio bounds on posterior samples.

    rho_police(pi) = (1 - pi) * lambda_SGO / mu_police.
    Returns per-operator posterior summaries of the identified interval
    [rho(pi_max), rho(pi_min)] plus a sensitivity grid.
    """
    rho = idata.posterior["rho"].values.reshape(-1, idata.posterior["rho"].shape[-1])
    out = []
    grid = np.linspace(0, 1, 21)
    for i in range(rho.shape[1]):
        lo = rho[:, i] * (1 - pi_max[i])
        hi = rho[:, i] * (1 - pi_min[i])
        sens = {f"pi={p:.2f}": np.percentile(rho[:, i] * (1 - p), [2.5, 50, 97.5])
                for p in grid if pi_min[i] <= p <= pi_max[i]}
        out.append({
            "operator": i,
            "rho_lower_bound_ci": np.percentile(lo, [2.5, 50, 97.5]),
            "rho_upper_bound_ci": np.percentile(hi, [2.5, 50, 97.5]),
            "sensitivity": sens,
        })
    return out


def fit(model: pm.Model, draws=1000, tune=1000, chains=4, seed=42, cores=None):
    if cores is None:
        import os
        cores = max(1, min(chains, os.cpu_count() or 1))
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores,
            random_seed=seed, target_accept=0.9, progressbar=False,
        )
    return idata
