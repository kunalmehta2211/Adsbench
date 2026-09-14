# adsbench: Operator-Agnostic ADS Crash Rate Benchmarking from Public Data

Companion pipeline for the paper "Benchmarking ADS Safety Without the Developer."
Everything here runs on public data. No developer cooperation required.

## Pipeline stages

1. **Acquire** (`src/acquire.py`): download raw public data. Run locally
   (needs open internet). Sources and exact URLs are listed below and in the
   script. Freeze the snapshot date in `data/SNAPSHOT.txt`.
2. **Ingest and deduplicate** (`src/ingest_sgo.py`, `src/dedup.py`): parse the
   NHTSA SGO ADS incident CSV, keep the latest version of each report, link
   records that describe the same physical crash across reporting entities and
   against California DMV OL316 filings, and emit crash counts per operator,
   metro, severity, and quarter, plus alternative counts for unresolvable
   (redacted) linkages.
3. **Exposure** (`src/exposure.py`): parse CPUC quarterly VMT filings per
   operator, tag every figure with its revision history (original vs corrected
   filings), and emit reported-VMT tables with revision magnitudes used to
   calibrate the measurement error priors.
4. **Benchmark** (`src/benchmark.py`): build human crash rates per metro and
   severity from SWITRS plus Blincoe underreporting corrections, and the UMTRI
   naturalistic ridehail benchmark as the bracketing likelihood.
5. **Fit** (`src/model.py`): the hierarchical Bayesian model (paper Section 4).
6. **Validate** (`src/synthetic.py`): simulate data with known ground truth and
   check the model recovers it. Run this before touching real data; its output
   becomes the model-validation paragraph in paper Section 5.

## Data sources (all public)

| Data | Source | Access |
|---|---|---|
| ADS crashes (national) | NHTSA SGO incident reports, "ADS" CSV | nhtsa.gov -> Standing General Order data portal, monthly refresh |
| ADS crashes (CA, 2nd channel) | CA DMV AV collision reports (OL316) | dmv.ca.gov -> AV collision reports page |
| ADS exposure (CA) | CPUC AV program quarterly reports (VMT per operator) | cpuc.ca.gov -> Autonomous Vehicle Programs -> Quarterly Reporting |
| Human crashes (CA) | SWITRS / TIMS extract | tims.berkeley.edu (free account) |
| Human crash underreporting | Blincoe et al. 2023, NHTSA DOT HS 813 403 | correction factors transcribed in `data/blincoe_corrections.csv` |
| Human ridehail benchmark | Flannagan et al. 2023 (UMTRI) | published rates transcribed in `data/umtri_benchmark.csv` |
| Human exposure | FHWA HPMS / Caltrans VMT | fhwa.dot.gov, dot.ca.gov |

## Run order

```bash
pip install -r requirements.txt
python src/synthetic.py            # model validation, no downloads needed
python tests/dryrun.py             # end-to-end pipeline check on fixtures
python src/acquire.py              # local machine only
python src/ingest_sgo.py && python src/dedup.py
python src/exposure.py survey    # then confirm data/interim/cpuc_mapping.csv
python src/exposure.py extract
python src/benchmark.py          # requires filled data/verify/*.csv
python src/run_real.py
python src/run_variants.py       # sensitivity suite
```

## Model status

`src/model.py` implements: Poisson counts, latent true VMT with lognormal
measurement error calibrated from documented filing corrections, partial
pooling of operator log-rates centered on human parity, benchmark with
uncertain reporting probability, and posterior threshold-mismatch bounds
computed on posterior samples. The monotone severity-thinning chain and the
mixture over ambiguous deduplication configurations are v1 items, flagged in
code.
