# qPCA for HJM caplet pricing — thesis code

Classical baseline and quantum PCA (qPCA) experiments for pricing a 3M×3M caplet in a
multi-factor Heath–Jarrow–Morton model calibrated on ECB euro-area AAA forward rates.
The qPCA part starts from the benchmark of Martin et al. (2021),
*Towards pricing financial derivatives with an IBM quantum computer*,
Phys. Rev. Research 3, 013167.

## Contents

| file | what it does |
|---|---|
| `b0_baseline.py` | **B0** classical baseline: covariance of daily forward changes, PCA, HJM volatilities, exact (Black) and Monte Carlo caplet prices, out-of-sample check on 2024 |
| `e1_replica_martin.py` | **E1** classical replication of Martin et al.: ρ = σ/Tr σ, spectra, U = e^{2πiρ}; Martin (Wilmott) data vs thesis (ECB) data |
| `e2_qpca_2x2.py` | **E2** QPE / qPCA on the 2×2 matrices (ideal simulator) |
| `e3_qpca_4x4.py` | **E3** QPE / qPCA on the 3×3 matrices embedded in 4×4 |
| `e4_qpca_noisy.py` | **E4** the same circuits on noisy simulators (fake IBM Yorktown, Fez) |
| `e6_qpca_pricing.py` | **E6** caplet price with λ, u estimated by qPCA, compared with B0 |
| `qpca_common.py` | functions shared by E1–E6 (input matrices, QPE circuits, tomography, logging) |
| `run_all.py` | runs B0, E1, E2, E3, E4, E6 in order |
| `download_data.py` | documents how the ECB data were downloaded (not part of the pipeline) |
| `data/` | frozen input data (ECB Data Portal, dataflow YC), with sha256 in `dataset_metadata.json` |
| `results/` | outputs of every experiment: CSV tables, figures, `<exp>_config.json`, `<exp>_log.txt` |
| `QPCA Martin/` | original code of Martin et al. ([github.com/amartinfer/QPCA](https://github.com/amartinfer/QPCA), Qiskit 0.7.1), kept unchanged as reference for the replication |

## How to run

Python 3.11 with the pinned versions in `requirements.txt`:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_all.py
```

The whole pipeline takes a few minutes on a laptop. B0 must run first because E1–E6 read
`results/b0/b0_reference.json`. Each script can also be run on its own, e.g.
`.venv/bin/python e1_replica_martin.py`. All random seeds are fixed, so every run gives the
same numbers.

B0 checks that `data/forward_rates_raw.csv` matches the sha256 recorded in
`data/dataset_metadata.json` and stops if it does not. `download_data.py` is only needed for a
new download (it also needs the `requests` package) and refuses to overwrite the frozen file.

## Data

ECB Data Portal, dataflow YC, series `YC.B.U2.EUR.4F.G_N_A.SV_C_YM.IF_<maturity>`:
euro-area AAA central-government bonds, Svensson instantaneous forward rates, 12 maturities
(3M–30Y), business daily 2022-01-03 → 2024-12-30 (767 days), in percent.
Calibration window 2022–2023 (512 days); 2024 is used only for the out-of-sample check in B0.

## Naming conventions in the code

| name | meaning |
|---|---|
| `eigvals` | eigenvalues, 1-D array, descending |
| `eigvecs` | eigenvectors, one per **column** (`eigvecs[:, j]` = u_{j+1}) |
| `lambda_hat`, `lambda_exact` | a single eigenvalue estimated by QPE / its exact value |
| `V`, `V_h`, `V_REF`, ... | **prices** only |
| `C`, `rho` | covariance matrix, density matrix ρ = C / Tr C |
