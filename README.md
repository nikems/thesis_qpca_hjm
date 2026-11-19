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
| `e1_replica_martin.py` | **E1** classical replication of Martin et al. Comparison of Martin (Wilmott) data vs thesis (ECB) data |
| `e2_qpca_2x2.py` |
| `e3_qpca_4x4.py` |
| `e4_qpca_noisy.py` |
| `e5_qpca_harw.py` |
| `e6_qpca_pricing.py` |
| `download_data.py` | Data download|
| `data/` | frozen input data (ECB Data Portal, dataflow YC), with sha256 in `dataset_metadata.json` |
| `results/` | outputs of every experiment: CSV tables, figures.
| `QPCA Martin/` | original code of Martin et al. ([github.com/amartinfer/QPCA](https://github.com/amartinfer/QPCA), Qiskit 0.7.1), kept unchanged as reference for the replication |

## How to run

Python 3.11 with the pinned versions in `requirements.txt`:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```


## Data

ECB Data Portal, dataflow YC, series `YC.B.U2.EUR.4F.G_N_A.SV_C_YM.IF_<maturity>`:
euro-area AAA central-government bonds, Svensson instantaneous forward rates, 12 maturities
(3M–30Y), business daily 2022-01-03 → 2024-12-30 (767 days), in percent.
Calibration window 2022–2023 (512 days); 2024 is used only for the out-of-sample check in B0.


