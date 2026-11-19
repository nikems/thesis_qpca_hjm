"""Download and FREEZE the input data (protocol point: freeze inputs with provenance).

Fetches the ECB euro-area AAA-rated government bond INSTANTANEOUS FORWARD curve
from the ECB Data Portal (SDMX API), for 12 maturities (3M..30Y), 2022-2024 daily,
and writes:
    data/forward_rates_raw.csv   - the frozen rates (percent), index = date
    data/maturities.csv          - maturity code -> years
    data/dataset_metadata.json   - source, dates, units, sha256 (provenance)

Requires internet access to data-api.ecb.europa.eu and the 'requests' package.
Run with:  python download_data.py

NOTE: the thesis uses the file frozen on 2026-11-15 (sha256 in data/dataset_metadata.json).
This script documents HOW it was obtained; it is NOT part of the pipeline. If the
frozen file exists it refuses to overwrite it. To download a new copy on purpose:
    python download_data.py --force
(a new download may differ if the ECB revises its data, and will change the sha256).
"""
import os, io, sys, json, hashlib, datetime
import numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if os.path.exists(os.path.join(OUT, "forward_rates_raw.csv")) and "--force" not in sys.argv:
    sys.exit("data/forward_rates_raw.csv already exists (frozen input of the thesis): not overwritten.\n"
             "Use  python download_data.py --force  only if you really want a new download.")
import requests
os.makedirs(OUT, exist_ok=True)

# ECB SDMX: dataflow YC, key = B.U2.EUR.4F.G_N_A.SV_C_YM.<IF_maturity>
# (AAA central-government curve, Svensson instantaneous forward 'IF_*')
MATS = {"IF_3M":0.25,"IF_6M":0.5,"IF_9M":0.75,"IF_1Y":1,"IF_2Y":2,"IF_3Y":3,
        "IF_5Y":5,"IF_7Y":7,"IF_10Y":10,"IF_15Y":15,"IF_20Y":20,"IF_30Y":30}
BASE  = "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM."
START, END = "2022-01-01", "2024-12-31"

def fetch(code):
    url = BASE + code
    r = requests.get(url, params=dict(format="csvdata", startPeriod=START, endPeriod=END),
                     timeout=120)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    return d[["TIME_PERIOD", "OBS_VALUE"]].rename(columns={"OBS_VALUE": code}).set_index("TIME_PERIOD")

print("Downloading ECB AAA instantaneous-forward curve ...")
frames = [fetch(c) for c in MATS]
wide = pd.concat(frames, axis=1).sort_index()
wide.index = pd.to_datetime(wide.index)


print(f"  got {wide.shape[0]} business days x {wide.shape[1]} maturities "
      f"({wide.index.min().date()} -> {wide.index.max().date()})")

# save rates 
wide.to_csv(os.path.join(OUT, "forward_rates_raw.csv"))
pd.DataFrame({"maturity_code": list(MATS), "maturity_years": list(MATS.values())}
            ).to_csv(os.path.join(OUT, "maturities.csv"), index=False)

# freeze provenance metadata (incl. sha256 of the raw file)
sha = hashlib.sha256(open(os.path.join(OUT, "forward_rates_raw.csv"), "rb").read()).hexdigest()
meta = dict(
    source="European Central Bank - Data Portal (SDMX), dataflow YC",
    series_key_template="YC.B.U2.EUR.4F.G_N_A.SV_C_YM.<IF_maturity>",
    curve="Euro-area AAA-rated central government bonds, Svensson instantaneous forward",
    data_type="Instantaneous forward rate (IF_*)",
    maturities_years=list(MATS.values()),
    start_period=START, end_period=END,
    first_date=str(wide.index.min().date()), last_date=str(wide.index.max().date()),
    n_business_days=int(len(wide)),
    units="percent (raw)",
    download_utc=datetime.datetime.utcnow().isoformat() + "Z",
    raw_sha256=sha,
)
json.dump(meta, open(os.path.join(OUT, "dataset_metadata.json"), "w"), indent=2)
print(f"  frozen: data/forward_rates_raw.csv  (sha256 {sha[:12]}...)")
print("  wrote maturities.csv,dataset_metadata.json")
print("Done.")
