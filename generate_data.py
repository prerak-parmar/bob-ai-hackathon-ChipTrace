"""
generate_data.py
Creates a synthetic but realistic wafer-lot dataset for the hackathon prototype.
Run this once to produce wafer_data.csv, which the Streamlit app will use
as the default/demo dataset (users can also upload their own CSV with the
same column names).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_LOTS = 500

# --- Baseline "good process" parameters ---
# Temperature (deg C), Pressure (torr/bar-ish unit), Power (W), Gas Flow (sccm)
temperature = np.random.normal(loc=450, scale=8, size=N_LOTS)
pressure    = np.random.normal(loc=2.2, scale=0.2, size=N_LOTS)
power       = np.random.normal(loc=800, scale=15, size=N_LOTS)
gas_flow    = np.random.normal(loc=120, scale=6, size=N_LOTS)

# --- Inject some "drifted" / abnormal lots (simulates real fab excursions) ---
drift_idx = np.random.choice(N_LOTS, size=int(N_LOTS * 0.15), replace=False)
temperature[drift_idx] += np.random.normal(loc=25, scale=8, size=len(drift_idx))
pressure[drift_idx]    += np.random.normal(loc=0.7, scale=0.3, size=len(drift_idx))

# --- Defects: driven mostly by temperature & pressure deviation from ideal ---
temp_dev = np.abs(temperature - 450)
pres_dev = np.abs(pressure - 2.2)

defects = (
    0.35 * temp_dev
    + 6.0 * pres_dev
    + 0.02 * np.abs(power - 800)
    + np.random.normal(loc=2, scale=1.5, size=N_LOTS)
)
defects = np.clip(defects, 0, None).round().astype(int)

# --- Yield: decreases as defects increase (with some noise) ---
base_yield = 99.0
yield_pct = base_yield - (defects * 0.55) - np.random.normal(0, 1.0, N_LOTS)
yield_pct = np.clip(yield_pct, 40, 99.9).round(2)

df = pd.DataFrame({
    "Lot": [f"L{str(i+1).zfill(3)}" for i in range(N_LOTS)],
    "Temperature": temperature.round(1),
    "Pressure": pressure.round(2),
    "Power": power.round(1),
    "GasFlow": gas_flow.round(1),
    "Defects": defects,
    "Yield": yield_pct,
})

df.to_csv("wafer_data.csv", index=False)
print(f"Generated wafer_data.csv with {len(df)} lots")
print(df.head(10).to_string(index=False))
print("\nSummary:")
print(df.describe().round(2).to_string())
