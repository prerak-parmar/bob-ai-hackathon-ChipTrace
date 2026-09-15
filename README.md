# Wafer Yield Root-Cause & Risk Prediction Platform

**IBM Bob AI Innovation Hackathon — CHARUSAT**

## Problem Statement

Semiconductor fabs run many wafer lots, and yield (the % of good chips produced)
fluctuates lot to lot. When yield drops, process engineers need to quickly:
1. Understand *why* yield dropped
2. Know which process parameters are most likely responsible
3. Get concrete next steps to investigate
4. Flag upcoming lots that are at risk **before** they run, based on planned
   process parameters

Doing this manually from raw process logs and defect reports is slow and
inconsistent. This project is an enterprise decision-support platform that automates it.

## Proposed Solution

An enterprise-grade Streamlit dashboard styled in a clean white industrial design that takes historical wafer-lot data (Lot ID, Temperature, Pressure, Power, Gas Flow, Defect count, Yield) and:

1. **Monitors production** with executive KPI cards, yield distribution, and an interactive 300mm wafer spatial die map.
2. **Analyzes process stability** using Statistical Process Control (SPC) run charts with 3-Sigma control limits (UCL / LCL), rolling means, and parameter correlation heatmaps.
3. **Ranks contributing factors** using Random Forest feature importance, distribution box plots, and standard 8D Corrective Action Protocols (RCCA).
4. **Simulates risk for planned lots** with recipe presets, quantitative risk scorecards, parameter deviation diagnostics, and CSV report export.

**Important framing:** the "root cause ranking" is a *statistical association*
ranking from model feature importance, not a proven causal analysis — this is
stated explicitly in the UI to stay scientifically honest for a fab context.

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Data handling | Pandas, NumPy |
| ML | Scikit-learn (RandomForestRegressor) |
| Visualization | Plotly |
| Dashboard/UI | Streamlit |

## Project Structure

```
wafer-yield-app/
├── .streamlit/
│   └── config.toml     # Enterprise light theme configuration
├── app.py              # Streamlit UI (4 enterprise modules)
├── engine.py           # Core analytics, SPC limits, wafer die mapping, and ML logic
├── generate_data.py    # Creates the synthetic demo dataset
├── test_engine.py      # Unit test suite
├── wafer_data.csv      # Demo dataset (500 lots)
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Clone the repo and cd into the project folder
git clone <your-repo-url>
cd wafer-yield-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Run unit tests
python test_engine.py

# 4. Launch the platform
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Using Your Own Data

Upload a CSV via the sidebar with (at minimum) these columns:

| Column | Description |
|---|---|
| `Lot` | Lot identifier (optional — auto-generated if missing) |
| `Temperature` | Process temperature |
| `Pressure` | Process pressure |
| `Power` | Process power |
| `Defects` | Defect count |
| `Yield` | Yield % |
| `GasFlow` | Gas flow rate (optional) |

The app validates columns, coerces numeric types, and drops invalid rows with
a warning shown in the sidebar.

## Platform Modules

1. **Overview Dashboard** — Executive KPIs, yield distribution histogram with specification limits, interactive 300mm wafer spatial die map, and production lot records.
2. **Yield & Defect Analytics** — Statistical Process Control (SPC) run chart with 3-sigma limits (UCL / LCL), rolling 10-lot trendline, bivariate scatter regressions, and correlation matrix.
3. **Root Cause Analysis** — Feature importance bar chart, model reliability metrics ($R^2$, MAE), parameter distribution boxplots (healthy vs excursion lots), and 8D RCCA remediation matrix.
4. **Lot Risk Simulator** — Pre-flight recipe simulation with quick presets, quantitative yield prediction, risk categorization badges, parameter deviation breakdown, and CSV report export.

## What This Prototype Does NOT Do

This is a decision-support software prototype, not:
- An actual semiconductor fab simulation or physical sensor system
- A proven causal-inference engine (ranking is association-based, clearly
  labeled as such)
- A production-grade MLOps pipeline — model is retrained in-session for
  demo purposes

## Future Improvements

- SHAP values for per-lot explainability instead of global feature importance
- Time-series / drift detection (e.g. CUSUM) for early excursion alerts
- Multi-model ensemble with confidence intervals on predicted yield
- Integration with real fab MES/SPC systems via API
