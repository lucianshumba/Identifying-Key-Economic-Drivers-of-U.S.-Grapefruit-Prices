# Identifying the Key Economic Drivers of U.S. Grapefruit Prices

### A Multivariate Feature-Engineered Time Series Analysis

**Course:** Financial Time Series Analytics 
**Product:** Grapefruit *(Citrus paradisi)*  
**Data Source:** USDA Economic Research Service — Fruit and Tree Nuts Yearbook  
**Period:** Marketing years 1980/81 – 2023/24 (44 annual observations)

---

## Overview

This project identifies and quantifies the key economic and structural drivers behind U.S. grapefruit grower prices over a 44-year period. Using USDA supply-side data combined with macroeconomic indicators, disease proxies, and weather shock variables, the analysis builds a full pipeline from exploratory data analysis through feature engineering, variable selection, and ARDL model estimation to interpretive 5-year scenario projections.

The central narrative: grapefruit prices rose from under $2/box in the early 1980s to nearly $19/box by 2020 — driven not by inflation or demand, but by a structural collapse in Florida's bearing acreage (down >85% from its 2004 peak), accelerated by back-to-back hurricanes in 2004 and the spread of Huanglongbing (HLB) citrus greening disease from 2005 onward.

---

## Repository Structure

```
├── Identifying_Key_Economic_Drivers_of_U_S_Grapefruit_Prices.ipynb   # Main analysis notebook
├── FruitYearbookCitrusFruit_CTables.csv                               # USDA ERS source data
├── grapefruit_master_dataset.csv                                      # Analysis-ready merged dataset (output)
├── visual1_price_series.png          # Price time series with event annotations
├── visual2_log_and_diff.png          # Log price and first difference (stationarity)
├── visual3_pct_change.png            # Year-on-year % price change
├── visual4_supply_collapse.png       # Bearing area and production decline
├── visual5_price_vs_prod.png         # Supply–price scatter plot
├── visual6_rolling_stats.png         # Rolling mean and volatility
├── visual7_acf.png                   # ACF: levels vs first difference
├── visual_fe_fourpanel.png           # Feature engineering visualizations
├── visual_rf_importance.png          # Random Forest feature importance
├── visual_ardl_diagnostics.png       # ARDL model evaluation diagnostics
├── visual_projection.png             # 5-year scenario projection
└── visual_partial_dependence.png     # Partial dependence plots (top 3 RF drivers)
```

---

## Notebook Structure

| Part | Section | Content |
|------|---------|---------|
| 1 | Setup | Imports, colour palette, shared `get_annual()` helper |
| 2 | Load & Clean | Load USDA CSV, extract all price/supply series |
| 3 | Descriptive Stats | Summary statistics and regime comparisons |
| 4–10 | Visuals 1–7 | Time series, log-diff, YoY change, supply collapse, scatter, rolling stats, ACF |
| 11 | Stationarity | ADF and KPSS formal tests — confirms I(1) |
| 12 | Regime Table | Five structural regimes identified from EDA |
| 13 | Variable Registry | 15 external drivers with sources and business justifications |
| 14 | FRED Download | CPI, unemployment, interest rates, FX, diesel from FRED |
| 15 | Shock Proxies | HLB infection rate, hurricane ACE index, freeze event dummy |
| 16 | Master Dataset | Merged analysis-ready DataFrame saved to CSV |
| 17 | Correlation Check | Sanity check — do variables move in expected directions? |
| 18–25 | Phase 3: Feature Engineering | Log transforms, differencing, rolling stats, lags, regime dummies, interactions, standardisation, EWMA volatility, 4 required visualisations |
| 26–27 | Phase 4: Variable Selection | VIF diagnostics + Random Forest OOB importance screening |
| 28–32 | Phase 5: Model Development | Three model proposals, ARDL estimation, Pesaran bounds test, CUSUM/RESET/threshold analysis, partial dependence, 5-year projection |
| 33–35 | Phase 6: Model Risk | 6 model assumptions, 6 limitations with mitigations, SR 11-7 monitoring plan, 5 early-warning indicators |

---

## Key Findings

**Primary driver — supply destruction:**
The inverse supply–price relationship is statistically significant across the full sample. The log-production coefficient (−0.570*) and lagged production coefficient (−0.001**) confirm that falling output is the dominant force behind rising prices.

**Price persistence:**
The AR(1) term is significant at the 1% level (coefficient +0.418***), indicating that price shocks propagate forward by one year — strong momentum in the grower price series.

**Secondary drivers:**
HLB disease intensity, Atlantic hurricane ACE, CPI, and the USD index are economically motivated and correctly signed, but are statistically insignificant due to multicollinearity — they all trend together in the post-2006 period and the model cannot disentangle their individual contributions with only 44 observations.

**Functional form:**
Ramsey RESET test confirms the linear ARDL specification is adequate (F = 0.045, p = 0.956). A production threshold at ln(prod) ≈ 6.7 (~812,000 tons) was identified, below which supply shocks have an amplified price impact.

**Structural stability:**
CUSUM detects a narrow breach (6.93 vs 6.22 critical bound) concentrated in the 2020–2021 COVID demand surge. Parameters are stable across the other 41 observations; a targeted `covid_spike` dummy would fully absorb the residual instability.

**5-year outlook (2024–2028):**
- Base case: ~$17/box — continued HLB-constrained supply, moderate inflation
- Stress case: ~$22/box — major hurricane + worsening HLB compound
- Recovery case: ~$11/box — HLB-resistant rootstock breakthrough + acreage expansion

---

## Data Sources

| Variable | Source | Series ID |
|----------|--------|-----------|
| Grapefruit grower price ($/box) | USDA ERS Fruit & Tree Nuts Yearbook | Table C-4 |
| Production (000 tons) | USDA ERS | Table C-2 |
| Bearing area (000 acres) | USDA ERS | Table C-1 |
| Orange grower price | USDA ERS | Table C-16 |
| Lemon grower price | USDA ERS | Table C-11 |
| CPI all urban | FRED / BLS | CPIAUCSL |
| Unemployment rate | FRED / BLS | UNRATE |
| Federal Funds Rate | FRED | FEDFUNDS |
| USD Broad Index | FRED | DTWEXBGS |
| Diesel price | FRED / EIA | GASDESW |
| HLB infection rate | FDACS Citrus Health Response Program | Annual reports |
| Hurricane ACE index | NOAA National Hurricane Center | Historical ACE |
| Florida freeze events | NOAA NCEI + USDA NASS crop damage reports | Manually coded |

The notebook attempts a live USDA CSV download at runtime; embedded fallback arrays are provided for offline use.

---

## Feature Engineering (Phase 3)

Eight transformation types are applied across the master dataset:

1. **Log transforms + first differencing** — stabilise variance; achieve stationarity
2. **Rolling statistics** — 5-year and 3-year rolling mean and std dev
3. **Lagged variables** — lags 1–2 for supply, disease, and macro variables
4. **Regime dummies** — five structural regimes plus a `post_hlb` binary break
5. **Seasonal indicator** — heavy-season vs light-season proxy from rolling median production
6. **Interaction terms** — HLB × ACE, supply × CPI, post_HLB × production
7. **Standardisation** — StandardScaler (z-score) and MinMaxScaler for ML inputs
8. **EWMA volatility** — exponentially weighted moving average std dev (span = 5)

---

## Variable Selection (Phase 4)

Two-step procedure:

1. **VIF analysis** — removes regressors with Variance Inflation Factor > 10 to address multicollinearity before model estimation
2. **Random Forest OOB importance** — ranks surviving features by predictive signal; OOB R² reported in lieu of time-series CV (n = 44 is too small for reliable fold-based validation)

---

## Model Choice (Phase 5)

**Selected: ARDL (Autoregressive Distributed Lag)**

Three model proposals were documented:

| Model | Interpretability | Rationale |
|-------|----------------|-----------|
| **ARDL** ✓ Selected | High | Handles mixed I(0)/I(1) regressors; coefficients = long-run price elasticities; appropriate for n = 44 |
| Random Forest | Medium | High overfitting risk with n = 44; acceptable for importance screening only |
| LSTM | Low | Requires 200+ sequences for stable weight learning; annual frequency provides no meaningful advantage over AR lags; not interpretable without post-hoc SHAP |

Cointegration is assessed via the Pesaran, Shin & Smith (2001) bounds F-test. Model diagnostics include: HC3 robust standard errors, Ramsey RESET test, Hansen-style threshold regression, CUSUM stability, and partial dependence analysis.

---

## Model Risk (Phase 6)

**Six model assumptions** are documented with specific monitoring triggers, breach conditions, and remediation steps — including integration order, parameter stability, exogeneity, homoskedasticity, multicollinearity limits, and HLB proxy validity.

**Six model limitations** are acknowledged — small sample size, structural breaks, proxy imprecision, annual data granularity, omitted demand-side variables, and out-of-sample extrapolation risk.

**Monitoring plan** follows SR 11-7 model risk management principles:
- Annual recalibration (every August, after USDA Yearbook release)
- Event-triggered reviews (hurricane landfall, new disease alert, rate cycle shift, trade policy change, CUSUM breach)
- Five-year comprehensive re-specification

Five early-warning indicators (EWIs) cover HLB acceleration, major hurricane approach, drought, import volume spikes, and residual drift.

---

## Requirements

```
pandas
numpy
matplotlib
statsmodels
scikit-learn
pandas_datareader   # optional — for live FRED download in Part 14
```

Install with:

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn pandas-datareader
```

---

## Usage

1. Clone this repository
2. Place `FruitYearbookCitrusFruit_CTables.csv` in the root directory (or run without it — embedded fallback arrays are provided)
3. Open `Identifying_Key_Economic_Drivers_of_U_S_Grapefruit_Prices.ipynb` in Jupyter
4. Run cells sequentially — each part depends on objects created in prior parts
5. Output figures and `grapefruit_master_dataset.csv` are saved to the working directory

> **Note:** Part 14 requires internet access to pull live FRED data. The notebook falls back to embedded annual averages automatically if the connection fails.

---

## Background: Why Grapefruit Prices Are Structurally Different

Two demand-side forces compound the supply destruction narrative:

**1. Grapefruit–drug interactions:** Grapefruit juice inhibits cytochrome P450 3A4, an enzyme that metabolises statins, calcium-channel blockers, and immunosuppressants. As statin prescriptions expanded from ~10 million (1990) to over 90 million Americans (2020), cardiologists increasingly advised patients to avoid grapefruit — creating a slow structural demand break independent of supply. Per-capita consumption fell from ~24 lbs (1995) to ~6 lbs (2023).

**2. Retail shelf contraction:** Supermarkets shifted allocation toward higher-margin citrus varieties (Cara Cara, Sumo oranges). Institutional food-service demand also declined as breakfast menus moved away from the traditional grapefruit half. These forces suppress the demand ceiling, meaning even a supply recovery may not restore prices to pre-1995 levels.

---

## License

This project was produced for academic purposes. USDA ERS data is public domain. FRED/BLS/NOAA data is subject to each agency's respective terms of use.
