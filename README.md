# Stock Price Prediction - ML Minor Project

Stock Price Prediction - ML Minor Project | Corizo Machine Learning October'26 (Live Edge) | Name: Dhaksith Subramanian

## Overview

This project investigates whether technical indicators derived from daily price and volume can forecast the next trading day's closing price of Reliance Industries (`RELIANCE.NS`, 8 years of daily OHLCV, 2018–2026) better than a naive random walk. Using 16 leakage-free features (lagged closes, simple moving averages, RSI, MACD, Bollinger-band width, rolling volatility, and daily return), we train and evaluate price-target models (Linear Regression, Random Forest, XGBoost, and a 60-day LSTM) against a naive "tomorrow = today" baseline on a strict chronological 80/20 train/test split. We also test a stationary return-target formulation and binary UP/DOWN directional classifiers to provide a thorough, benchmarked assessment of short-horizon predictability.

> **Viewing the notebook:** The fully executed notebook with all outputs, evaluation tables, and interactive diagnostic plots can be viewed directly on GitHub at [`notebooks/stock_price_prediction.ipynb`](notebooks/stock_price_prediction.ipynb). An exported HTML version is also available at [`notebooks/stock_price_prediction.html`](notebooks/stock_price_prediction.html).

---

## Results

Held-out test set evaluation (385 trading days, 2025-04-23 → 2026-11-09, prices in INR):

| Model | RMSE (INR) | MAE (INR) | MAPE (%) | R² | Directional acc. (%) |
|---|---:|---:|---:|---:|---:|
| **Naive (tomorrow = today)** | **19.00** | **13.53** | **0.99** | **0.9599** | **N/A** |
| Linear Regression (return target) — best fitted | 19.14 | 13.89 | 1.01 | 0.9593 | 46.5 |
| Random Forest (return target) | 19.18 | 13.74 | 1.00 | 0.9591 | 48.8 |
| XGBoost (return target) | 19.56 | 14.08 | 1.02 | 0.9575 | 50.1 |
| Linear Regression (price target) | 19.77 | 14.23 | 1.04 | 0.9566 | 49.9 |
| XGBoost (price target) | 22.74 | 17.05 | 1.24 | 0.9425 | 50.4 |
| Random Forest (price target) | 27.00 | 20.68 | 1.50 | 0.9189 | 47.0 |
| LSTM (60-day window) | 40.81 | 31.56 | 2.26 | 0.8148 | 48.3 |

> *Note:* The naive forecast predicts zero change, so its directional accuracy is **N/A** (not 0%): an up/down call is undefined when the forecast is flat.

### Direction Classification (UP vs DOWN, test set)

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|---:|
| Random Forest | 51.69 | 51.40 | 48.17 | 49.73 |
| Logistic Regression | 50.91 | 50.29 | 91.62 | 64.94 |
| XGBoost | 50.13 | 49.62 | 34.55 | 40.74 |
| Majority-class baseline | 49.61 | 49.61 | 100.00 | 66.32 |

With 385 test days the binomial standard error is ≈2.5 percentage points; the best classifier's +2.1 pp edge over the 49.6% majority baseline is not statistically meaningful.

---

## Honest Verdict

- **The naive "tomorrow = today" forecast is the overall winner on both RMSE and MAE** (19.00 INR and 13.53 INR).
- **Linear Regression is the best fitted model**: its return-target variant reaches 19.14 INR RMSE / 13.89 INR MAE (only 0.71% worse RMSE than the random walk), and the price-target variant reaches 19.77 / 14.23 INR.
- Return-target regression substantially helps tree models (Random Forest RMSE improves from 27.00 to 19.18 INR, XGBoost from 22.74 to 19.56 INR) by removing the non-stationary price level, but still does not beat the random walk.
- Directional accuracy for all models is close to 50% (indistinguishable from a coin flip).
- **Conclusion:** This is consistent with the random-walk / weak-form efficient market view, although one stock and one test window cannot prove it. Technical indicators do not provide an exploitable next-day edge on this liquid large-cap asset.

---

## How to Run

### 1. Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

*(TensorFlow is optional: if omitted, the notebook runs seamlessly and skips only the LSTM).*

### 2. Run Headless via nbconvert

Execute the notebook in place:

```powershell
jupyter nbconvert --execute --to notebook --inplace notebooks/stock_price_prediction.ipynb
```

Export the executed notebook to HTML:

```powershell
jupyter nbconvert --to html notebooks/stock_price_prediction.ipynb --output-dir notebooks --output stock_price_prediction.html
```

To rebuild the notebook from source first:

```powershell
python build_notebook.py
jupyter nbconvert --execute --to notebook --inplace notebooks/stock_price_prediction.ipynb
```

---

## Predicting Tomorrow's Close

Inside Python or the notebook, `predict_next_day()` loads the persisted deployment artifact (`models/best_model.joblib`) and scores the most recent completed trading day:

```python
from build_notebook import predict_next_day  # or run cell in notebook
predict_next_day(return_details=True)
# Output:
# {
#   "model": "Linear Regression (return target)",
#   "as_of": "2026-11-11",
#   "last_close": 1337.10,
#   "predicted_next_close": 1338.09,
#   "predicted_change_pct": 0.07
# }
```

---

## Project Structure

```
data/stock.csv                          raw daily OHLCV (2018-11-14 -> 2026-11-11)
download_data.py                        data downloader script (RELIANCE.NS)
build_notebook.py                       source generator for stock_price_prediction.ipynb
run_notebook.py                         headless execution helper with progress logging
notebooks/stock_price_prediction.ipynb  executed Jupyter Notebook (viewable on GitHub)
notebooks/stock_price_prediction.html   exported HTML of the executed notebook
models/best_model.joblib                persisted deployment bundle (LR return model + scaler)
outputs/model_comparison.csv            test metrics for all price and return models
outputs/classification_comparison.csv   test metrics for direction classifiers
outputs/figures/fig01..fig14.png        all generated diagnostic & EDA figures
REPORT.md                               comprehensive 4-page technical report
requirements.txt                        project dependencies
.gitignore                              git ignore configuration
```

---

## Disclaimer

This is a project for the Corizo Machine Learning program, **not investment advice**. Past performance does not guarantee future results.
