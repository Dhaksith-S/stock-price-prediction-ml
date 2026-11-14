# Stock Price Prediction with Technical Indicators

Stock Price Prediction - ML Minor Project | Corizo Machine Learning October'26 (Live Edge) | Name: Dhaksith Subramanian

### ML Minor Project — Report

---

## Abstract

This project tests whether technical indicators derived from daily price and volume can forecast the next
trading day's close of Reliance Industries (RELIANCE.NS) better than a naive random walk. Eight years of daily
OHLCV data (1,974 observations, 2018-11-14 to 2026-11-11) were used to engineer 16 leakage-free features
(lagged closes, moving averages, RSI, MACD, Bollinger-band width, rolling volatility and daily return). Five
price-target models were trained on a chronological 80/20 split: a naive baseline, Linear Regression, Random
Forest, XGBoost and a 60-day LSTM. A second experiment reformulated the target as the next-day percentage
return, and a third predicted direction (UP/DOWN) with three classifiers.

The naive forecast is the overall winner on both RMSE and MAE (19.00 INR and 13.53 INR). Linear Regression is
the best fitted model — its return-target variant reaches 19.14 INR RMSE and 13.89 INR MAE, 0.71% worse than
the random walk. Directional accuracy is statistically indistinguishable from chance for every model (46.5%–
50.4% for regressors, 49.6%–51.7% for classifiers, versus a 49.6% majority-class baseline). The honest
conclusion is that technical-indicator ML does not provide a usable next-day edge on this stock and period.

## 1. Introduction

Short-horizon stock forecasting is a deliberately hard test for machine learning. Prices are noisy,
non-stationary and close to a random walk, so a model that simply repeats today's close is a strong benchmark.
The project asks three questions:

1. Can engineered technical features beat the naive "tomorrow = today" forecast on price error?
2. Does changing the target from price level to next-day return help, especially for tree models that cannot
   extrapolate a trend?
3. Can a classifier predict the direction of the next day's move better than chance or a majority-class
   baseline?

All experiments use one asset, one chronological split and one random seed, so differences between models are
not confounded by different data. The contribution is an honest benchmark: every result is compared with the
random walk, including the cases where the answer is negative.

## 2. Dataset

Daily OHLCV data were downloaded with `yfinance` and stored at `data/stock.csv`.

| Property | Value |
|---|---|
| Ticker | RELIANCE.NS (NSE, India) |
| Rows | 1,974 daily bars |
| Date range | 2018-11-14 → 2026-11-11 (8.0 years) |
| Columns | Date, Open, High, Low, Close, Adj Close, Volume |
| Missing values / duplicate dates | 0 / 0 |
| Trading days per year | 247 |
| Closing-price range | 404 → 1,601 INR |
| Mean daily return / volatility | +0.065% / 1.75% |
| Annualised volatility | 27.8% |
| Skewness / excess kurtosis | 0.55 / 10.32 |
| Best / worst day | +14.72% / −13.15% |
| Average / maximum volume | 18.0M / 142.7M shares (2020-04-22) |

The series shows a multi-year uptrend interrupted by the March 2020 crash, with volatility clustering and fat
tails (excess kurtosis 10.32). Volume spikes coincide with turbulent periods rather than with the trend:
the correlation between close and volume is −0.37, while close and daily return are essentially uncorrelated
(−0.001). Figure `fig01_closing_price.png` shows the price history, `fig02_volume.png` the volume profile,
`fig03_moving_averages.png` the 50/200-day averages and `fig04_returns_distribution.png` the return
distribution. OHLC levels are near-perfectly correlated by construction (`fig05_correlation_heatmap.png`),
which is why the model relies on returns and indicators rather than four near-duplicate price columns.

After feature engineering, 50 warm-up rows are dropped, leaving **1,924 modelling rows** (2019-01-24 →
2026-11-09). The first 80% (1,539 rows) is training data and the last 20% (385 rows, 2025-04-23 → 2026-11-09)
is the test set. Mean target price is 1,046.32 ± 276.08 INR in training and 1,385.50 ± 94.84 INR in the test
period; the test window is calmer (annualised volatility 22.5% versus 28.9% in training).

## 3. Methodology

### 3.1 Features

Each row uses only information available at the close of that day:

| Group | Features | Intuition |
|---|---|---|
| Lagged closes | `Lag_1`, `Lag_2`, `Lag_3`, `Lag_5`, `Lag_10` | Short-term memory |
| Moving averages | `SMA_5`, `SMA_10`, `SMA_20`, `SMA_50` | Trend and support/resistance |
| Momentum | `RSI_14`, `MACD`, `MACD_Signal`, `MACD_Diff` | Overbought/oversold, trend shifts |
| Volatility | `BB_Width`, `Volatility_20` | Band width, 20-day realised volatility |
| Returns | `Daily_Return` | Today's percentage move |

### 3.2 Target and leakage control

The price target is `Close[t+1]`, created with a single `shift(-1)`. All indicators are backward-looking, rows
with warm-up NaNs are dropped, and the scaler is fitted on training data only. The split is chronological with
no shuffling, and hyperparameter search uses `TimeSeriesSplit(n_splits=3)` inside the training window.

### 3.3 Models and evaluation

Five price-target models were compared: naive (tomorrow = today), Linear Regression (standardised features),
Random Forest, XGBoost, and an LSTM on 60-day windows of scaled features (48 units, dropout 0.2, early
stopping; trained for 24 epochs, best validation MSE 0.01722). The tree models were lightly tuned with
`TimeSeriesSplit`; the selected settings were 200 trees / unlimited depth / minimum leaf 3 for Random Forest
and 400 trees / depth 3 / learning rate 0.03 for XGBoost.

Models are scored on RMSE, MAE, MAPE, R² and directional accuracy. The naive forecast predicts zero change,
so its directional accuracy is **N/A** — an up/down call is undefined when the forecast is flat.

### 3.4 Return and direction formulations

The return experiment predicts `(Close[t+1] / Close[t]) − 1` with Linear Regression, Random Forest and XGBoost,
then converts the prediction back to a price (`Close[t] × (1 + predicted return)`) so the same metrics apply.
The classification experiment labels the next day UP (1) if its return is strictly positive and DOWN (0)
otherwise, and trains Logistic Regression, Random Forest and XGBoost. It is compared with a majority-class
baseline fitted on the training labels. No hyperparameter search is used for the classifiers, to avoid tuning
on a small sample.

## 4. Results

### 4.1 Price and return targets

Table 1 (also `outputs/model_comparison.csv`) reports the held-out test results. The naive forecast has the
lowest RMSE and MAE of all eight entries.

**Table 1 — Test-set metrics (385 days, INR).**

| Model | RMSE | MAE | MAPE (%) | R² | Direction (%) |
|---|---:|---:|---:|---:|---:|
| **Naive (tomorrow = today)** | **19.00** | **13.53** | **0.99** | **0.9599** | **N/A** |
| Linear Regression (return target) | 19.14 | 13.89 | 1.01 | 0.9593 | 46.5 |
| Random Forest (return target) | 19.18 | 13.74 | 1.00 | 0.9591 | 48.8 |
| XGBoost (return target) | 19.56 | 14.08 | 1.02 | 0.9575 | 50.1 |
| Linear Regression (price target) | 19.77 | 14.23 | 1.04 | 0.9566 | 49.9 |
| XGBoost (price target) | 22.74 | 17.05 | 1.24 | 0.9425 | 50.4 |
| Random Forest (price target) | 27.00 | 20.68 | 1.50 | 0.9189 | 47.0 |
| LSTM (60-day window) | 40.81 | 31.56 | 2.26 | 0.8148 | 48.3 |

Linear Regression's return-target variant is the best fitted model: 0.71% worse RMSE and 2.6% worse MAE than
the naive forecast. The other fitted models are 0.9%–114.8% worse on RMSE. The return target helps the tree
models substantially (Random Forest 27.00 → 19.18 INR, XGBoost 22.74 → 19.56 INR) but does not close the gap
completely. On the return scale the three return models score RMSE 0.01426, 0.01431 and 0.01459, i.e. about
1.43%–1.46% of price per day.

Figure `fig06_model_comparison.png` shows RMSE, MAPE and directional accuracy for the price-target models;
`fig11_price_vs_return_rmse.png` compares the two formulations side by side. `fig07_actual_vs_predicted.png`
shows that the best fitted model tracks the price level closely but lags turning points, which is exactly the
behaviour of a near-random-walk forecast. Residuals are centred near zero with occasional large misses
(`fig08_residuals.png`), and feature importance is dominated by `Lag_1` and `SMA_5`; RSI, MACD and
Bollinger-band features contribute almost nothing (`fig09_feature_importance.png`). The LSTM learning curve is
in `fig10_lstm_training.png`.

### 4.2 Direction classification

**Table 2 — UP/DOWN classification on the test set.**

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|---:|
| Random Forest | 51.69 | 51.40 | 48.17 | 49.73 |
| Logistic Regression | 50.91 | 50.29 | 91.62 | 64.94 |
| XGBoost | 50.13 | 49.62 | 34.55 | 40.74 |
| Majority-class baseline | 49.61 | 49.61 | 100.00 | 66.32 |

The best classifier (Random Forest, 51.69%) beats the majority-class baseline (49.61%) by 2.1 percentage
points. With 385 test days the binomial standard error is ≈2.5 pp, so this is not statistically distinguishable
from chance. Logistic Regression reaches 50.91% by predicting UP on 348 of 385 days (recall 91.62%, precision
50.29%), which is effectively a no-skill strategy with a high F1 for the UP class. The baseline's F1 of 66.32%
comes from always predicting the training majority class (UP was 51.4% of training days) while only 49.6% of
test days were UP; it illustrates why accuracy, not F1 alone, is the fair headline for balanced classes.
Confusion matrices are in `fig12_confusion_matrices.png` and the full comparison in
`fig13_classifier_comparison.png`. The training/test up-day shares were 51.4% and 49.6% respectively.

### 4.3 Deployment check

The best fitted model (Linear Regression on returns) was saved with `joblib` and reloaded in a fresh process.
For 2026-11-11 (last close 1,337.10 INR) it predicts a next-day close of **1,338.09 INR (+0.07%)**, confirming
that the persisted artifact reproduces the notebook's prediction.

## 5. Discussion and Limitations

**Why the naive forecast wins.** At a one-day horizon the best predictor of tomorrow's close is today's close.
The remaining move is close to unforecastable noise, and every fitted model introduces estimation error on top
of that noise. The best fitted model loses by only 0.71%, but the sign of the gap is consistent across
formulations. A high R² (0.81–0.96 for all price-level models) mostly reflects the persistence of the price
level: repeating today's close already achieves R² 0.96.

**Why returns help the trees.** Random Forest and XGBoost cannot extrapolate beyond the price range seen in
training, so a trending price level produces large absolute errors. Training on returns removes the level from
the target and cuts RMSE by 7.8 INR for Random Forest and 3.2 INR for XGBoost. Linear Regression benefits less
(−0.6 INR) because it already extrapolates linearly.

**Why the cross-validation errors look worse.** CV RMSE during tuning was 101.27 INR (Random Forest) and 102.90
INR (XGBoost), far above the 19–27 INR test errors. The CV folds cover the more volatile 2019–2024 training
window, where prices ranged from roughly 500 to 1,500 INR. Absolute-INR errors scale with price level, so CV
and test errors are not directly comparable.

**Direction is harder than level.** Classifiers hover around 50%, and the best result is within one binomial
standard error of the majority baseline. Direction is a low-signal target with roughly balanced classes; 385
test days cannot resolve a 1–2 pp edge, and there is no evidence of one here.

**Limitations.** (i) One stock, one eight-year period and one split — results may not generalise to other
tickers or regimes. (ii) The feature set contains only price and volume; earnings, news, sentiment, index flows,
crude prices and currency moves are absent. (iii) The information ceiling is visible in feature importance:
`Lag_1` and `SMA_5` dominate, and momentum/volatility indicators add little. (iv) Evaluation ignores brokerage,
taxes, slippage and position sizing; a price forecast is not a strategy, and a 50% hit rate loses money after
costs. (v) Hyperparameters were tuned with time-series CV, but repeated experiments on the same test window can
still leak information through model selection. (vi) Past performance does not guarantee future results.

## 6. Conclusion

The naive forecast is the overall winner on both RMSE and MAE. Linear Regression is the best fitted model:
its return-target variant achieves 19.14 INR RMSE, 13.89 INR MAE and 1.01% MAPE, 0.71% worse RMSE than repeating
today's close, while the price-target variant achieves 19.77 INR RMSE. Return targets materially improve the
tree models but do not beat the random walk. Direction classifiers (49.6%–51.7%) show no reliable edge over
chance or over the majority-class baseline.

The honest conclusion is that technical-indicator machine learning does not provide a usable next-day forecast
for RELIANCE.NS in this period, in price or return form, and does not reliably predict direction. This is a
legitimate result: it is consistent with the random-walk / weak-form efficient market view, although one stock
and one test window cannot prove it. It is more useful than an inflated claim of predictive power.

## 7. Future Work

Promising extensions are richer information rather than more model capacity: market/sector indices, crude and
currency features, news and earnings-call sentiment, and cross-sectional models across many stocks. On the
evaluation side, walk-forward retraining over longer histories, calibrated probabilities and conformal
prediction intervals would give a better picture of uncertainty, and a transaction-cost-aware backtest would
test whether any statistical edge survives in practice. The return and direction formulations have now been
tested; the next step is to combine them with fundamentals and macro context, and to validate on multiple
assets before drawing general conclusions.
