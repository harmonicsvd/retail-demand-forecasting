# Retail Demand Forecasting: Problem Statement and Dataset

## Business question

A store planner has to decide every week how much stock of each product to order. Ordering too much ties up cash and risks waste. Ordering too little causes stockouts and lost sales. Both decisions depend on a demand forecast.

**Question:** For one store's food category, how accurately can we forecast daily unit sales per product for the next 28 days, and how large is the remaining forecast error that a planner would need to buffer against?

## Objective

Forecast daily unit sales for every product in one store's FOODS category over a 28-day horizon, and measure the forecast error in a form that could later feed a safety-stock calculation.

## Scope

| Item | Choice |
|---|---|
| Granularity | product x day |
| Store | one store (suggested: `CA_1`; can be changed) |
| Category | `FOODS` only |
| Horizon | 28 days ahead |
| Task type | Supervised time-series forecasting (point forecasts) |

## Dataset

**Name:** M5 Forecasting - Accuracy (Walmart unit sales, released by the Makridakis Open Forecasting Center)
**Source:** https://www.kaggle.com/c/m5-forecasting-accuracy/data (requires a Kaggle account and accepting the competition rules to download)

**Coverage:** 3,049 products, 3 categories (FOODS, HOBBIES, HOUSEHOLD), 7 departments, 10 stores in 3 US states (CA, TX, WI). Daily data from 2011-01-29 to 2016-06-19.

**Files used:**

| File | Contents |
|---|---|
| `sales_train_evaluation.csv` | Daily unit sales per product per store, columns `d_1` to `d_1941`, plus item, department, category, store, and state identifiers |
| `calendar.csv` | Date for each day index, weekday, month, year, events (holidays etc.), and SNAP food-assistance flags per state |
| `sell_prices.csv` | Weekly sell price per store and item |

`sample_submission.csv` and the Kaggle leaderboard are not used. The true competition test labels are not public, so evaluation uses a local holdout (below).

## Data split (time-based only)

- **Train:** `d_1` to `d_1913`
- **Test (holdout):** `d_1914` to `d_1941` (the final 28 days)
- Never shuffle. Random splits leak future information into training.
- If time allows, repeat the evaluation over several earlier rolling 28-day windows to check the result is stable.

## Baselines to beat

1. **Seasonal naive:** the value from the same weekday one week earlier
2. **Moving average:** mean of the last N days (test a few values of N)

A model only earns its complexity if it beats both.

## Models to compare

- The two baselines above
- Gradient boosting (LightGBM) using lagged sales, rolling statistics, price, and calendar/event features

## Evaluation

- **Primary metric: WAPE** (sum of absolute errors divided by sum of actual sales)
- **Secondary metrics:** RMSSE or MASE, and forecast bias (mean signed error)
- **Do not use MAPE as the main metric.** Many products have days with zero sales, which makes MAPE undefined or unstable.
- Report errors overall and by department, and separately for low-volume vs. high-volume products, since intermittent demand behaves differently.

## Success criterion

The best model beats the seasonal naive baseline on WAPE by a clear, stated margin on the holdout, and the residuals are stored per product so the error spread can be used for safety stock in a later step.

## Out of scope for now

Price optimization, markdown decisions, inventory policy, assortment optimization, multi-store hierarchies, and dashboards. These come after the forecasting baseline works.
