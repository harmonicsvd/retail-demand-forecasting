# Retail Demand Forecasting

Retail demand forecasting project using the M5 Forecasting - Accuracy dataset from Kaggle. The goal is to forecast daily unit sales for products in the FOODS category at a single Walmart store over a 28-day horizon.

## Problem Statement

A store planner must decide weekly how much stock of each product to order. Ordering too much ties up cash and risks waste; ordering too little causes stockouts and lost sales. This project builds a demand forecast to support inventory planning decisions.

**Objective**: Forecast daily unit sales for every product in one store's FOODS category over a 28-day horizon and measure forecast error to inform safety stock calculations.

## Dataset

**Source**: [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/data) (Walmart unit sales data)

**Coverage**: 
- 3,049 products across 3 categories (FOODS, HOBBIES, HOUSEHOLD)
- 7 departments, 10 stores in 3 US states (CA, TX, WI)
- Daily data from 2011-01-29 to 2016-06-19

**Project Scope**:
- **Category**: FOODS only
- **Store**: CA_1 (California Store 1)
- **Products**: 1,437 products (3 departments: FOODS_1, FOODS_2, FOODS_3)
- **Granularity**: product × day
- **Forecast Horizon**: 28 days

## Data Split

Time-based split (no shuffling to avoid data leakage):
- **Training**: d_1 to d_1913 (2011-01-29 to 2016-04-24) - 1,913 days
- **Test (Holdout)**: d_1914 to d_1941 (2016-04-25 to 2016-05-22) - 28 days

## Project Structure

```
product-forcasting/
├── data-prep-eda.ipynb              # Data loading, filtering, EDA, train/test split
├── seasonal_naive.ipynb             # Seasonal naive baseline (7-day lag)
├── moving_average_baseline.ipynb    # Moving average baselines (7, 14, 21, 28-day windows)
├── lightgbm_model.ipynb             # LightGBM model with feature engineering
├── train_data.csv                   # Preprocessed training data
├── test_data.csv                    # Preprocessed test data
├── lightgbm_retail_model.txt       # Trained LightGBM model
├── requirements.txt                 # Python dependencies
├── retail-forecasting-problem-statement.md  # Detailed problem specification
└── m5-forecasting-accuracy/         # Raw Kaggle dataset
```

## Notebooks

### 1. Data Preparation & EDA (`data-prep-eda.ipynb`)
- Loads raw M5 dataset files (sales, calendar, prices)
- Filters to FOODS category and CA_1 store (1,437 products)
- Exploratory data analysis:
  - Overall sales statistics: Mean 1.96 units/day, Median 0 units/day
  - 56.9% of product-days have zero sales (intermittent demand)
  - Department breakdown: FOODS_3 (823 products), FOODS_2 (398), FOODS_1 (216)
- Creates train/test split based on time
- Saves processed data to CSV files

### 2. Seasonal Naive Baseline (`seasonal_naive.ipynb`)
- **Method**: Forecast using sales from 7 days ago (same weekday last week)
- **Result**: WAPE = 94.4%
- Simple baseline that captures weekly seasonality

### 3. Moving Average Baseline (`moving_average_baseline.ipynb`)
- **Method**: Forecast using average of last N days
- **Results**:
  - 7-day MA: WAPE = 93.2%
  - 14-day MA: WAPE = 84.6%
  - 21-day MA: WAPE = 76.2%
  - **28-day MA: WAPE = 68.5%** (best baseline)
- Shows that longer windows perform better for this data

### 4. LightGBM Model (`lightgbm_model.ipynb`)
- **Features**:
  - **Lag features**: Sales from 7, 14, 28, and 365 days ago
  - **Rolling statistics**: 7, 14, 28-day rolling means and standard deviations
  - **Time features**: Cyclical encoding for day-of-week, month, day-of-year
  - **Calendar features**: Events, SNAP (food assistance) indicators
  - **Categorical features**: Department, weekday, month
- **Model**: LightGBM gradient boosting with 24 features
- **Training**: 2.18M training samples, 40K validation samples
- **Hyperparameters**: 
  - Learning rate: 0.015
  - Num leaves: 31
  - Feature/bagging fraction: 0.8
  - Early stopping: 100 rounds
- **Result**: WAPE = 53.3% (beats best baseline by 22.2%)

## Results Summary

| Model | WAPE | Improvement vs Baseline |
|-------|------|------------------------|
| Seasonal Naive (7-day lag) | 94.4% | - |
| 7-day Moving Average | 93.2% | +1.2% |
| 14-day Moving Average | 84.6% | +9.8% |
| 21-day Moving Average | 76.2% | +18.2% |
| **28-day Moving Average** | **68.5%** | **+25.9%** (best baseline) |
| **LightGBM** | **53.3%** | **+22.2% vs best baseline** |

## Key Findings

1. **Intermittent Demand**: 56.9% of product-days have zero sales, making MAPE unstable (WAPE used instead)
2. **Feature Importance**: Lag features and rolling statistics are critical for capturing temporal patterns
3. **Model Performance**: LightGBM significantly outperforms simple baselines, demonstrating the value of feature engineering
4. **Department Variation**: Different departments show different sales patterns and zero-sale percentages

## Evaluation Metrics

- **Primary Metric**: WAPE (Weighted Absolute Percentage Error) = sum(|actual - forecast|) / sum(actual)
  - Avoids issues with zero sales that make MAPE undefined
- **Secondary Metrics**: MAE (Mean Absolute Error), forecast bias

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Data Preparation
1. Download the M5 Forecasting - Accuracy dataset from Kaggle
2. Extract files to `m5-forecasting-accuracy/` directory
3. Run `data-prep-eda.ipynb` to filter and prepare train/test data

### Training Models
1. Run baseline notebooks in order:
   - `seasonal_naive.ipynb`
   - `moving_average_baseline.ipynb`
2. Run `lightgbm_model.ipynb` to train the final model
3. The trained model is saved as `lightgbm_retail_model.txt`

## Future Work

- Incorporate price features (sell prices available in dataset)
- Try ensemble methods combining multiple models
- Forecast by department or product hierarchies
- Add uncertainty quantification for safety stock calculations
- Extend to multiple stores or all categories

## References

- M5 Forecasting Competition: https://www.kaggle.com/c/m5-forecasting-accuracy
- Makridakis Open Forecasting Center (MOFC)
