# Forecaster Module Documentation

## Overview

The Forecaster module provides time series demand forecasting capabilities for supply chain optimization. It uses LightGBM gradient boosting with automated feature engineering to train models on historical sales data and generate future demand predictions.

### Key Features

- **Multi-series forecasting**: Supports forecasting multiple time series simultaneously (e.g., different store-product combinations)
- **Automated feature engineering**: Creates temporal, lag, and rolling window features automatically
- **No data leakage**: Proper train/test splits and feature engineering to prevent information leakage
- **LightGBM integration**: Fast, efficient gradient boosting with categorical feature support
- **Recursive forecasting**: Multi-step ahead predictions using previous forecasts as inputs
- **Flexible configuration**: Customizable forecast horizon, frequency, and grouping keys

### Feature Engineering

The module automatically generates the following features:

**Temporal Features:**
- `day_of_week`: Day of the week (0-6)
- `week_of_year`: Week number in the year (1-52)
- `month`: Month (1-12)
- `quarter`: Quarter (1-4)
- `year`: Year
- `day_of_month`: Day of the month (1-31)

**Lag Features:**
- `lag_1`, `lag_2`, `lag_3`, `lag_4`: Previous 1-4 period values

**Rolling Window Features (windows: 3, 4, 8):**
- `rolling_mean_X`: Rolling mean over X periods
- `rolling_std_X`: Rolling standard deviation over X periods
- `rolling_min_X`: Rolling minimum over X periods
- `rolling_max_X`: Rolling maximum over X periods

**Categorical Features:**
- All primary keys (e.g., store, product) are encoded as categorical features

### Data Leakage Prevention

The module implements several safeguards to prevent data leakage:

1. **Lag features**: Use `shift()` to only access past values
2. **Rolling features**: Use `shift(1)` before rolling to ensure only past data is used
3. **Train/test split**: Chronological split with 80/20 ratio for validation
4. **Recursive forecasting**: Each prediction uses only known historical values and previous predictions

## Class: Forecaster

### Initialization

```python
from forecaster.forecaster import Forecaster

forecaster = Forecaster(
    primary_keys=['store', 'product'],
    date_col='date',
    target_col='sales',
    frequency='W',
    forecast_horizon=4
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `primary_keys` | `List[str]` | Required | Column names that uniquely identify each time series (e.g., `['store', 'product']`) |
| `date_col` | `str` | `'date'` | Name of the column containing dates/timestamps |
| `target_col` | `str` | `'sales'` | Name of the column containing values to forecast |
| `frequency` | `str` | `'W'` | Pandas frequency string: `'W'` (weekly), `'D'` (daily), `'M'` (monthly), etc. |
| `forecast_horizon` | `int` | `4` | Number of future periods to forecast |
| `verbose` | `bool` | `None` | Show detailed logs (None = use config) |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `lgb.Booster` | Trained LightGBM model (None until `fit()` is called) |
| `feature_cols` | `List[str]` | List of feature column names used for training |
| `categorical_features` | `List[str]` | List of categorical feature names |

## Methods

### fit(df: pd.DataFrame) -> None

Train the LightGBM forecasting model on historical time series data.

**Parameters:**
- `df` (pd.DataFrame): Training DataFrame with columns matching `primary_keys`, `date_col`, and `target_col`

**Raises:**
- `ValueError`: If DataFrame has insufficient data for training
- `RuntimeError`: If model training fails

**Example:**
```python
import pandas as pd
import numpy as np

# Create training data
train_df = pd.DataFrame({
    'store': ['A'] * 30,
    'product': ['X'] * 30,
    'date': pd.date_range('2024-01-01', periods=30, freq='W'),
    'sales': np.random.randint(50, 200, 30)
})

# Train model
forecaster.fit(train_df)
```

**Training Process:**
1. Prepares and cleans data (fills gaps, handles missing values)
2. Engineers features (temporal, lag, rolling)
3. Removes rows with NaN features (due to lags/rolling windows)
4. Splits data chronologically (80% train, 20% validation)
5. Trains LightGBM with early stopping

**Model Parameters:**
- Objective: regression
- Metric: RMSE
- Boosting type: GBDT
- Number of leaves: 31
- Learning rate: 0.05
- Feature fraction: 0.8
- Bagging fraction: 0.8
- Early stopping rounds: 20
- Max boost rounds: 200

### predict(df: pd.DataFrame) -> pd.DataFrame

Generate forecasts for future periods using the trained model.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing historical data (same format as training data)

**Returns:**
- `pd.DataFrame`: DataFrame with columns:
  - All `primary_keys` columns
  - `date_col`: Date/timestamp
  - `target_col`: Actual values (NaN for future periods)
  - `sample`: 'train' for historical data, 'test' for forecasts
  - `prediction`: NaN for historical data, forecasted values for future periods

**Raises:**
- `RuntimeError`: If model has not been trained (`fit()` not called)
- `ValueError`: If input DataFrame is invalid

**Example:**
```python
# Generate forecasts
forecasts = forecaster.predict(train_df)

# Separate historical and forecast data
historical = forecasts[forecasts['sample'] == 'train']
future = forecasts[forecasts['sample'] == 'test']

print(future[['store', 'product', 'date', 'prediction']])
```

**Prediction Process:**
1. Prepares historical data
2. For each time series group:
   - Creates future date range
   - Recursively predicts one step at a time:
     - Engineers features using historical + previous predictions
     - Makes prediction with LightGBM
     - Adds prediction to history for next step
3. Combines historical and forecast data

### prepare_data(df: pd.DataFrame) -> pd.DataFrame

Prepare and clean time series data for forecasting.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame

**Returns:**
- `pd.DataFrame`: Cleaned DataFrame with continuous date range

**Process:**
1. Validates required columns exist
2. Converts date column to datetime
3. Aggregates duplicates by summing
4. Creates continuous date range for each time series
5. Fills missing values with 0

### engineer_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame

Apply all feature engineering steps.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `is_training` (bool): If True, this is training data

**Returns:**
- `pd.DataFrame`: DataFrame with all engineered features

**Process:**
1. Creates temporal features from date column
2. Creates lag features (shift by 1-4 periods)
3. Creates rolling window features (shift by 1, then rolling)

## Usage Examples

### Basic Usage

```python
from forecaster.forecaster import Forecaster
import pandas as pd

# Initialize forecaster
forecaster = Forecaster(
    primary_keys=['store', 'product'],
    date_col='date',
    target_col='sales',
    frequency='W',
    forecast_horizon=4
)

# Load historical data
data = pd.read_csv('sales_data.csv')
data['date'] = pd.to_datetime(data['date'])

# Train model
forecaster.fit(data)

# Generate forecasts
forecasts = forecaster.predict(data)

# Extract future predictions
future = forecasts[forecasts['sample'] == 'test']
print(future)
```

### Multiple Time Series

```python
# Data with multiple stores and products
data = pd.DataFrame({
    'store': ['A', 'A', 'B', 'B'] * 20,
    'product': ['X', 'Y', 'X', 'Y'] * 20,
    'date': pd.date_range('2024-01-01', periods=80, freq='W'),
    'sales': np.random.randint(50, 200, 80)
})

# Forecaster handles multiple series automatically
forecaster = Forecaster(
    primary_keys=['store', 'product'],
    forecast_horizon=8
)

forecaster.fit(data)
forecasts = forecaster.predict(data)

# Forecasts for each store-product combination
for (store, product), group in forecasts.groupby(['store', 'product']):
    future = group[group['sample'] == 'test']
    print(f"\nStore {store}, Product {product}:")
    print(future[['date', 'prediction']])
```

### Custom Configuration

```python
# Daily frequency with longer horizon
forecaster = Forecaster(
    primary_keys=['location', 'sku'],
    date_col='timestamp',
    target_col='demand',
    frequency='D',  # Daily
    forecast_horizon=14,  # 2 weeks
    verbose=True  # Show training progress
)

forecaster.fit(daily_data)
forecasts = forecaster.predict(daily_data)
```

## Best Practices

### Data Requirements

- **Minimum history**: At least 12 periods (or 2× forecast horizon, whichever is larger)
- **Continuous dates**: The module fills gaps, but large gaps may affect quality
- **Sufficient variance**: Time series with zero variance get minimal safety stock

### Feature Engineering

- **Lag features**: Automatically created for 1-4 periods back
- **Rolling windows**: Use windows of 3, 4, and 8 periods
- **No leakage**: All features use only past data (shift before rolling)

### Model Training

- **Validation split**: Last 20% of data used for validation
- **Early stopping**: Stops if validation RMSE doesn't improve for 20 rounds
- **Categorical encoding**: Primary keys automatically encoded as categorical

### Forecasting

- **Recursive approach**: Each prediction uses previous predictions as inputs
- **Non-negative**: Predictions are clipped to be non-negative
- **Uncertainty**: LightGBM provides point forecasts (not probabilistic)

## Performance Considerations

### Training Time

- **Fast**: LightGBM is much faster than deep learning models
- **Scalability**: Handles hundreds of time series efficiently
- **Memory**: Feature engineering creates temporary columns

### Prediction Time

- **Recursive**: Each forecast step requires feature engineering
- **Overhead**: Minimal compared to model training
- **Batch**: All time series predicted in one pass

## Troubleshooting

### Common Issues

**Issue**: "No valid training data after feature engineering"
- **Cause**: Insufficient historical data for lag/rolling features
- **Solution**: Increase training data to at least 12 periods

**Issue**: Predictions are constant
- **Cause**: Insufficient variance in training data or overfitting
- **Solution**: Check data quality, increase training data, or adjust model parameters

**Issue**: "Model has not been trained"
- **Cause**: Calling `predict()` before `fit()`
- **Solution**: Call `fit()` first to train the model

## Configuration

The module reads configuration from `config.yaml`:

```yaml
forecasting:
  num_boost_round: 200
  learning_rate: 0.05
  num_leaves: 31
  early_stopping_rounds: 20
  verbose: false
```

## Dependencies

- `pandas>=1.5.0`: Data manipulation
- `numpy>=1.23.0`: Numerical operations
- `lightgbm>=4.0.0`: Gradient boosting model

## See Also

- [Optimizer Module](optimizer.md): Inventory optimization using forecasts
- [Router Module](router.md): Delivery routing using forecasts
- [Main Pipeline](../README.md): Complete workflow integration
