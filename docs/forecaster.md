# Forecaster Module Documentation

## Overview

The Forecaster module provides time series demand forecasting capabilities for supply chain optimization. It uses the GluonTS library with a SimpleFeedForwardEstimator to train neural network models on historical sales data and generate future demand predictions.

### Key Features

- **Multi-series forecasting**: Supports forecasting multiple time series simultaneously (e.g., different store-product combinations)
- **Automatic data preparation**: Handles missing dates, aggregates duplicates, and fills gaps
- **GluonTS integration**: Leverages state-of-the-art deep learning forecasting models
- **Flexible configuration**: Customizable forecast horizon, frequency, and grouping keys

### GluonTS Usage

GluonTS is a probabilistic time series modeling toolkit developed by Amazon. This module uses the `SimpleFeedForwardEstimator`, which implements a simple feed-forward neural network suitable for quick training and reasonable accuracy on small to medium datasets.

**Model Configuration:**
- **Estimator**: SimpleFeedForwardEstimator
- **Training epochs**: 10
- **Learning rate**: 1e-3
- **Prediction type**: Point forecasts (mean values)

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

### Attributes

- `model`: Trained GluonTS predictor (None until `fit()` is called)

## Methods

### prepare_data()

Prepares and cleans time series data for forecasting by aggregating, ensuring continuous date ranges, and filling missing values.

**Signature:**
```python
def prepare_data(df: pd.DataFrame) -> pd.DataFrame
```

**Input:**
- `df`: DataFrame with columns matching `primary_keys`, `date_col`, and `target_col`

**Output:**
- Cleaned DataFrame with continuous date range and no missing values

**Processing Steps:**
1. Validates required columns exist
2. Converts date column to datetime type
3. Aggregates data by primary keys and date (handles duplicates)
4. Creates continuous date range for each time series
5. Fills missing values using forward fill, then backward fill
6. Returns sorted, cleaned DataFrame

**Example:**
```python
import pandas as pd

df = pd.DataFrame({
    'store': ['A', 'A', 'A', 'A'],
    'product': ['X', 'X', 'X', 'X'],
    'date': ['2024-01-01', '2024-01-08', '2024-01-22', '2024-01-29'],  # Missing 2024-01-15
    'sales': [100, 150, 120, 180]
})

clean_df = forecaster.prepare_data(df)
# Result will have 5 rows with 2024-01-15 filled in
```

### fit()

Trains the forecasting model on historical time series data.

**Signature:**
```python
def fit(df: pd.DataFrame) -> None
```

**Input:**
- `df`: Training DataFrame with historical data

**Output:**
- None (stores trained model in `self.model`)

**Processing Steps:**
1. Validates input data
2. Warns if time series have insufficient history (< 2x forecast horizon)
3. Prepares data using `prepare_data()`
4. Converts DataFrame to GluonTS PandasDataset format
5. Creates item IDs by combining primary keys
6. Initializes SimpleFeedForwardEstimator
7. Trains model and stores in `self.model`

**Example:**
```python
train_df = pd.DataFrame({
    'store': ['A'] * 52 + ['B'] * 52,
    'product': ['X'] * 52 + ['X'] * 52,
    'date': pd.date_range('2023-01-01', periods=52, freq='W').tolist() * 2,
    'sales': np.random.randint(50, 200, 104)
})

forecaster.fit(train_df)
# Model is now trained and ready for predictions
```

**Recommendations:**
- Provide at least 2x forecast_horizon periods of history per time series
- More history generally improves forecast accuracy
- For weekly data with 4-week horizon, aim for at least 8-12 weeks of history

### predict()

Generates forecasts for future periods using the trained model.

**Signature:**
```python
def predict(df: pd.DataFrame) -> pd.DataFrame
```

**Input:**
- `df`: DataFrame containing historical data (same format as training data)

**Output:**
- DataFrame with both historical and forecasted data

**Output Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `primary_keys` | varies | Grouping columns (e.g., store, product) |
| `date_col` | datetime | Date/timestamp |
| `target_col` | float | Actual historical values (NaN for forecast periods) |
| `sample` | str | `'train'` for historical, `'test'` for forecast |
| `prediction` | float | NaN for historical, forecasted values for future |

**Processing Steps:**
1. Validates model has been trained
2. Prepares input data
3. Creates historical records with `sample='train'` and `prediction=NaN`
4. For each time series (primary key combination):
   - Converts to GluonTS format
   - Generates forecast using trained model
   - Extracts mean prediction values
   - Creates future date range
   - Builds forecast records with `sample='test'`
5. Combines historical and forecast data
6. Returns unified DataFrame

**Example:**
```python
# After training the model
forecasts = forecaster.predict(train_df)

# Separate historical and forecast data
historical = forecasts[forecasts['sample'] == 'train']
future = forecasts[forecasts['sample'] == 'test']

print(future[['store', 'product', 'date', 'prediction']])
```

**Example Output:**

```
   store product       date  prediction
0      A       X 2024-01-01         NaN
1      A       X 2024-01-08         NaN
...
50     A       X 2024-12-25         NaN
51     A       X 2025-01-01       145.3
52     A       X 2025-01-08       152.7
53     A       X 2025-01-15       148.9
54     A       X 2025-01-22       151.2
```

## Complete Usage Example

```python
import pandas as pd
import numpy as np
from forecaster.forecaster import Forecaster

# Generate sample data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=52, freq='W')
data = []

for store in ['A', 'B', 'C']:
    for product in ['X', 'Y']:
        for date in dates:
            sales = np.random.randint(50, 200)
            data.append({
                'store': store,
                'product': product,
                'date': date,
                'sales': sales
            })

df = pd.DataFrame(data)

# Initialize forecaster
forecaster = Forecaster(
    primary_keys=['store', 'product'],
    date_col='date',
    target_col='sales',
    frequency='W',
    forecast_horizon=4
)

# Split data into train/test
train_df = df[df['date'] < '2024-10-01']
test_df = df[df['date'] >= '2024-10-01']

# Train model
print("Training model...")
forecaster.fit(train_df)

# Generate forecasts
print("Generating forecasts...")
forecasts = forecaster.predict(train_df)

# View forecast results
forecast_only = forecasts[forecasts['sample'] == 'test']
print("\nForecasts:")
print(forecast_only.head(12))

# Calculate accuracy metrics (if actuals available)
# This would require comparing predictions with actual test data
```

## Error Handling

The Forecaster module includes comprehensive error handling:

### Common Errors

1. **Empty DataFrame**
   ```
   ValueError: Input DataFrame is empty
   ```
   - Ensure your DataFrame contains data before calling methods

2. **Missing Columns**
   ```
   ValueError: Missing required columns: ['date', 'sales']
   ```
   - Verify all required columns exist in your DataFrame

3. **Model Not Trained**
   ```
   RuntimeError: Model has not been trained. Call fit() first.
   ```
   - Call `fit()` before calling `predict()`

4. **Training Failure**
   ```
   RuntimeError: Model training failed: [error details]
   ```
   - Check data quality, ensure sufficient history, verify GluonTS installation

### Warnings

- **Insufficient History**: Warns when time series have fewer than 2x forecast_horizon periods
- **Forecast Generation Failure**: Prints warning but continues with other time series

## Performance Considerations

### Training Time
- **Small datasets** (< 10 time series, < 100 periods): < 1 minute
- **Medium datasets** (10-50 time series, 100-500 periods): 1-5 minutes
- **Large datasets** (> 50 time series, > 500 periods): 5-15 minutes

### Memory Usage
- Approximately 100-500 MB for typical supply chain datasets
- Scales with number of time series and history length

### Optimization Tips
1. Use appropriate `forecast_horizon` (don't forecast too far ahead)
2. Limit training epochs if speed is critical
3. Consider training on a subset of time series for initial testing
4. Use GPU acceleration if available (requires PyTorch GPU setup)

## Integration with Other Modules

The Forecaster module outputs are designed to integrate seamlessly with downstream modules:

### Optimizer Module
```python
# Forecaster output feeds directly into Optimizer
forecasts = forecaster.predict(train_df)
optimizer = Optimizer(primary_keys=['store', 'product'])
inventory_plan = optimizer.simulate(forecasts)
```

### Router Module
```python
# Forecaster output can be used for delivery planning
forecasts = forecaster.predict(train_df)
router = Router(primary_keys=['store', 'product'])
deliveries = router.distribute_demand(forecasts)
```

## Limitations and Future Enhancements

### Current Limitations
- Uses simple feed-forward network (not state-of-the-art)
- Point forecasts only (no prediction intervals)
- Limited hyperparameter tuning
- No automatic model selection

### Potential Enhancements
1. **Advanced Models**: Support for DeepAR, Transformer, or Prophet models
2. **Probabilistic Forecasts**: Generate prediction intervals and quantiles
3. **Hyperparameter Tuning**: Automatic optimization of model parameters
4. **Feature Engineering**: Support for external regressors (holidays, promotions)
5. **Model Evaluation**: Built-in accuracy metrics (RMSE, MAE, MAPE)
6. **Ensemble Methods**: Combine multiple models for better accuracy

## References

- [GluonTS Documentation](https://ts.gluon.ai/)
- [SimpleFeedForwardEstimator API](https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.model.simple_feedforward.html)
- [Pandas Frequency Strings](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases)
