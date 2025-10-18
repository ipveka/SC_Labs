# Auxiliar Module Documentation

## Overview

The Auxiliar module provides utilities for generating synthetic supply chain data. It creates realistic datasets with sales patterns, inventory levels, and customer information that can be used to demonstrate and test the SC Labs supply chain optimization system without requiring real business data.

The module generates data with realistic characteristics including:
- **Trend**: Gradual increase or decrease in demand over time
- **Seasonality**: Cyclical patterns that repeat over time (quarterly patterns)
- **Random noise**: Natural variability in demand

This synthetic data is ideal for proof-of-concept demonstrations, testing, and educational purposes.

## Module Purpose

The primary purpose of the Auxiliar module is to:
1. Generate reproducible synthetic datasets for testing and demonstration
2. Simulate realistic supply chain scenarios with multiple stores and products
3. Provide data in a format compatible with downstream modules (Forecaster, Optimizer, Router)
4. Enable quick prototyping without requiring access to real business data

## Functions

### `generate_data()`

Generates synthetic supply chain data with realistic demand patterns.

#### Function Signature

```python
def generate_data(
    n_stores: int = 3,
    n_products: int = 2,
    n_weeks: int = 52,
    start_date: str = '2024-01-01',
    seed: Optional[int] = 42
) -> pd.DataFrame
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_stores` | int | 3 | Number of stores to generate. Stores are labeled with uppercase letters (A, B, C, ...) |
| `n_products` | int | 2 | Number of products to generate. Products are labeled with uppercase letters (X, Y, Z, ...) |
| `n_weeks` | int | 52 | Number of weeks of historical data to generate. Determines the length of the time series |
| `start_date` | str | '2024-01-01' | Starting date for the time series in 'YYYY-MM-DD' format |
| `seed` | int or None | 42 | Random seed for reproducibility. Set to None for non-deterministic generation |

#### Return Value

Returns a pandas DataFrame with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `store` | str | Store identifier (e.g., 'A', 'B', 'C') |
| `product` | str | Product identifier (e.g., 'X', 'Y', 'Z') |
| `date` | datetime64 | Date of observation (weekly frequency) |
| `sales` | float | Units sold (non-negative) |
| `inventory` | float | Current inventory level (200-500 range) |
| `customer_id` | str | Customer identifier (format: CUST_XXXX) |
| `destination` | str | Postal code (5-digit string) |

The DataFrame will have `n_stores × n_products × n_weeks` rows.

#### Example Usage

```python
from auxiliar.auxiliar import generate_data

# Generate data for 3 stores, 2 products, 52 weeks
data = generate_data(n_stores=3, n_products=2, n_weeks=52)

print(f"Generated {len(data)} rows")
print(f"Date range: {data['date'].min()} to {data['date'].max()}")
print(f"Stores: {data['store'].unique()}")
print(f"Products: {data['product'].unique()}")
```

#### Example Output

```
  store product       date       sales   inventory customer_id destination
0     A       A 2024-01-07  115.896346  291.272673   CUST_5658       83969
1     A       A 2024-01-14  140.247001  357.426929   CUST_2899       53001
2     A       A 2024-01-21  122.980537  329.583506   CUST_8734       86552
3     A       A 2024-01-28  128.301355  287.368742   CUST_2267       33897
4     A       A 2024-02-04  156.273821  383.555868   CUST_2528       78148
...
```

## Data Generation Logic

The `generate_data()` function creates realistic sales patterns by combining multiple components:

### 1. Base Demand Level

Each store-product combination is assigned a random base demand level between 50 and 200 units. This represents the average weekly sales for that combination.

```python
base_demand = random value between 50 and 200
```

### 2. Trend Component

A linear trend is added to simulate gradual growth or decline in demand over time. The trend slope is randomly chosen between 0.5 and 2.0 units per week.

```python
trend = time_index × random_slope
where random_slope is between 0.5 and 2.0
```

### 3. Seasonality Component

A sinusoidal pattern with a 13-week period creates quarterly seasonality, simulating cyclical demand patterns (e.g., seasonal products, promotional cycles).

```python
seasonality = 20 × sin(2π × time_index / 13)
```

This creates peaks and troughs that repeat approximately every quarter.

### 4. Random Noise

Gaussian noise with mean 0 and standard deviation 15 is added to simulate natural variability and unpredictability in demand.

```python
noise = normal distribution(mean=0, std=15)
```

### 5. Final Sales Calculation

All components are combined and constrained to be non-negative:

```python
sales = max(0, base_demand + trend + seasonality + noise)
```

### 6. Additional Data

- **Inventory**: Random values between 200 and 500 units for each observation
- **Customer IDs**: Random 4-digit numbers in format "CUST_XXXX"
- **Postal Codes**: Random 5-digit strings representing delivery destinations

## Reproducibility

The `seed` parameter ensures reproducible data generation. Using the same seed value will always produce the same dataset, which is crucial for:
- Testing and debugging
- Comparing different algorithms on the same data
- Demonstrating consistent results

```python
# These will generate identical datasets
data1 = generate_data(seed=42)
data2 = generate_data(seed=42)
assert data1.equals(data2)  # True
```

## Integration with Other Modules

The generated data is designed to work seamlessly with other SC Labs modules:

- **Forecaster Module**: Uses the `store`, `product`, `date`, and `sales` columns for demand forecasting
- **Optimizer Module**: Uses forecasts along with `inventory` column for inventory simulation
- **Router Module**: Uses `customer_id` and `destination` columns for delivery routing

## Notes and Limitations

1. **Synthetic Nature**: This data is artificially generated and may not capture all complexities of real-world supply chains
2. **Simple Patterns**: The trend and seasonality patterns are simplified for demonstration purposes
3. **Independent Series**: Each store-product combination is generated independently without cross-correlations
4. **Fixed Frequency**: Data is generated at weekly frequency only
5. **No Constraints**: The generation doesn't enforce business constraints like warehouse capacity or budget limits

## Future Enhancements

Potential improvements to the Auxiliar module:
- Support for daily or monthly frequencies
- Cross-correlation between products (e.g., substitute or complementary products)
- More complex seasonality patterns (multiple seasonal cycles)
- Special events and promotions
- Supply disruptions and stockouts
- Price elasticity effects
