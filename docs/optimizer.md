# Optimizer Module Documentation

## Overview

The Optimizer module implements inventory management simulation using a reorder point policy with safety stock. It simulates period-by-period inventory flow, automatically placing replenishment orders when inventory falls below calculated thresholds, and tracking shipments with realistic lead time delays.

The module uses classic inventory control theory to balance two competing objectives:
- **Minimize stockouts**: Maintain sufficient inventory to meet demand
- **Minimize holding costs**: Avoid excessive inventory levels

## Inventory Policy

The Optimizer implements a **(s, S) reorder point policy** with the following logic:

1. **Safety Stock Calculation**: Buffer inventory to protect against demand uncertainty
   - Based on demand variability (standard deviation)
   - Scaled by target service level (e.g., 95% = 1.65 z-score)
   - Adjusted for lead time duration

2. **Reorder Point Determination**: Inventory level that triggers replenishment
   - Expected demand during lead time + safety stock
   - When inventory falls below this point, place an order

3. **Periodic Review**: Check inventory at regular intervals
   - Review period = 1: continuous review (check every period)
   - Review period > 1: periodic review (check every N periods)

4. **Order Quantity**: Bring inventory back to reorder point
   - Order quantity = reorder point - current inventory

5. **Lead Time Tracking**: Orders arrive after specified delay
   - Orders placed in period t arrive in period t + lead_time
   - Multiple orders can be in transit simultaneously

## Class Parameters

### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `primary_keys` | List[str] | Required | Grouping columns (e.g., ['store', 'product']) |
| `date_col` | str | 'date' | Name of the date/time column |
| `target_col` | str | 'sales' | Name of the demand/sales column |
| `inv_col` | str | 'inventory' | Name of the initial inventory column |
| `planning_horizon` | int | 8 | Number of periods to simulate |
| `service_level` | float | 0.95 | Target service level (0.5 to 0.999) |
| `review_period` | int | 1 | Frequency of inventory reviews |
| `lead_time` | int | 2 | Periods between order placement and receipt |

### Parameter Impact

**Service Level**: Higher values increase safety stock and reorder points
- 0.90 (90%) → z-score = 1.28 → Lower safety stock
- 0.95 (95%) → z-score = 1.65 → Moderate safety stock
- 0.99 (99%) → z-score = 2.33 → Higher safety stock

**Lead Time**: Longer lead times increase safety stock and reorder points
- Lead time = 1 → sqrt(1) = 1.0x multiplier
- Lead time = 4 → sqrt(4) = 2.0x multiplier
- Lead time = 9 → sqrt(9) = 3.0x multiplier

**Review Period**: Affects order frequency
- Review period = 1 → Check every period (continuous review)
- Review period = 7 → Check weekly (periodic review)

## Methods

### calculate_safety_stock(demand_series)

Calculates safety stock based on demand variability and service level.

**Formula**: `safety_stock = z_score × std_dev × sqrt(lead_time)`

**Parameters**:
- `demand_series` (pd.Series): Historical demand values

**Returns**: float - Safety stock quantity

**Example**:
```python
demand = pd.Series([100, 110, 95, 105, 98, 102, 108])
safety_stock = optimizer.calculate_safety_stock(demand)
# Returns: ~15.8 (for service_level=0.95, lead_time=2)
```

### calculate_reorder_point(avg_demand, safety_stock)

Calculates the reorder point for triggering replenishment orders.

**Formula**: `reorder_point = (avg_demand × lead_time) + safety_stock`

**Parameters**:
- `avg_demand` (float): Average demand per period
- `safety_stock` (float): Safety stock quantity

**Returns**: float - Reorder point quantity

**Example**:
```python
reorder_point = optimizer.calculate_reorder_point(
    avg_demand=100,
    safety_stock=15.8
)
# Returns: 215.8 (for lead_time=2)
```

### simulate(df)

Simulates inventory flow over the planning horizon.

**Parameters**:
- `df` (pd.DataFrame): Forecast data with columns:
  - Primary keys (e.g., 'store', 'product')
  - 'date': Date/time column
  - 'prediction': Forecasted demand
  - 'sample': 'train' or 'test' indicator

**Returns**: pd.DataFrame with additional columns:
- `safety_stock`: Calculated safety stock level
- `reorder_point`: Calculated reorder point
- `inventory`: End-of-period inventory level
- `order`: Quantity ordered in this period
- `shipment`: Quantity received in this period

**Example**:
```python
# After forecasting
forecast_df = forecaster.predict(data)

# Run inventory simulation
optimizer = Optimizer(
    primary_keys=['store', 'product'],
    planning_horizon=12,
    service_level=0.95,
    lead_time=2
)
inventory_plan = optimizer.simulate(forecast_df)
```

## Simulation Logic Flow

```
For each store-product combination:
  1. Calculate safety stock from historical demand variability
  2. Calculate reorder point = (avg_demand × lead_time) + safety_stock
  3. Initialize inventory from last historical period
  
  For each forecast period:
    a. Process arriving shipments (orders placed lead_time periods ago)
    b. Add shipments to inventory
    c. Deduct forecasted demand from inventory
    d. If review period AND inventory < reorder_point:
       - Place order = reorder_point - current_inventory
       - Add order to transit queue (arrives in lead_time periods)
    e. Record: inventory, order, shipment
```

## Example Output

### Input: Forecast Data
```
store  product  date        sales  sample  prediction
A      X        2024-10-01  100    train   NaN
A      X        2024-10-08  105    train   NaN
A      X        2024-10-15  98     train   NaN
A      X        2024-10-22  NaN    test    102.5
A      X        2024-10-29  NaN    test    103.8
A      X        2024-11-05  NaN    test    105.2
```

### Output: Inventory Simulation
```
store  product  date        prediction  safety_stock  reorder_point  inventory  order  shipment
A      X        2024-10-01  NaN         15.8          215.8          NaN        0.0    0.0
A      X        2024-10-08  NaN         15.8          215.8          NaN        0.0    0.0
A      X        2024-10-15  NaN         15.8          215.8          NaN        0.0    0.0
A      X        2024-10-22  102.5       15.8          215.8          297.5      0.0    0.0
A      X        2024-10-29  103.8       15.8          215.8          193.7      215.8  0.0
A      X        2024-11-05  105.2       15.8          215.8          304.3      0.0    215.8
```

**Interpretation**:
- **Period 2024-10-22**: Starting inventory = 400, demand = 102.5, ending = 297.5
- **Period 2024-10-29**: Inventory = 193.7 < reorder point (215.8), so order 215.8 units
- **Period 2024-11-05**: Shipment of 215.8 arrives (2 periods after order), inventory = 304.3

## Key Features

### Demand Uncertainty Handling
- Calculates safety stock based on historical demand variability
- Uses normal distribution assumption (appropriate for most demand patterns)
- Automatically adjusts for zero-variance edge cases

### Lead Time Management
- Tracks multiple orders in transit simultaneously
- Ensures orders arrive exactly after specified lead time
- Prevents double-ordering during lead time periods

### Flexible Review Policies
- Supports continuous review (review_period=1)
- Supports periodic review (review_period>1)
- Reduces order frequency for operational efficiency

### Stockout Visibility
- Allows negative inventory to indicate stockouts
- Tracks magnitude and duration of stockouts
- Enables service level analysis

## Usage Example

```python
from optimizer.optimizer import Optimizer
from forecaster.forecaster import Forecaster
import pandas as pd

# Generate or load data
data = pd.read_csv('sales_data.csv')

# Step 1: Forecast demand
forecaster = Forecaster(
    primary_keys=['store', 'product'],
    forecast_horizon=12
)
forecaster.fit(data[data['date'] < '2024-10-01'])
forecasts = forecaster.predict(data)

# Step 2: Simulate inventory
optimizer = Optimizer(
    primary_keys=['store', 'product'],
    planning_horizon=12,
    service_level=0.95,
    review_period=1,
    lead_time=2
)
inventory_plan = optimizer.simulate(forecasts)

# Step 3: Analyze results
print("Inventory Summary:")
print(inventory_plan[inventory_plan['sample'] == 'test'].groupby('store').agg({
    'inventory': 'mean',
    'order': 'sum',
    'shipment': 'sum'
}))

# Check for stockouts
stockouts = inventory_plan[inventory_plan['inventory'] < 0]
print(f"\nStockout periods: {len(stockouts)}")
```

## Best Practices

1. **Service Level Selection**:
   - Use 0.90-0.95 for standard products
   - Use 0.95-0.99 for critical/high-value items
   - Consider cost of stockout vs. holding cost

2. **Lead Time Accuracy**:
   - Use realistic lead times based on supplier performance
   - Include safety margin for variability
   - Consider transportation and processing time

3. **Review Period**:
   - Use continuous review (1) for high-value items
   - Use periodic review (7) for operational efficiency
   - Align with ordering cycles and delivery schedules

4. **Planning Horizon**:
   - Set to at least 2-3x lead time for meaningful results
   - Extend for seasonal planning (26-52 weeks)
   - Balance detail vs. forecast accuracy degradation

5. **Historical Data**:
   - Use at least 26 weeks of history for stable estimates
   - More data improves safety stock calculations
   - Consider seasonality and trend in demand patterns

## Limitations

- Assumes normal distribution for demand (may not fit all patterns)
- Does not consider capacity constraints or budget limits
- Uses simple order-up-to policy (not optimal for all scenarios)
- Does not account for batch sizes or minimum order quantities
- Assumes constant lead time (no variability)

## Future Enhancements

- Support for stochastic lead times
- Multi-echelon inventory optimization
- Batch size and MOQ constraints
- Budget and capacity constraints
- Alternative policies (EOQ, base stock)
- Service level measurement and reporting
