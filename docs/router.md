# Router Module Documentation

## Overview

The Router module provides functionality for delivery routing and truck assignment in supply chain operations. It takes forecasted demand and distributes it among customers, assigns deliveries to trucks while respecting payload constraints, and optimizes delivery routes using a nearest neighbor heuristic.

The module implements a simple but effective routing approach based on postal code proximity, making it suitable for proof-of-concept demonstrations and educational purposes.

## Key Features

- **Customer Generation**: Create synthetic customer databases with unique IDs and postal codes
- **Demand Distribution**: Split forecasted sales across multiple customers for realistic delivery scenarios
- **Truck Assignment**: Assign deliveries to trucks while respecting maximum payload constraints
- **Route Optimization**: Order delivery stops using a nearest neighbor heuristic to minimize travel distance

## Class: Router

### Initialization

```python
Router(
    primary_keys: List[str],
    date_col: str = 'date',
    target_col: str = 'sales',
    max_payload: int = 10,
    origin: str = '08020'
)
```

**Parameters:**

- `primary_keys`: List of column names to group by (e.g., `['store', 'product']`)
- `date_col`: Name of the date column (default: `'date'`)
- `target_col`: Name of the target variable column (default: `'sales'`)
- `max_payload`: Maximum units that can be loaded on a single truck (default: `10`)
- `origin`: Starting postal code for all delivery routes (default: `'08020'`)

**Example:**

```python
from router.router import Router

router = Router(
    primary_keys=['store', 'product'],
    max_payload=100,
    origin='08020'
)
```

## Methods

### 1. generate_customers()

Generate a database of sample customers with unique IDs and postal codes.

**Signature:**

```python
generate_customers(n_customers: int = 20) -> pd.DataFrame
```

**Parameters:**

- `n_customers`: Number of customers to generate (default: `20`)

**Returns:**

- `pd.DataFrame`: Customer database with columns `['customer_id', 'destination']`

**Example:**

```python
customers = router.generate_customers(n_customers=50)
print(customers.head())
```

**Output:**

| customer_id | destination |
|-------------|-------------|
| CUST_0001   | 45678       |
| CUST_0002   | 23456       |
| CUST_0003   | 67890       |
| CUST_0004   | 12345       |
| CUST_0005   | 89012       |

### 2. distribute_demand()

Distribute forecasted sales among customers for delivery planning.

**Signature:**

```python
distribute_demand(df: pd.DataFrame) -> pd.DataFrame
```

**Parameters:**

- `df`: DataFrame with forecast data containing primary_keys, date_col, and target_col

**Returns:**

- `pd.DataFrame`: Delivery records with columns `[primary_keys, date, sales, customer, destination, units]`

**Logic:**

1. Filters to forecast data (where `sample == 'test'`)
2. For each (store, product, date) combination:
   - Randomly selects 2-5 customers
   - Splits the sales quantity evenly across selected customers
   - Creates individual delivery records

**Example:**

```python
deliveries = router.distribute_demand(forecasts)
print(deliveries.head())
```

**Output:**

| store | product | date       | sales | customer  | destination | units |
|-------|---------|------------|-------|-----------|-------------|-------|
| A     | X       | 2024-10-07 | 150.5 | CUST_0012 | 45678       | 50.17 |
| A     | X       | 2024-10-07 | 150.5 | CUST_0023 | 23456       | 50.17 |
| A     | X       | 2024-10-07 | 150.5 | CUST_0034 | 67890       | 50.17 |
| A     | Y       | 2024-10-07 | 200.3 | CUST_0005 | 12345       | 100.15|
| A     | Y       | 2024-10-07 | 200.3 | CUST_0018 | 89012       | 100.15|

### 3. assign_trucks()

Assign deliveries to trucks while respecting payload constraints.

**Signature:**

```python
assign_trucks(deliveries_df: pd.DataFrame) -> pd.DataFrame
```

**Parameters:**

- `deliveries_df`: DataFrame with delivery records containing `'units'` and `'destination'`

**Returns:**

- `pd.DataFrame`: Updated deliveries DataFrame with `'truck'` column added

**Logic:**

1. Groups deliveries by date
2. Sorts deliveries by destination for locality
3. Assigns deliveries to trucks sequentially
4. Creates a new truck when adding a delivery would exceed `max_payload`
5. Adds truck identifier (e.g., `'truck_1'`, `'truck_2'`)

**Example:**

```python
deliveries_with_trucks = router.assign_trucks(deliveries)
print(deliveries_with_trucks[['customer', 'destination', 'units', 'truck']].head())
```

**Output:**

| customer  | destination | units | truck   |
|-----------|-------------|-------|---------|
| CUST_0023 | 12345       | 50.17 | truck_1 |
| CUST_0012 | 23456       | 50.17 | truck_1 |
| CUST_0034 | 45678       | 50.17 | truck_2 |
| CUST_0005 | 67890       | 100.15| truck_2 |
| CUST_0018 | 89012       | 100.15| truck_3 |

### 4. optimize_routes()

Optimize delivery routes using a nearest neighbor heuristic.

**Signature:**

```python
optimize_routes(deliveries_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]
```

**Parameters:**

- `deliveries_df`: DataFrame with delivery records including `'truck'` and `'destination'`

**Returns:**

- `Tuple[pd.DataFrame, pd.DataFrame]`:
  - Delivery-level data with route information
  - Route summary with columns `[truck, route_order, origin, destinations, total_distance]`

**Logic:**

1. Groups deliveries by truck
2. For each truck:
   - Extracts unique destinations
   - Starts from the origin location
   - Repeatedly selects the nearest unvisited destination (nearest neighbor heuristic)
   - Calculates distance as the numeric difference between postal codes
   - Accumulates total route distance
3. Returns both delivery-level data and route summary

**Example:**

```python
deliveries_final, routes = router.optimize_routes(deliveries_with_trucks)
print(routes)
```

**Route Summary Output:**

| truck   | route_order | origin | destinations              | total_distance |
|---------|-------------|--------|---------------------------|----------------|
| truck_1 | 1, 2        | 08020  | 12345, 23456              | 15436          |
| truck_2 | 1, 2        | 08020  | 45678, 67890              | 59870          |
| truck_3 | 1           | 08020  | 89012                     | 80992          |

## Routing Algorithm: Nearest Neighbor Heuristic

The Router module uses a **nearest neighbor heuristic** for route optimization:

1. **Start**: Begin at the origin postal code
2. **Select**: Find the nearest unvisited destination
3. **Move**: Travel to that destination and mark it as visited
4. **Repeat**: Continue until all destinations are visited
5. **Distance Calculation**: Use the absolute numeric difference between postal codes as a distance proxy

### Algorithm Characteristics

- **Time Complexity**: O(n²) where n is the number of destinations per truck
- **Optimality**: Not guaranteed to find the optimal route, but provides a reasonable approximation
- **Speed**: Fast and suitable for real-time applications
- **Simplicity**: Easy to understand and implement

### Distance Metric

The module uses a simplified distance calculation:

```python
distance = abs(int(postal_code_1) - int(postal_code_2))
```

This assumes postal codes have some geographic ordering. For production use, consider:
- Real coordinate-based distances (using geopy)
- Road network distances (using routing APIs)
- Travel time instead of distance

## Complete Workflow Example

```python
from router.router import Router
import pandas as pd

# Initialize router
router = Router(
    primary_keys=['store', 'product'],
    max_payload=100,
    origin='08020'
)

# Step 1: Generate customers
customers = router.generate_customers(n_customers=50)

# Step 2: Distribute demand to customers
deliveries = router.distribute_demand(forecasts_df)

# Step 3: Assign deliveries to trucks
deliveries_with_trucks = router.assign_trucks(deliveries)

# Step 4: Optimize routes
deliveries_final, routes = router.optimize_routes(deliveries_with_trucks)

# Display results
print("Delivery Summary:")
print(deliveries_final.groupby('truck')['units'].sum())

print("\nRoute Summary:")
print(routes)
```

## Input Requirements

### For distribute_demand()

The input DataFrame should contain:
- Primary key columns (e.g., `store`, `product`)
- Date column
- Target column (e.g., `sales`)
- Optional: `sample` column to filter forecast data
- Optional: `prediction` column with forecasted values

### For assign_trucks()

The input DataFrame should contain:
- All columns from `distribute_demand()` output
- `units` column with delivery quantities
- `destination` column with postal codes

### For optimize_routes()

The input DataFrame should contain:
- All columns from `assign_trucks()` output
- `truck` column with truck assignments

## Configuration Parameters

### max_payload

Controls the maximum number of units that can be loaded on a single truck. Lower values create more trucks with smaller loads, while higher values consolidate deliveries.

**Impact:**
- **Low values** (e.g., 50): More trucks, shorter routes, higher transportation costs
- **High values** (e.g., 500): Fewer trucks, longer routes, better vehicle utilization

### origin

The starting postal code for all delivery routes. Should match the warehouse or distribution center location.

## Limitations and Future Enhancements

### Current Limitations

1. **Distance Metric**: Uses postal code numeric difference, not actual geographic distance
2. **Route Optimization**: Nearest neighbor is a greedy heuristic, not optimal
3. **Single Origin**: All routes start from the same location
4. **No Time Windows**: Doesn't consider delivery time constraints
5. **Uniform Vehicles**: All trucks have the same capacity

### Potential Enhancements

1. **Real Distances**: Integrate geopy or routing APIs for accurate distances
2. **Advanced Optimization**: Use OR-Tools for optimal VRP solutions
3. **Multi-Depot**: Support multiple distribution centers
4. **Time Windows**: Add delivery time constraints
5. **Vehicle Types**: Support different truck capacities and costs
6. **Return Routes**: Optimize return trips to origin
7. **Dynamic Routing**: Real-time route adjustments based on traffic

## Error Handling

The Router module includes error handling for common scenarios:

- **Missing customer database**: Raises `ValueError` if `distribute_demand()` is called before `generate_customers()`
- **Zero sales**: Automatically filters out deliveries with zero or negative sales
- **Empty destinations**: Skips route optimization for trucks with no destinations
- **Invalid postal codes**: Assumes 5-digit numeric postal codes

## Performance Considerations

- **Scalability**: Suitable for up to 1000 deliveries per day
- **Memory**: Minimal memory footprint, processes data in groups
- **Speed**: Route optimization is O(n²) per truck, fast for typical scenarios
- **Parallelization**: Can be extended to process multiple trucks in parallel

## Integration with Other Modules

The Router module integrates seamlessly with other SC Labs modules:

1. **Forecaster**: Takes forecast output as input for demand distribution
2. **Optimizer**: Can be used alongside inventory optimization for complete planning
3. **Auxiliar**: Uses customer data structure compatible with synthetic data generation

## References

- Nearest Neighbor Algorithm: Classic TSP heuristic
- Vehicle Routing Problem (VRP): Broader class of routing optimization problems
- OR-Tools: Google's optimization toolkit for advanced routing solutions
