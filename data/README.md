# Sample Data

This folder contains sample datasets for SC Labs.

## Sample Dataset

**File:** `sample_sales_data.csv`

This is a pre-generated sample dataset you can use to test the forecasting, inventory, and routing modules without generating synthetic data.

### Data Format

To upload your own data, your CSV file must have the following columns:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `store` | string | Store identifier (e.g., "A", "B", "Store_1") | ✓ Yes |
| `product` | string | Product identifier (e.g., "A", "B", "Product_1") | ✓ Yes |
| `date` | date | Date in YYYY-MM-DD format | ✓ Yes |
| `sales` | numeric | Sales quantity for that period | ✓ Yes |
| `inventory` | numeric | Inventory level (optional, can be 0) | No |
| `customer_id` | string | Customer identifier (optional) | No |
| `destination` | string | Destination postal code (optional) | No |

### Example Format

```csv
store,product,date,sales,inventory,customer_id,destination
A,A,2024-01-07,116,202,CUST_2306,61663
A,A,2024-01-14,140,445,CUST_7776,25708
A,B,2024-01-07,89,150,CUST_1234,08020
B,A,2024-01-07,105,180,CUST_5678,28001
```

### Requirements

1. **Date Format:** Must be in YYYY-MM-DD format
2. **Frequency:** Data should be at regular intervals (weekly recommended)
3. **Completeness:** Each store-product combination should have continuous time series data
4. **Numeric Values:** Sales and inventory must be numeric (integers or decimals)

### Tips

- Use weekly data for best results (daily data may be too granular)
- Ensure at least 20-30 historical periods for meaningful forecasts
- Missing dates will be handled by the system, but continuous data is preferred
- Store and product identifiers can be any string (letters, numbers, or combinations)

### Using Your Own Data

1. Prepare your CSV file following the format above
2. Go to the **Data & Settings** module in the app
3. Select **"Upload Data"** tab
4. Upload your CSV file
5. The system will validate the format and show a preview
6. Proceed to Forecasting, Inventory, or Routing modules
