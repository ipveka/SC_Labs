# SC Labs - Supply Chain Optimization

A modular end-to-end supply chain optimization system demonstrating demand forecasting, inventory management, and delivery routing capabilities.

## Overview

SC Labs is a proof-of-concept Python application that showcases a complete supply chain optimization workflow. The system uses machine learning for demand forecasting (GluonTS), implements classic inventory control policies (reorder point/safety stock), and provides basic route optimization for delivery planning.

### Key Features

- **Demand Forecasting**: Time series forecasting using GluonTS SimpleFeedForwardEstimator
- **Inventory Optimization**: Reorder point policy with safety stock calculations
- **Delivery Routing**: Truck assignment and route optimization with payload constraints
- **Synthetic Data Generation**: Realistic test data with trend, seasonality, and noise
- **Modular Architecture**: Clean separation of concerns with extensible design

## Project Structure

```
SC_Labs/
├── auxiliar/              # Synthetic data generation module
│   ├── auxiliar.py        # Data generation functions
│   ├── utils/             # Utility functions
│   └── __init__.py
├── forecaster/            # Demand forecasting module
│   ├── forecaster.py      # Forecaster class with GluonTS
│   ├── utils/             # Forecasting utilities
│   └── __init__.py
├── optimizer/             # Inventory optimization module
│   ├── optimizer.py       # Optimizer class with reorder point policy
│   ├── utils/             # Optimization utilities
│   └── __init__.py
├── router/                # Delivery routing module
│   ├── router.py          # Router class with truck assignment
│   ├── utils/             # Routing utilities
│   └── __init__.py
├── docs/                  # Module documentation
│   ├── auxiliar.md        # Auxiliar module documentation
│   ├── forecaster.md      # Forecaster module documentation
│   ├── optimizer.md       # Optimizer module documentation
│   └── router.md          # Router module documentation
├── outputs/               # Generated output files (created at runtime)
├── main.py                # Main orchestration script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python >= 3.10
- pip package manager

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- pandas (data manipulation)
- numpy (numerical operations)
- gluonts (time series forecasting)
- mxnet (GluonTS backend)
- scipy (statistical functions)

## Usage

### Basic Usage

Run the pipeline with default parameters:

```bash
python main.py
```

This will:
1. Generate synthetic data (3 stores × 2 products × 52 weeks)
2. Forecast demand for 4 periods ahead
3. Simulate inventory for 8 periods
4. Optimize delivery routes with truck assignment

### Advanced Usage

Customize parameters via command-line arguments:

```bash
# Run with 5 stores, 3 products, and save outputs
python main.py --n_stores 5 --n_products 3 --save

# Run with verbose output
python main.py --verbose

# Customize forecasting and inventory parameters
python main.py --forecast_horizon 8 --service_level 0.99 --lead_time 3

# Customize routing parameters
python main.py --max_payload 20 --n_customers 50
```

### Available Parameters

**Data Generation:**
- `--n_stores`: Number of stores (default: 3)
- `--n_products`: Number of products (default: 2)
- `--n_weeks`: Weeks of historical data (default: 52)
- `--start_date`: Start date in YYYY-MM-DD format (default: 2024-01-01)

**Forecasting:**
- `--forecast_horizon`: Periods to forecast (default: 4)

**Inventory Optimization:**
- `--planning_horizon`: Periods to simulate (default: 8)
- `--service_level`: Target service level 0.0-1.0 (default: 0.95)
- `--lead_time`: Order lead time in periods (default: 2)

**Routing:**
- `--max_payload`: Maximum units per truck (default: 10)
- `--n_customers`: Number of customers (default: 20)

**Output Options:**
- `--save`: Save outputs to CSV files in outputs/ directory
- `--verbose`: Display detailed information

### Example Output

```
================================================================================
SC LABS - SUPPLY CHAIN OPTIMIZATION SUMMARY
================================================================================

📊 FORECAST SUMMARY
--------------------------------------------------------------------------------
Forecast periods: 24
Total forecasted demand: 2847.32 units
Average forecasted demand: 118.64 units/period

📦 INVENTORY SUMMARY
--------------------------------------------------------------------------------
Average inventory level: 245.67 units
Min inventory level: 89.23 units
Max inventory level: 456.78 units
Total orders placed: 1234.56 units
Total shipments received: 1234.56 units
Stockout periods: 0

🚚 ROUTING SUMMARY
--------------------------------------------------------------------------------
Total trucks required: 12
Total deliveries: 240
Total route distance: 4567.89 units
Average route distance: 380.66 units/truck
Average truck payload: 9.5 units

================================================================================
```

### Output Files

When using the `--save` flag, the following CSV files are generated in the `outputs/` directory:

- `forecasts_YYYYMMDD_HHMMSS.csv`: Demand forecasts with historical and predicted values
- `inventory_plan_YYYYMMDD_HHMMSS.csv`: Inventory simulation with orders and shipments
- `deliveries_YYYYMMDD_HHMMSS.csv`: Delivery assignments with truck and customer details
- `routes_YYYYMMDD_HHMMSS.csv`: Route summaries with distances and stop sequences

## Module Documentation

Detailed documentation for each module is available in the `docs/` directory:

- [Auxiliar Module](docs/auxiliar.md) - Synthetic data generation
- [Forecaster Module](docs/forecaster.md) - Demand forecasting with GluonTS
- [Optimizer Module](docs/optimizer.md) - Inventory optimization with reorder point policy
- [Router Module](docs/router.md) - Delivery routing and truck assignment

## Architecture

The system follows a modular pipeline architecture:

```
Synthetic Data → Forecaster → Optimizer → Router → Outputs
                                    ↓
                              Router (parallel)
```

Each module is independently testable and can be used standalone or as part of the complete pipeline.

## Development

### Running Tests

Unit tests can be found in each module's test files (when implemented):

```bash
pytest
```

### Extending the System

The modular design allows easy extension:

- **Custom Forecasting Models**: Extend the Forecaster class or swap GluonTS models
- **Advanced Routing**: Replace the nearest neighbor heuristic with OR-Tools VRP solver
- **Real Data Integration**: Modify data loading in main.py to read from CSV/database
- **Additional Constraints**: Add capacity, time windows, or vehicle type constraints

## Technical Details

### Forecasting Approach

- Uses GluonTS SimpleFeedForwardEstimator (neural network)
- Supports multiple time series (store-product combinations)
- Configurable forecast horizon and frequency
- Handles missing data with forward fill

### Inventory Policy

- Implements (s, S) reorder point system
- Safety stock based on demand variability and service level
- Configurable review period and lead time
- Tracks orders in flight and shipment arrivals

### Routing Algorithm

- Nearest neighbor heuristic for route optimization
- Payload-constrained truck assignment
- Distance calculation based on postal code proximity
- Supports multiple trucks per day

## Limitations & Future Enhancements

### Current Limitations

- Uses synthetic data only
- Simple distance heuristic (postal code difference)
- Basic forecasting model (SimpleFeedForwardEstimator)
- Single-echelon inventory (no warehouse-to-store)

### Potential Enhancements

- Integration with real coordinate systems (geopy)
- Advanced routing with OR-Tools VRP solver
- More sophisticated forecasting models (DeepAR, Transformer)
- Multi-echelon inventory optimization
- Interactive dashboards with Plotly/Streamlit
- API endpoints for web service deployment

## License

See LICENSE file for details.

## Contributing

This is a proof-of-concept project. For questions or suggestions, please refer to the module documentation in the `docs/` directory.
