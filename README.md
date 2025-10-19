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
├── notebooks/             # Jupyter notebooks for interactive exploration
│   ├── 01_data_generation.ipynb
│   ├── 02_forecasting.ipynb
│   ├── 03_inventory_optimization.ipynb
│   ├── 04_delivery_routing.ipynb
│   └── README.md
├── tests/                 # Test suite
│   ├── test_auxiliar.py
│   ├── test_forecaster.py
│   ├── test_optimizer.py
│   ├── test_router.py
│   ├── test_integration.py
│   └── README.md
├── docs/                  # Module documentation
│   ├── auxiliar.md
│   ├── forecaster.md
│   ├── optimizer.md
│   └── router.md
├── output/                # Generated output files
├── app/                   # Streamlit web dashboard
│   └── app.py             # Main dashboard application
├── main.py                # Main orchestration script
├── setup.py               # Automated setup script
├── run_app.py             # Streamlit app launcher
├── requirements.txt       # Python dependencies
├── config.yaml            # Configuration file
└── README.md              # This file
```

## Installation

### Prerequisites

- Python >= 3.10
- pip package manager

### Quick Setup

1. Clone or download this repository

2. Run the automated setup script:
```bash
python setup.py
```

This will:
- Create all necessary output directories
- Install all Python dependencies from requirements.txt
- Install Streamlit for the web dashboard
- Verify the installation

### Manual Setup

Alternatively, install dependencies manually:
```bash
pip install -r requirements.txt
pip install streamlit>=1.28.0
```

This will install:
- pandas (data manipulation)
- numpy (numerical operations)
- gluonts (time series forecasting)
- torch (PyTorch backend for GluonTS)
- scipy (statistical functions)
- streamlit (web dashboard)

## Usage

### Web Dashboard (Recommended)

Launch the interactive dashboard using the app runner:

```bash
python run_app.py
```

Or directly with Streamlit:

```bash
streamlit run app/app.py
```

Navigate through 4 sections:
1. **Data Generation** - Create synthetic data
2. **Forecasting** - Train models and predict
3. **Inventory** - Optimize stock levels
4. **Routing** - Plan deliveries

The dashboard will open in your browser at http://localhost:8501

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

## 🌐 Web Dashboard

Launch the interactive Streamlit dashboard for a complete visual experience:

```bash
# Quick start (recommended)
python run_app.py

# Or directly with Streamlit
streamlit run app/app.py
```

**Features:**
- 📊 Interactive data generation with real-time visualization
- 📈 Neural network forecasting with accuracy metrics
- 📦 Inventory optimization with dynamic charts
- 🚚 Route planning with truck utilization analysis

The dashboard provides a user-friendly interface to explore all supply chain optimization modules without writing code.

## 📓 Interactive Notebooks

Explore the pipeline interactively with Jupyter notebooks in the `notebooks/` directory:

- **[01_data_generation.ipynb](notebooks/01_data_generation.ipynb)** - Generate synthetic sales data
- **[02_forecasting.ipynb](notebooks/02_forecasting.ipynb)** - Train forecasting models and predict demand
- **[03_inventory_optimization.ipynb](notebooks/03_inventory_optimization.ipynb)** - Optimize inventory levels
- **[04_delivery_routing.ipynb](notebooks/04_delivery_routing.ipynb)** - Plan and optimize delivery routes

Each notebook includes:
- Step-by-step explanations
- Rich visualizations
- Interactive analysis
- Saves outputs for the next stage

See [notebooks/README.md](notebooks/README.md) for details.

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

All tests are located in the `tests/` directory:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_integration.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

See [tests/README.md](tests/README.md) for detailed test documentation.

### Preventing __pycache__ Directories

The project is configured to prevent Python from creating `__pycache__` directories:

**Option 1: Environment Variable (Recommended)**
```bash
# Set environment variable before running Python
export PYTHONDONTWRITEBYTECODE=1  # Linux/Mac
set PYTHONDONTWRITEBYTECODE=1     # Windows CMD
$env:PYTHONDONTWRITEBYTECODE=1    # Windows PowerShell

# Or add to your .env file (already configured)
PYTHONDONTWRITEBYTECODE=1
```

**Option 2: Run Python with -B flag**
```bash
python -B main.py
python -B -m pytest tests/
```

**Option 3: Clean existing cache**
```bash
# Use the provided cleanup script
python clean_pycache.py

# Or manually with PowerShell
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

The `.gitignore` file is already configured to exclude `__pycache__/` directories from version control

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
