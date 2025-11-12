# SC Labs - Supply Chain Optimization

A modular end-to-end supply chain optimization system demonstrating demand forecasting, inventory management, and delivery routing capabilities.

## 🎯 Quick Facts

- **Authentication**: Supabase (user accounts only) or local demo mode
- **Data Storage**: Local CSV files in `output/` directory
- **Forecasting**: LightGBM with automated feature engineering
- **Deployment**: Streamlit Cloud, Railway, or local

**See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture explanation.**

## Overview

SC Labs is a proof-of-concept Python application that showcases a complete supply chain optimization workflow. The system uses machine learning for demand forecasting (LightGBM with automated feature engineering), implements classic inventory control policies (reorder point/safety stock), and provides basic route optimization for delivery planning.

### Key Features

- **Demand Forecasting**: Time series forecasting using LightGBM with automated feature engineering (temporal, lag, rolling statistics)
- **Inventory Optimization**: Reorder point policy with safety stock calculations
- **Delivery Routing**: Truck assignment and route optimization with payload constraints
- **Synthetic Data Generation**: Realistic test data with trend, seasonality, and noise
- **Modular Architecture**: Clean separation of concerns with extensible design
- **No Data Leakage**: Proper train/test splits and feature engineering to prevent information leakage

## Project Structure

```
SC_Labs/
├── src/                   # Source code
│   └── main.py            # Main orchestration script
├── auxiliar/              # Synthetic data generation module
│   ├── auxiliar.py        # Data generation functions
│   ├── utils/             # Utility functions
│   └── __init__.py
├── forecaster/            # Demand forecasting module
│   ├── forecaster.py      # Forecaster class with LightGBM
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
├── utils/                 # Shared utilities
│   ├── output_manager.py  # Centralized output file handling
│   ├── logger.py          # Logging utilities
│   └── config.py          # Configuration management
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
│   ├── app.py             # Main dashboard application
│   └── app_utils.py       # Dashboard utilities
├── main.py                # Redirect to src/main.py (backward compatibility)
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

3. Configure authentication (for web dashboard):

**Windows:**
```bash
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```

**Linux/Mac:**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

The default configuration uses simple authentication with demo credentials:
- Username: `demo`
- Password: `demo123`

For production deployment with Supabase, see [Deployment Guide](docs/deploy.md).

### Manual Setup

Alternatively, install dependencies manually:
```bash
pip install -r requirements.txt
pip install streamlit>=1.28.0
```

This will install:
- pandas (data manipulation)
- numpy (numerical operations)
- lightgbm (gradient boosting for time series forecasting)
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
python src/main.py
```

Or use the backward-compatible redirect:

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
python src/main.py --n_stores 5 --n_products 3 --save

# Run with verbose output
python src/main.py --verbose

# Customize forecasting and inventory parameters
python src/main.py --forecast_horizon 8 --service_level 0.99 --lead_time 3

# Customize routing parameters
python src/main.py --max_payload 20 --n_customers 50
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

When using the `--save` flag, the following CSV files are generated in the `output/` directory:

- `forecasts_YYYYMMDD_HHMMSS.csv`: Demand forecasts with historical and predicted values
- `inventory_plan_YYYYMMDD_HHMMSS.csv`: Inventory simulation with orders and shipments
- `deliveries_YYYYMMDD_HHMMSS.csv`: Delivery assignments with truck and customer details
- `routes_YYYYMMDD_HHMMSS.csv`: Route summaries with distances and stop sequences

**File Naming Convention:**
All output files use a consistent timestamp format (`YYYYMMDD_HHMMSS`) to enable versioning and prevent overwrites. The `OutputManager` utility class ensures consistent naming across both CLI and web dashboard interfaces.

**Example:**
```
output/
├── forecasts_20241112_143022.csv
├── inventory_plan_20241112_143022.csv
├── deliveries_20241112_143022.csv
└── routes_20241112_143022.csv
```

## 🌐 Web Dashboard

Launch the interactive Streamlit dashboard for a complete visual experience:

```bash
# Quick start (recommended)
python run_app.py

# Or directly with Streamlit
streamlit run app/app.py
```

**Features:**
- 🔐 **Secure Authentication** - User login with Supabase or simple auth
- 🏠 **Beautiful Landing Page** - Modern interface with module cards and navigation
- 📊 **Interactive Data Generation** - Real-time visualization with trend analysis
- 📈 **LightGBM Forecasting** - Gradient boosting with automated feature engineering
- 📦 **Inventory Optimization** - Dynamic charts with safety stock visualization
- 🚚 **Route Planning** - Truck utilization analysis and distance optimization
- 💾 **Data Persistence** - Save results to Supabase or local storage
- ⚙️ **Settings Panel** - Centralized configuration for all modules
- 🔄 **Seamless Navigation** - Navigate between modules with persistent state

The dashboard provides a user-friendly interface to explore all supply chain optimization modules without writing code. Each module can be accessed independently, and you can navigate between them at any time.

### Authentication

The app includes secure authentication with Supabase:

**Login Credentials:**
- Contact your administrator for login credentials

**Configuration Options:**

To disable user signup (login only), set in `.env` or `.streamlit/secrets.toml`:
```bash
ALLOW_SIGNUP=false
```

To enable user registration:
```bash
ALLOW_SIGNUP=true
```

**Note**: Supabase is used ONLY for authentication. All data (forecasts, inventory, routes) is stored locally in the `output/` directory as CSV files.

See [Deployment Guide](docs/deploy.md) for production setup.

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

## Documentation

### 📚 Essential Guides

- **[Architecture Overview](docs/ARCHITECTURE.md)** - ⭐ **START HERE** - Understand what goes where
- **[Deployment Guide](docs/deploy.md)** - Deploy to production with Supabase

### 📖 Module Documentation

- [Auxiliar Module](docs/auxiliar.md) - Synthetic data generation
- [Forecaster Module](docs/forecaster.md) - Demand forecasting with LightGBM
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

The project automatically prevents Python from creating `__pycache__` directories:

**Automatic Prevention:**
- All scripts (`main.py`, `run_app.py`) use the `-B` flag automatically
- `.env` file sets `PYTHONDONTWRITEBYTECODE=1`
- `.gitignore` excludes `__pycache__/` from version control

**Manual Cleanup:**
```bash
# Clean existing __pycache__ directories
python clean_cache.py
```

**Manual Prevention (if needed):**
```bash
# Run any script with -B flag
python -B your_script.py

# Or set environment variable
export PYTHONDONTWRITEBYTECODE=1  # Linux/Mac
set PYTHONDONTWRITEBYTECODE=1     # Windows CMD
$env:PYTHONDONTWRITEBYTECODE=1    # Windows PowerShell
```

### Extending the System

The modular design allows easy extension:

- **Custom Forecasting Models**: Extend the Forecaster class or add custom features
- **Advanced Routing**: Replace the nearest neighbor heuristic with OR-Tools VRP solver
- **Real Data Integration**: Modify data loading in main.py to read from CSV/database
- **Additional Constraints**: Add capacity, time windows, or vehicle type constraints

## Technical Details

### Forecasting Approach

- Uses LightGBM gradient boosting with automated feature engineering
- Features include: temporal (day/week/month), lag (1-4 periods), rolling statistics (mean/std/min/max)
- Proper train/test split with no data leakage
- Recursive forecasting for multi-step ahead predictions
- Supports multiple time series (store-product combinations)
- Configurable forecast horizon and frequency

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
- Single-echelon inventory (no warehouse-to-store)

### Potential Enhancements

- Integration with real coordinate systems (geopy)
- Advanced routing with OR-Tools VRP solver
- Additional feature engineering (holidays, promotions, external factors)
- Multi-echelon inventory optimization
- Hyperparameter tuning for LightGBM
- API endpoints for web service deployment

## License

See LICENSE file for details.

## Troubleshooting

### "No secrets found" Error

**Solution**: Copy the secrets template:
```bash
# Windows
copy .streamlit\secrets.toml.example .streamlit\secrets.toml

# Linux/Mac
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

### "Supabase credentials not found"

**Solution**: Either:
1. Set `USE_SUPABASE = "false"` in `.streamlit/secrets.toml` (for demo mode)
2. Or add your Supabase credentials to the file

### Can't login with demo/demo123

**Check**: `.streamlit/secrets.toml` exists and contains:
```toml
USE_SUPABASE = "false"

[users]
demo = "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"
```

### Where is my data stored?

**Answer**: Check the `output/` directory for CSV files:
- `forecasts_*.csv`
- `inventory_plan_*.csv`
- `deliveries_*.csv`
- `routes_*.csv`

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for complete explanation.

## Contributing

This is a proof-of-concept project. For questions or suggestions, please refer to the module documentation in the `docs/` directory.
