# SC Labs - Supply Chain Optimization

A modular end-to-end supply chain optimization platform with demand forecasting, inventory management, and delivery routing.

## 🎯 Quick Facts

- **Interface**: Web dashboard (Streamlit) + CLI
- **Forecasting**: LightGBM with automated feature engineering
- **Data**: Upload your own CSV or generate synthetic demo data
- **Storage**: Local CSV files in `output/` directory
- **Authentication**: Supabase or local demo mode

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Launch
```bash
python run_app.py
```

### 3. Use
1. **Data & Settings** → Upload data or generate demo data
2. **Forecasting** → Train model and predict demand
3. **Inventory** → Optimize stock levels
4. **Routing** → Plan delivery routes

## 📊 Key Features

### Data Management
- **Upload CSV** - Use your own historical sales data
- **Generate Demo Data** - Create synthetic data for testing
- **Load Previous Results** - Resume from saved outputs
- **Data Quality Validation** - Automatic warnings and recommendations

### Demand Forecasting
- LightGBM gradient boosting model
- Automated feature engineering (temporal, lag, rolling statistics)
- Proper train/test splits (no data leakage)
- Multi-step ahead predictions
- Configurable forecast horizon (1-52 weeks)

### Inventory Optimization
- Reorder point policy with safety stock
- Configurable service levels (80-99%)
- Lead time management (1-8 weeks)
- Stockout prevention
- Fill rate tracking

### Delivery Routing
- Truck assignment with payload constraints
- Route optimization (nearest neighbor)
- Utilization analysis
- Distance minimization

## ⚙️ Configuration Settings

All settings can be adjusted in the **Data & Settings** → **Configuration** tab:

### Data Generation
| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Stores | 1-10 | 3 | Number of stores |
| Products | 1-10 | 2 | Number of products |
| Historical Weeks | 20-208 | 52 | Weeks of historical data (up to 4 years) |

### Forecasting
| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Forecast Horizon | 1-52 | 12 | Weeks to forecast ahead |

**Advanced Settings** (in `config.yaml`):
```yaml
forecasting:
  num_boost_round: 200        # LightGBM training iterations
  learning_rate: 0.05         # Model learning rate
  num_leaves: 31              # Tree complexity
  early_stopping_rounds: 20   # Stop if no improvement
```

### Inventory
| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Planning Horizon | 1-52 | 12 | Weeks to simulate |
| Service Level | 80-99% | 95% | Target service level |
| Lead Time | 1-8 | 2 | Order lead time (weeks) |
| Review Period | 1-4 | 1 | Inventory review frequency (weeks) |

### Routing
| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Max Payload | 50-500 | 100 | Maximum units per truck |
| Customers | 10-100 | 15 | Number of delivery customers |

## 📁 Data Format

Your CSV must have these columns:

| Column | Type | Required | Example |
|--------|------|----------|---------|
| `store` | string | ✓ | "A", "Store_1" |
| `product` | string | ✓ | "A", "Product_1" |
| `date` | date | ✓ | "2024-01-07" |
| `sales` | numeric | ✓ | 116, 89.5 |
| `inventory` | numeric | Optional | 202 |
| `customer_id` | string | Optional | "CUST_123" |
| `destination` | string | Optional | "08020" |

**Tips:**
- Use weekly data for best results
- Minimum 20-30 historical periods recommended
- Date format must be YYYY-MM-DD
- Continuous time series preferred

Download sample data from the app to see the exact format.

## 🗂️ Project Structure

```
SC_Labs/
├── app/                    # Streamlit web dashboard
│   ├── pages/              # Page modules (landing, data, forecast, inventory, routing)
│   ├── app.py              # Main application
│   ├── navigation.py       # Navigation components
│   ├── components.py       # Shared UI components
│   └── styles.css          # Custom styling
├── forecaster/             # Demand forecasting (LightGBM)
├── optimizer/              # Inventory optimization
├── router/                 # Delivery routing
├── auxiliar/               # Synthetic data generation
├── utils/                  # Shared utilities
├── data/                   # Sample datasets
├── output/                 # Generated results (CSV files)
├── tests/                  # Test suite
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks
└── src/                    # CLI scripts
```

## 🧹 Maintenance

Clean up temporary files regularly:

```bash
# Clean everything
python src/cleanup.py --all

# Clean specific items
python src/cleanup.py --logs          # Remove lightning_logs
python src/cleanup.py --cache         # Remove __pycache__
python src/cleanup.py --outputs       # Remove old CSV files (30+ days)

# Preview without deleting
python src/cleanup.py --all --dry-run
```

## 🔧 Advanced Usage

### CLI Mode

Run the complete pipeline from command line:

```bash
# Basic usage
python src/main.py

# Custom parameters
python src/main.py --n_stores 5 --n_products 3 --forecast_horizon 8 --save
```

### Jupyter Notebooks

Interactive exploration in `notebooks/`:
- `01_data_generation.ipynb` - Generate synthetic data
- `02_forecasting.ipynb` - Train and predict
- `03_inventory_optimization.ipynb` - Optimize stock
- `04_delivery_routing.ipynb` - Plan routes

### Python API

```python
from forecaster.forecaster import Forecaster
from optimizer.optimizer import Optimizer
from router.router import Router

# Load your data
data = pd.read_csv('your_data.csv')

# Forecast
forecaster = Forecaster(
    primary_keys=['store', 'product'],
    date_col='date',
    target_col='sales',
    forecast_horizon=12
)
forecaster.fit(data)
forecasts = forecaster.predict(data)

# Optimize inventory
optimizer = Optimizer(
    primary_keys=['store', 'product'],
    planning_horizon=12,
    service_level=0.95
)
inventory_plan = optimizer.simulate(forecasts)

# Route deliveries
router = Router(max_payload=100)
router.generate_customers(n_customers=20)
deliveries = router.distribute_demand(forecasts)
deliveries, routes = router.optimize_routes(deliveries)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module
pytest tests/test_forecaster.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## 📚 Documentation

- **[Quick Start Guide](QUICK_START.md)** - Get started in 5 minutes
- **[Cleanup Guide](src/README_CLEANUP.md)** - Maintenance utilities
- **[Deployment Guide](docs/deploy.md)** - Production deployment
- **Module Docs**: [forecaster](docs/forecaster.md), [optimizer](docs/optimizer.md), [router](docs/router.md)

## � ACuthentication

The app includes authentication via Supabase or simple local auth.

**Demo Mode** (default):
- Username: `demo`
- Password: `demo123`

**Production**: Configure Supabase credentials in `.env` or `.streamlit/secrets.toml`

See [Deployment Guide](docs/deploy.md) for details.

## 💡 Tips

**Data Quality:**
- Use at least 20-30 historical periods
- Weekly data works best
- Check for missing dates
- Review validation warnings in the app

**Forecasting:**
- Longer history = better forecasts
- Recommended horizon: 20-25% of historical data
- Adjust LightGBM parameters in `config.yaml` for fine-tuning

**Inventory:**
- Higher service level = more safety stock
- Adjust lead time based on your supply chain
- Planning horizon should be ≥ forecast horizon

**Routing:**
- Start with fewer customers (15-20) for faster results
- Adjust max payload based on vehicle capacity
- Review truck utilization for efficiency

## 🚀 What's New

**Recent Enhancements:**
- ✅ Data upload with validation
- ✅ Load previous results
- ✅ Data quality warnings
- ✅ Modular page structure
- ✅ Enhanced UI/UX with consistent design
- ✅ Progress indicators
- ✅ Cleanup utilities

## 🛠️ Tech Stack

- **Backend**: Python 3.10+
- **ML**: LightGBM
- **Web**: Streamlit
- **Data**: Pandas, NumPy
- **Auth**: Supabase (optional)

## 📝 License

See LICENSE file for details.

## 🤝 Contributing

This is a proof-of-concept project. For questions or suggestions, see the module documentation in `docs/`.

---

**SC Labs** • Barcelona • 2025
