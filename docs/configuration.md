# Configuration Guide

## Overview

SC Labs uses `config.yaml` as the **single source of truth** for all default parameters. This ensures consistency between the CLI and web dashboard.

**Important:** All defaults are defined in `config.yaml`. The `utils/config.py` module is just a loader with no hardcoded fallbacks.

## Configuration File Location

```
config.yaml (project root)
```

## How It Works

1. **config.yaml** - Contains all default values
2. **utils/config.py** - Loads config.yaml and provides access via `get_config()`
3. **app/app.py** - Reads defaults on startup, stores in session state
4. **Settings page** - Allows runtime modification (session only, doesn't update config.yaml)

## Configuration Structure

### Data Generation

Controls synthetic data generation parameters:

```yaml
data_generation:
  default_n_stores: 3          # Number of stores to simulate
  default_n_products: 2        # Number of products to simulate
  default_n_weeks: 52          # Weeks of historical data
  default_start_date: '2024-01-01'  # Start date for data
  default_seed: 42             # Random seed for reproducibility
```

### Forecasting

Controls demand forecasting with LightGBM:

```yaml
forecasting:
  default_forecast_horizon: 4  # Number of periods to predict
  num_boost_round: 200         # LightGBM boosting rounds
  learning_rate: 0.05          # Learning rate
  num_leaves: 31               # Max leaves per tree
  early_stopping_rounds: 20    # Early stopping patience
```

**Key Parameter: `forecast_horizon`**
- Defines how many periods ahead to forecast
- Used by the Forecaster module
- Example: `forecast_horizon=4` means predict next 4 weeks

### Inventory Optimization

Controls inventory simulation parameters:

```yaml
inventory:
  default_planning_horizon: 8  # Number of periods to simulate
  default_service_level: 0.95  # Target service level (95%)
  default_lead_time: 2         # Order lead time in periods
  default_review_period: 1     # Inventory review frequency
```

**Key Parameter: `planning_horizon`**
- Defines how many periods to simulate inventory
- Used by the Optimizer module
- Should be >= `forecast_horizon`
- Example: `planning_horizon=8` means simulate 8 weeks of inventory

### Routing

Controls delivery routing parameters:

```yaml
routing:
  default_max_payload: 100     # Maximum units per truck
  default_n_customers: 30      # Number of customers to generate
  default_origin: '08020'      # Origin postal code
  algorithm: 'nearest_neighbor'  # Routing algorithm
```

## Understanding Horizons

The system uses two key horizon parameters that work together:

### 1. Forecast Horizon
- **Location**: `forecasting.default_forecast_horizon`
- **Purpose**: Number of periods to predict into the future
- **Used by**: Forecaster module
- **Example**: `forecast_horizon=4` → predict next 4 weeks

### 2. Planning Horizon
- **Location**: `inventory.default_planning_horizon`
- **Purpose**: Number of periods to simulate inventory management
- **Used by**: Optimizer module
- **Constraint**: Should be >= forecast_horizon
- **Example**: `planning_horizon=8` → simulate 8 weeks of inventory

### Workflow Example

```
Historical Data (52 weeks)
    ↓
Forecaster (forecast_horizon=4)
    ↓
Predictions (4 weeks)
    ↓
Optimizer (planning_horizon=8)
    ↓
Inventory Plan (8 weeks)
    ↓
Router
    ↓
Delivery Routes
```

## How Configuration is Used

### CLI (src/main.py)
- Reads defaults from `config.yaml`
- Can be overridden with command-line arguments
- Example: `python src/main.py --forecast_horizon 6`

### Web Dashboard (app/app.py)
- Reads defaults from `config.yaml` on startup
- Stores values in Streamlit session state
- Can be modified in Settings page
- Changes persist during the session

### Modules
- Each module can access config via `utils.config.get_config()`
- Falls back to hardcoded defaults if config.yaml is missing

## Modifying Configuration

### Option 1: Edit config.yaml (Recommended)
```yaml
forecasting:
  default_forecast_horizon: 6  # Change from 4 to 6
```

### Option 2: CLI Arguments
```bash
python src/main.py --forecast_horizon 6 --planning_horizon 12
```

### Option 3: Web Dashboard Settings
1. Launch app: `python run_app.py`
2. Navigate to Settings (⚙️)
3. Adjust sliders
4. Changes apply immediately

## Best Practices

1. **Keep planning_horizon >= forecast_horizon**
   - Ensures inventory simulation covers all forecasted periods
   - Typical ratio: planning_horizon = 2 × forecast_horizon

2. **Adjust horizons based on business needs**
   - Short-term planning: forecast_horizon=4, planning_horizon=8
   - Long-term planning: forecast_horizon=12, planning_horizon=24

3. **Use config.yaml for team defaults**
   - Commit config.yaml to version control
   - Ensures consistent defaults across team

4. **Use CLI args for experiments**
   - Quick parameter testing without modifying config
   - Example: `python src/main.py --forecast_horizon 8 --verbose`

## Configuration Priority

When multiple configuration sources exist:

1. **CLI arguments** (highest priority)
2. **Session state** (web dashboard only)
3. **config.yaml**
4. **Hardcoded defaults** (lowest priority)

## Troubleshooting

### Config not loading
- Check that `config.yaml` exists in project root
- Verify YAML syntax (use a YAML validator)
- Check file permissions

### Values not updating
- **CLI**: Ensure you're passing arguments correctly
- **Web Dashboard**: Check Settings page, values persist in session
- **Restart**: Restart app to reload config.yaml

### Horizon mismatch errors
- Ensure `planning_horizon >= forecast_horizon`
- Check both config.yaml and Settings page values

## See Also

- [Architecture Overview](ARCHITECTURE.md) - System design
- [Forecaster Module](forecaster.md) - Forecasting details
- [Optimizer Module](optimizer.md) - Inventory optimization details
