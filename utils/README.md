# Utils Module

Shared utility functions used across the project.

## Files

### config.py
**Purpose:** Loads and provides access to `config.yaml`

**Usage:**
```python
from utils.config import get_config

config = get_config()
value = config.get('forecasting', 'default_forecast_horizon', default=4)
```

**Important:** 
- No hardcoded defaults - all values come from `config.yaml`
- Single source of truth: `config.yaml`
- If config.yaml is missing, returns empty dict with warning

### output_manager.py
**Purpose:** Manages output file naming and saving

**Usage:**
```python
from utils.output_manager import OutputManager

output_mgr = OutputManager()
filepath = output_mgr.save_forecasts(forecasts_df)
```

### logger.py
**Purpose:** Logging utilities for consistent output formatting

## Why utils/ and not app/?

The `utils/` folder contains shared utilities that can be used by:
- ✅ CLI (`src/main.py`)
- ✅ Web App (`app/app.py`)
- ✅ Modules (forecaster, optimizer, router)
- ✅ Tests

If a utility is only used by the web app, it should go in `app/` instead.
