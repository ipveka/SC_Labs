# SC Labs - Test Suite

This directory contains all unit and integration tests for the SC Labs Supply Chain Optimization project.

## Test Structure

```
tests/
├── __init__.py                 # Package initialization
├── test_auxiliar.py           # Tests for data generation module
├── test_forecaster.py         # Tests for demand forecasting module
├── test_optimizer.py          # Tests for inventory optimization module
├── test_router.py             # Tests for delivery routing module
└── test_integration.py        # End-to-end integration tests
```

## Running Tests

### Run all tests
```bash
python -m pytest tests/ -v
```

### Run specific test file
```bash
python -m pytest tests/test_integration.py -v
```

### Run specific test class or method
```bash
python -m pytest tests/test_integration.py::TestEndToEndWorkflow::test_complete_pipeline -v
```

### Run tests with coverage
```bash
python -m pytest tests/ --cov=. --cov-report=html
```

## Test Categories

### Unit Tests

- **test_auxiliar.py**: Tests for synthetic data generation
  - Schema validation
  - Data dimensions
  - Reproducibility
  - Value ranges and formats

- **test_forecaster.py**: Tests for demand forecasting
  - Data preparation with missing dates
  - Model training with minimal datasets
  - Prediction output schema
  - Multiple time series handling

- **test_optimizer.py**: Tests for inventory optimization
  - Safety stock calculations
  - Reorder point logic
  - Inventory simulation
  - Order placement and shipment arrival

- **test_router.py**: Tests for delivery routing
  - Customer generation
  - Demand distribution
  - Truck assignment with payload constraints
  - Route optimization

### Integration Tests

- **test_integration.py**: End-to-end workflow tests
  - Complete pipeline with small dataset (1 store, 1 product, 20 weeks)
  - Multiple stores and products scalability
  - Edge case handling
  - Data flow validation across modules
  - Output format verification

## Test Requirements

All tests validate against the requirements specified in `.kiro/specs/supply-chain-optimization/requirements.md`.

Key requirements covered:
- **6.1**: Data generation and pipeline setup
- **6.2**: Forecaster output format and accuracy
- **6.3**: Optimizer output format and inventory logic
- **6.4**: Router output format and routing logic
- **6.7**: Data flow consistency across modules

## Notes

- Tests use `pytest` framework
- Random seeds are set for reproducibility where applicable
- Integration tests may take longer due to model training
- All tests include path setup to import modules from parent directory
