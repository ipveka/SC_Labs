# Implementation Plan

- [x] 1. Set up project structure and dependencies
  - Create directory structure: sc_labs/ with subdirectories for forecaster/, optimizer/, router/, auxiliar/, and docs/
  - Create __init__.py files in each module's utils/ subdirectory
  - Create requirements.txt with pandas, numpy, gluonts, mxnet, scipy
  - _Requirements: 5.1, 5.2_

- [x] 2. Implement Auxiliar module for synthetic data generation
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 2.1 Create generate_data() function in auxiliar/auxiliar.py
  - Implement function signature with parameters: n_stores, n_products, n_weeks, start_date, seed
  - Generate date range using pd.date_range() with weekly frequency
  - Create all combinations of stores (A, B, C...) and products (X, Y...)
  - For each combination, generate realistic sales with trend + seasonality + noise
  - Assign random initial inventory levels (200-500 range)
  - Generate random customer IDs and postal codes
  - Return DataFrame with columns: store, product, date, sales, inventory, customer_id, destination
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 2.2 Write unit tests for data generation
  - Test output schema matches specification
  - Test data dimensions (rows = n_stores × n_products × n_weeks)
  - Test reproducibility with same seed
  - Test non-negative sales values
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 2.3 Create auxiliar.md documentation
  - Document module overview and purpose
  - Document function parameters and return values
  - Include example output table
  - Describe data generation logic (trend, seasonality, noise)
  - _Requirements: 5.3, 5.4_

- [x] 3. Implement Forecaster module for demand forecasting
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [x] 3.1 Create Forecaster class in forecaster/forecaster.py
  - Implement __init__ method with parameters: primary_keys, date_col, target_col, frequency, forecast_horizon
  - Add type hints for all parameters
  - Initialize model attribute as None
  - Add docstring explaining class purpose and parameters
  - _Requirements: 1.1_

- [x] 3.2 Implement prepare_data() method
  - Aggregate data by primary_keys and date_col
  - Ensure continuous date range using pd.date_range()
  - Fill missing values with forward fill method
  - Handle edge cases (empty DataFrame, single row)
  - Return cleaned DataFrame
  - _Requirements: 1.2, 1.6_

- [x] 3.3 Implement fit() method
  - Convert DataFrame to GluonTS PandasDataset format
  - Group data by primary_keys to create multiple time series
  - Initialize SimpleFeedForwardEstimator with prediction_length=forecast_horizon
  - Configure Trainer with epochs=10, learning_rate=1e-3
  - Train model and store in self.model
  - Add error handling for training failures
  - _Requirements: 1.3_

- [x] 3.4 Implement predict() method
  - Use trained model to generate forecasts for forecast_horizon periods
  - Create predictions for each primary_key combination
  - Combine historical data with sample='train' and prediction=NaN
  - Add forecast data with sample='test' and prediction values
  - Return unified DataFrame with all required columns
  - _Requirements: 1.4, 1.5_

- [x] 3.5 Write unit tests for Forecaster
  - Test prepare_data() with missing dates
  - Test fit() with minimal valid dataset
  - Test predict() output schema and dimensions
  - Test handling of multiple time series
  - _Requirements: 1.2, 1.3, 1.4_

- [x] 3.6 Create forecaster.md documentation
  - Document module overview and GluonTS usage
  - Document class initialization parameters
  - Document each method with inputs/outputs
  - Include example forecast output table
  - _Requirements: 5.3, 5.4_

- [x] 4. Implement Optimizer module for inventory simulation
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

- [x] 4.1 Create Optimizer class in optimizer/optimizer.py
  - Implement __init__ method with parameters: primary_keys, date_col, target_col, inv_col, planning_horizon, service_level, review_period, lead_time
  - Add type hints and parameter validation
  - Add docstring explaining inventory policy
  - _Requirements: 2.1_

- [x] 4.2 Implement calculate_safety_stock() method
  - Calculate standard deviation of demand series
  - Compute z-score from service_level using scipy.stats.norm.ppf()
  - Return safety stock: z_score * std_dev * sqrt(lead_time)
  - Handle edge case of zero variance
  - _Requirements: 2.2, 2.3_

- [x] 4.3 Implement calculate_reorder_point() method
  - Calculate reorder point: (avg_demand * lead_time) + safety_stock
  - Return reorder point value
  - _Requirements: 2.3_

- [x] 4.4 Implement simulate() method
  - Group input DataFrame by primary_keys
  - For each group, calculate safety stock and reorder point
  - Initialize inventory from inv_col
  - Iterate through planning_horizon periods
  - Deduct forecasted demand from inventory each period
  - Check if inventory < reorder_point on review periods
  - Place orders when needed (quantity = reorder_point - current_inventory)
  - Track orders with lead_time counter
  - Add shipments when orders arrive
  - Record inventory, orders, shipments per period
  - Return consolidated DataFrame with all simulation columns
  - _Requirements: 2.4, 2.5, 2.6, 2.7, 2.8_

- [x] 4.5 Write unit tests for Optimizer
  - Test safety stock calculation with known variance
  - Test reorder point calculation
  - Test simulation with zero lead time
  - Test order placement logic at reorder point
  - Test shipment arrival after lead time
  - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 4.6 Create optimizer.md documentation
  - Document module overview and inventory policy
  - Document class parameters and their impact
  - Document simulation logic flow
  - Include example simulation output table
  - _Requirements: 5.3, 5.4_

- [x] 5. Implement Router module for delivery routing
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [x] 5.1 Create Router class in router/router.py
  - Implement __init__ method with parameters: primary_keys, date_col, target_col, max_payload, origin
  - Initialize customers_db as None
  - Add type hints and docstring
  - _Requirements: 3.1_

- [x] 5.2 Implement generate_customers() method
  - Create n_customers customer records with unique IDs
  - Generate random 5-digit postal codes
  - Return DataFrame with columns: customer_id, destination
  - Store in self.customers_db
  - _Requirements: 3.2_

- [x] 5.3 Implement distribute_demand() method
  - For each (store, product, date) combination in forecast data
  - Randomly select customers from customers_db
  - Split sales quantity evenly across selected customers
  - Create delivery records with units per customer
  - Return DataFrame: store, product, date, sales, customer, destination, units
  - _Requirements: 3.3_

- [x] 5.4 Implement assign_trucks() method
  - Group deliveries by date
  - For each date, sort deliveries by destination for locality
  - Assign deliveries to trucks sequentially
  - Ensure each truck's payload ≤ max_payload
  - Create new truck when capacity exceeded
  - Add truck column (truck_1, truck_2, etc.)
  - Return updated deliveries DataFrame
  - _Requirements: 3.4_

- [x] 5.5 Implement optimize_routes() method
  - Group deliveries by truck
  - For each truck, extract unique destinations
  - Calculate simple distance heuristic (postal code numeric difference)
  - Order destinations using nearest neighbor heuristic
  - Calculate total route distance
  - Return delivery-level data and route summary DataFrame
  - Route summary columns: truck, route_order, origin, destinations, total_distance
  - _Requirements: 3.5, 3.6, 3.7, 3.8_

- [x] 5.6 Write unit tests for Router
  - Test customer generation count and uniqueness
  - Test demand distribution sums to original forecast
  - Test truck assignment respects max_payload
  - Test route optimization ordering
  - _Requirements: 3.2, 3.3, 3.4, 3.5_

- [x] 5.7 Create router.md documentation
  - Document module overview and routing approach
  - Document class parameters and methods
  - Document routing algorithm (nearest neighbor heuristic)
  - Include example delivery and route summary tables
  - _Requirements: 5.3, 5.4_

- [x] 6. Implement main orchestration script
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 6.1 Create main.py with workflow orchestration
  - Import all modules (auxiliar, Forecaster, Optimizer, Router)
  - Implement main() function with complete workflow
  - Generate synthetic data using auxiliar.generate_data()
  - Split data into train/test sets
  - Initialize and run Forecaster: fit() then predict()
  - Initialize and run Optimizer: simulate()
  - Initialize and run Router: generate_customers(), distribute_demand(), assign_trucks(), optimize_routes()
  - Add error handling for each stage
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.7_

- [x] 6.2 Implement print_summary() helper function
  - Display summary statistics for forecasts (RMSE, MAE if actuals available)
  - Display inventory summary (avg inventory, stockouts, orders placed)
  - Display routing summary (total trucks, total distance, avg payload)
  - Format output tables using pandas display options
  - _Requirements: 6.5_

- [x] 6.3 Implement save_outputs() helper function
  - Create outputs/ directory if it doesn't exist
  - Save forecasts to outputs/forecasts.csv
  - Save inventory plan to outputs/inventory_plan.csv
  - Save deliveries to outputs/deliveries.csv
  - Save routes to outputs/routes.csv
  - Add timestamp to filenames for versioning
  - _Requirements: 6.6_

- [x] 6.4 Add command-line interface
  - Use argparse to accept parameters (n_stores, n_products, n_weeks, etc.)
  - Add --save flag to enable output saving
  - Add --verbose flag for detailed logging
  - Add if __name__ == "__main__": block
  - _Requirements: 6.1, 6.6_

- [x] 6.5 Write integration test for end-to-end workflow






  - Test complete pipeline with small dataset (1 store, 1 product, 20 weeks)
  - Verify forecaster output format
  - Verify optimizer output format
  - Verify router output format
  - Validate data flows between modules
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.7_

- [x] 7. Final integration and documentation
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [x] 7.1 Create project README.md
  - Add project overview and goals
  - Add installation instructions (pip install -r requirements.txt)
  - Add usage examples (running main.py)
  - Add folder structure diagram
  - Add links to module documentation
  - Include sample output screenshots or tables
  - _Requirements: 5.3, 5.4, 5.5_

- [x] 7.2 Verify all module documentation is complete
  - Check forecaster.md completeness
  - Check optimizer.md completeness
  - Check router.md completeness
  - Check auxiliar.md completeness
  - Ensure consistent formatting across all docs
  - _Requirements: 5.3, 5.4_

- [x] 7.3 Add inline code comments and docstrings
  - Review all classes for complete docstrings
  - Review all methods for parameter and return documentation
  - Add inline comments for complex logic sections
  - Ensure type hints are present throughout
  - _Requirements: 5.5, 5.6_

- [x] 7.4 Create requirements.txt with pinned versions
  - List all required packages with version constraints
  - Test installation in clean virtual environment
  - Document Python version requirement (>=3.10)
  - _Requirements: 5.1_

- [ ]* 7.5 Run full test suite and verify coverage
  - Execute all unit tests
  - Execute integration test
  - Verify all tests pass
  - Check test coverage is reasonable (>70%)
  - _Requirements: 5.6_
