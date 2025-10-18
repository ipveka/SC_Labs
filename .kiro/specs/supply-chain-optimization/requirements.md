# Requirements Document

## Introduction

This document outlines the requirements for SC Labs, a modular End-to-End Supply Chain Optimization Proof of Concept (PoC). The system will provide demand forecasting, inventory optimization, and delivery routing capabilities through clean, documented Python modules. The solution will use synthetic data generation for demonstration purposes and leverage GluonTS for time series forecasting.

## Requirements

### Requirement 1: Demand Forecasting Module

**User Story:** As a supply chain analyst, I want to forecast future demand for each store-product combination, so that I can plan inventory and orders proactively.

#### Acceptance Criteria

1. WHEN the Forecaster class is initialized with primary keys, date column, target column, frequency, and forecast horizon THEN the system SHALL store these configuration parameters for subsequent operations
2. WHEN prepare_data() is called with a DataFrame THEN the system SHALL aggregate data by primary keys, ensure continuous date index, and handle missing values
3. WHEN fit() is called with training data THEN the system SHALL train a GluonTS model (SimpleFeedForwardEstimator) on historical time series for each primary key combination
4. WHEN predict() is called after model training THEN the system SHALL generate forecasts for the specified forecast_horizon periods
5. WHEN predictions are generated THEN the system SHALL return a DataFrame containing columns: primary_keys (store, product), date, sales, sample ('train' or 'test'), and prediction (NaN for train, forecasted value for test)
6. IF the input data has gaps or missing dates THEN the system SHALL fill missing periods to maintain continuous time series

### Requirement 2: Inventory Optimization Module

**User Story:** As an inventory manager, I want to simulate replenishment policies based on forecasted demand, so that I can maintain optimal stock levels while meeting service level targets.

#### Acceptance Criteria

1. WHEN the Optimizer class is initialized with primary keys, date column, target column, inventory column, planning horizon, service level, review period, and lead time THEN the system SHALL store these parameters for simulation
2. WHEN the optimizer processes forecasted demand THEN the system SHALL calculate demand variability and z-score based on the specified service level
3. WHEN safety stock is computed THEN the system SHALL use demand variability and service level to determine appropriate buffer inventory
4. WHEN the simulation runs period by period THEN the system SHALL deduct forecasted demand from current inventory levels
5. WHEN inventory falls below the reorder point THEN the system SHALL place an order for replenishment
6. WHEN an order reaches its lead time THEN the system SHALL add the shipment quantity to inventory
7. WHEN the simulation completes THEN the system SHALL return a DataFrame with columns: primary_keys (store, product), date, sample, prediction, safety_stock, reorder_point, inventory, order, and shipment
8. IF review_period is greater than 1 THEN the system SHALL only check inventory and place orders at the specified review frequency

### Requirement 3: Delivery Routing Module

**User Story:** As a logistics coordinator, I want to distribute forecasted sales among customers and optimize delivery routes, so that I can minimize transportation costs and meet delivery commitments.

#### Acceptance Criteria

1. WHEN the Router class is initialized with primary keys, date column, target column, max payload, and origin postal code THEN the system SHALL store these routing parameters
2. WHEN customer generation is triggered THEN the system SHALL create sample customers with unique customer_id and destination postal codes
3. WHEN forecasted sales are distributed THEN the system SHALL evenly split sales per (store, product, date) combination across generated customers
4. WHEN deliveries are assigned to trucks THEN the system SHALL ensure each truck's payload does not exceed max_payload
5. WHEN route optimization is performed THEN the system SHALL start from the origin location and minimize total travel distance
6. WHEN routing is complete THEN the system SHALL return delivery-level data with columns: store, product, date, sales, truck, customer, destination, units
7. WHEN routing is complete THEN the system SHALL return route summary data with columns: truck, route_order, origin, destinations, total_distance
8. IF a simple distance heuristic is used THEN the system SHALL calculate distances based on postal code proximity or coordinate-based calculations

### Requirement 4: Synthetic Data Generation

**User Story:** As a developer, I want to generate realistic synthetic supply chain data, so that I can demonstrate and test the system without requiring real business data.

#### Acceptance Criteria

1. WHEN generate_data() is called with parameters (n_stores, n_products, n_weeks, start_date) THEN the system SHALL create a synthetic dataset with the specified dimensions
2. WHEN time series data is generated THEN the system SHALL include realistic patterns with trend and random noise components
3. WHEN the synthetic dataset is created THEN the system SHALL include columns: store, product, date, sales, inventory, customer_id, destination
4. WHEN base inventory is assigned THEN the system SHALL use random but realistic initial inventory levels for each store-product combination
5. WHEN customer data is generated THEN the system SHALL assign random customer IDs and postal codes for routing simulation
6. WHEN the function completes THEN the system SHALL return a pandas DataFrame ready for use by other modules

### Requirement 5: Project Structure and Documentation

**User Story:** As a developer or stakeholder, I want clear project organization and comprehensive documentation, so that I can understand, maintain, and extend the system easily.

#### Acceptance Criteria

1. WHEN the project is structured THEN the system SHALL organize code into separate modules: forecaster/, optimizer/, router/, and auxiliar/
2. WHEN each module is created THEN the system SHALL include a corresponding utils/ subdirectory with __init__.py for extensibility
3. WHEN documentation is provided THEN the system SHALL include separate markdown files in docs/ for each module (forecaster.md, optimizer.md, router.md, auxiliar.md)
4. WHEN each documentation file is created THEN the system SHALL include: module overview, inputs and outputs description, logic summary, and example tables or flow descriptions
5. WHEN code is written THEN the system SHALL include clear docstrings, type hints, and inline comments following Python best practices
6. WHEN classes and functions are defined THEN the system SHALL use object-oriented design principles with modular, extendable architecture

### Requirement 6: Main Orchestration Script

**User Story:** As a user, I want a single entry point that orchestrates the entire supply chain optimization workflow, so that I can run the complete pipeline with minimal effort.

#### Acceptance Criteria

1. WHEN main.py is executed THEN the system SHALL generate synthetic data using auxiliar.generate_data()
2. WHEN the Forecaster is initialized and run THEN the system SHALL call fit() on training data and predict() to generate forecasts
3. WHEN the Optimizer is initialized and run THEN the system SHALL simulate inventory flow using the forecasted demand
4. WHEN the Router is initialized and run THEN the system SHALL distribute demand to customers and optimize delivery routes
5. WHEN all modules complete execution THEN the system SHALL display summary tables showing key results from each stage
6. WHEN results are generated THEN the system SHALL optionally save output files to an /outputs/ directory
7. WHEN the workflow executes THEN the system SHALL handle data flow between modules seamlessly, passing outputs from one module as inputs to the next
