"""
Integration test for end-to-end supply chain optimization workflow.

This test validates the complete pipeline with a small dataset:
- 1 store, 1 product, 20 weeks of data
- Verifies data flows between all modules
- Validates output formats and data consistency
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from auxiliar.auxiliar import generate_data
from forecaster.forecaster import Forecaster
from optimizer.optimizer import Optimizer
from router.router import Router


class TestEndToEndWorkflow:
    """Integration tests for the complete supply chain optimization pipeline."""
    
    @pytest.fixture
    def small_dataset(self):
        """Generate a small dataset for testing: 1 store, 1 product, 20 weeks."""
        data = generate_data(
            n_stores=1,
            n_products=1,
            n_weeks=20,
            start_date='2024-01-01',
            seed=42
        )
        return data
    
    def test_complete_pipeline(self, small_dataset):
        """
        Test the complete end-to-end workflow with small dataset.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.7
        """
        # Verify initial data
        assert len(small_dataset) == 20, "Should have 20 weeks of data"
        assert small_dataset['store'].nunique() == 1, "Should have 1 store"
        assert small_dataset['product'].nunique() == 1, "Should have 1 product"
        
        # Stage 1: Forecasting
        # Split data: use first 16 weeks for training, forecast 4 weeks
        split_date = small_dataset['date'].max() - pd.Timedelta(weeks=4)
        train_data = small_dataset[small_dataset['date'] <= split_date]
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W',
            forecast_horizon=4
        )
        
        forecaster.fit(train_data)
        forecasts = forecaster.predict(small_dataset)
        
        # Verify forecaster output format
        self._verify_forecaster_output(forecasts, small_dataset)
        
        # Stage 2: Inventory Optimization
        optimizer = Optimizer(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            inv_col='inventory',
            planning_horizon=8,
            service_level=0.95,
            review_period=1,
            lead_time=2
        )
        
        inventory_plan = optimizer.simulate(forecasts)
        
        # Verify optimizer output format
        self._verify_optimizer_output(inventory_plan, forecasts)
        
        # Stage 3: Delivery Routing
        router = Router(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            max_payload=10,
            origin='08020'
        )
        
        # Generate customers
        router.generate_customers(n_customers=5)
        
        # Distribute demand
        deliveries = router.distribute_demand(forecasts)
        
        # Check if there are deliveries to route
        if len(deliveries) == 0:
            print("⚠ No deliveries generated (forecast may have no positive sales)")
            # Skip routing tests if no deliveries
            print("✓ Pipeline completed (no deliveries to route)")
            return
        
        # Assign trucks
        deliveries = router.assign_trucks(deliveries)
        
        # Optimize routes
        deliveries, routes = router.optimize_routes(deliveries)
        
        # Verify router output format
        self._verify_router_output(deliveries, routes, forecasts)
        
        # Validate data flows between modules
        self._validate_data_flows(small_dataset, forecasts, inventory_plan, deliveries)
        
        print("\n✅ End-to-end integration test passed!")
    
    def _verify_forecaster_output(self, forecasts, original_data):
        """
        Verify forecaster output format and content.
        
        Requirements: 6.2
        """
        # Check required columns
        required_cols = ['store', 'product', 'date', 'sales', 'sample', 'prediction']
        assert all(col in forecasts.columns for col in required_cols), \
            f"Forecasts missing required columns. Expected: {required_cols}, Got: {list(forecasts.columns)}"
        
        # Check sample values
        assert set(forecasts['sample'].unique()).issubset({'train', 'test'}), \
            "Sample column should only contain 'train' or 'test'"
        
        # Check that train samples have NaN predictions
        train_data = forecasts[forecasts['sample'] == 'train']
        assert train_data['prediction'].isna().all(), \
            "Train samples should have NaN predictions"
        
        # Check that test samples have non-NaN predictions
        test_data = forecasts[forecasts['sample'] == 'test']
        assert not test_data['prediction'].isna().any(), \
            "Test samples should have non-NaN predictions"
        
        # Check that predictions are non-negative
        assert (test_data['prediction'] >= 0).all(), \
            "Predictions should be non-negative"
        
        # Check data continuity
        assert len(forecasts) >= len(original_data), \
            "Forecasts should include at least all original data points"
        
        print("✓ Forecaster output format verified")
    
    def _verify_optimizer_output(self, inventory_plan, forecasts):
        """
        Verify optimizer output format and content.
        
        Requirements: 6.3
        """
        # Check required columns
        required_cols = [
            'store', 'product', 'date', 'sample', 'prediction',
            'safety_stock', 'reorder_point', 'inventory', 'order', 'shipment'
        ]
        assert all(col in inventory_plan.columns for col in required_cols), \
            f"Inventory plan missing required columns. Expected: {required_cols}, Got: {list(inventory_plan.columns)}"
        
        # Check that safety stock and reorder point are non-negative
        assert (inventory_plan['safety_stock'] >= 0).all(), \
            "Safety stock should be non-negative"
        assert (inventory_plan['reorder_point'] >= 0).all(), \
            "Reorder point should be non-negative"
        
        # Check that orders and shipments are non-negative
        assert (inventory_plan['order'] >= 0).all(), \
            "Orders should be non-negative"
        assert (inventory_plan['shipment'] >= 0).all(), \
            "Shipments should be non-negative"
        
        # Check that inventory plan includes test data
        test_data = inventory_plan[inventory_plan['sample'] == 'test']
        assert len(test_data) > 0, \
            "Inventory plan should include test/forecast periods"
        
        # Check that primary keys match forecasts
        assert set(inventory_plan['store'].unique()) == set(forecasts['store'].unique()), \
            "Stores should match between forecasts and inventory plan"
        assert set(inventory_plan['product'].unique()) == set(forecasts['product'].unique()), \
            "Products should match between forecasts and inventory plan"
        
        print("✓ Optimizer output format verified")
    
    def _verify_router_output(self, deliveries, routes, forecasts):
        """
        Verify router output format and content.
        
        Requirements: 6.4
        """
        # Check deliveries required columns
        delivery_cols = [
            'store', 'product', 'date', 'sales', 'truck',
            'customer', 'destination', 'units'
        ]
        assert all(col in deliveries.columns for col in delivery_cols), \
            f"Deliveries missing required columns. Expected: {delivery_cols}, Got: {list(deliveries.columns)}"
        
        # Check routes required columns
        route_cols = ['truck', 'route_order', 'origin', 'destinations', 'total_distance']
        assert all(col in routes.columns for col in route_cols), \
            f"Routes missing required columns. Expected: {route_cols}, Got: {list(routes.columns)}"
        
        # Check that units are non-negative
        assert (deliveries['units'] >= 0).all(), \
            "Delivery units should be non-negative"
        
        # Check that each delivery has a truck assigned
        assert not deliveries['truck'].isna().any(), \
            "All deliveries should have a truck assigned"
        
        # Check that each delivery has a customer and destination
        assert not deliveries['customer'].isna().any(), \
            "All deliveries should have a customer"
        assert not deliveries['destination'].isna().any(), \
            "All deliveries should have a destination"
        
        # Check that route distances are non-negative
        assert (routes['total_distance'] >= 0).all(), \
            "Route distances should be non-negative"
        
        # Check that trucks in deliveries match trucks in routes
        delivery_trucks = set(deliveries['truck'].unique())
        route_trucks = set(routes['truck'].unique())
        assert delivery_trucks == route_trucks, \
            f"Trucks should match between deliveries and routes. Deliveries: {delivery_trucks}, Routes: {route_trucks}"
        
        # Check that primary keys match forecasts
        assert set(deliveries['store'].unique()) == set(forecasts['store'].unique()), \
            "Stores should match between forecasts and deliveries"
        assert set(deliveries['product'].unique()) == set(forecasts['product'].unique()), \
            "Products should match between forecasts and deliveries"
        
        print("✓ Router output format verified")
    
    def _validate_data_flows(self, original_data, forecasts, inventory_plan, deliveries):
        """
        Validate that data flows correctly between modules.
        
        Requirements: 6.7
        """
        # Check that primary keys are preserved
        original_stores = set(original_data['store'].unique())
        forecast_stores = set(forecasts['store'].unique())
        inventory_stores = set(inventory_plan['store'].unique())
        delivery_stores = set(deliveries['store'].unique())
        
        assert original_stores == forecast_stores == inventory_stores == delivery_stores, \
            "Store primary keys should be preserved across all modules"
        
        original_products = set(original_data['product'].unique())
        forecast_products = set(forecasts['product'].unique())
        inventory_products = set(inventory_plan['product'].unique())
        delivery_products = set(deliveries['product'].unique())
        
        assert original_products == forecast_products == inventory_products == delivery_products, \
            "Product primary keys should be preserved across all modules"
        
        # Check that date ranges are consistent
        original_dates = set(original_data['date'].unique())
        forecast_dates = set(forecasts['date'].unique())
        
        # Forecasts should include all original dates plus forecast periods
        assert original_dates.issubset(forecast_dates), \
            "Forecasts should include all original dates"
        
        # Check that test data exists in all modules
        assert (forecasts['sample'] == 'test').any(), \
            "Forecasts should have test data"
        assert (inventory_plan['sample'] == 'test').any(), \
            "Inventory plan should have test data"
        
        # Check that deliveries sum approximately matches forecast sales
        # (may not be exact due to customer distribution)
        forecast_test = forecasts[forecasts['sample'] == 'test']
        total_forecast_sales = forecast_test['prediction'].sum()
        total_delivery_units = deliveries['units'].sum()
        
        # Allow for small rounding differences
        assert abs(total_forecast_sales - total_delivery_units) < 1.0, \
            f"Total delivery units ({total_delivery_units}) should approximately match forecast sales ({total_forecast_sales})"
        
        print("✓ Data flows validated across all modules")
    
    def test_pipeline_with_multiple_stores_products(self):
        """
        Test pipeline with slightly larger dataset to ensure scalability.
        
        Requirements: 6.1, 6.7
        """
        # Generate data with 2 stores, 2 products, 20 weeks
        data = generate_data(
            n_stores=2,
            n_products=2,
            n_weeks=20,
            start_date='2024-01-01',
            seed=123
        )
        
        assert len(data) == 80, "Should have 2 stores × 2 products × 20 weeks = 80 records"
        
        # Run forecasting
        split_date = data['date'].max() - pd.Timedelta(weeks=4)
        train_data = data[data['date'] <= split_date]
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            forecast_horizon=4
        )
        forecaster.fit(train_data)
        forecasts = forecaster.predict(data)
        
        # Verify multiple time series
        test_data = forecasts[forecasts['sample'] == 'test']
        n_series = test_data.groupby(['store', 'product']).ngroups
        assert n_series == 4, "Should have 4 time series (2 stores × 2 products)"
        
        # Run optimizer
        optimizer = Optimizer(
            primary_keys=['store', 'product'],
            planning_horizon=8
        )
        inventory_plan = optimizer.simulate(forecasts)
        
        # Verify inventory for all series
        inv_test = inventory_plan[inventory_plan['sample'] == 'test']
        n_inv_series = inv_test.groupby(['store', 'product']).ngroups
        assert n_inv_series == 4, "Should have inventory plans for all 4 series"
        
        # Run router
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=20
        )
        router.generate_customers(n_customers=10)
        deliveries = router.distribute_demand(forecasts)
        deliveries = router.assign_trucks(deliveries)
        deliveries, routes = router.optimize_routes(deliveries)
        
        # Verify routing for all series
        n_delivery_series = deliveries.groupby(['store', 'product']).ngroups
        assert n_delivery_series == 4, "Should have deliveries for all 4 series"
        
        print("✓ Pipeline works with multiple stores and products")
    
    def test_pipeline_error_handling(self):
        """
        Test that pipeline handles edge cases gracefully.
        
        Requirements: 6.7
        """
        # Test with minimal data
        data = generate_data(
            n_stores=1,
            n_products=1,
            n_weeks=10,  # Minimal weeks
            start_date='2024-01-01',
            seed=999
        )
        
        # Should not raise errors
        split_date = data['date'].max() - pd.Timedelta(weeks=2)
        train_data = data[data['date'] <= split_date]
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            forecast_horizon=2
        )
        
        # This should work even with minimal data
        forecaster.fit(train_data)
        forecasts = forecaster.predict(data)
        
        assert len(forecasts) > 0, "Should produce forecasts even with minimal data"
        
        print("✓ Pipeline handles edge cases")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
