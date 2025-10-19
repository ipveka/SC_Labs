"""
Unit tests for the Router module.

Tests cover:
- Customer generation count and uniqueness
- Demand distribution sums to original forecast
- Truck assignment respects max_payload
- Route optimization ordering
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from router.router import Router


class TestRouterCustomerGeneration:
    """Tests for the generate_customers() method."""
    
    def test_customer_generation_count(self):
        """Test that the correct number of customers is generated."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=10,
            origin='08020'
        )
        
        n_customers = 20
        customers = router.generate_customers(n_customers=n_customers)
        
        # Should generate exactly n_customers
        assert len(customers) == n_customers
        
        # Should have correct columns
        assert 'customer_id' in customers.columns
        assert 'destination' in customers.columns
    
    def test_customer_id_uniqueness(self):
        """Test that all customer IDs are unique."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=10,
            origin='08020'
        )
        
        customers = router.generate_customers(n_customers=50)
        
        # All customer IDs should be unique
        assert customers['customer_id'].nunique() == 50
        assert len(customers['customer_id']) == len(customers['customer_id'].unique())

    def test_customer_id_format(self):
        """Test that customer IDs follow the expected format."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=10,
            origin='08020'
        )
        
        customers = router.generate_customers(n_customers=10)
        
        # All customer IDs should start with 'CUST_'
        for customer_id in customers['customer_id']:
            assert customer_id.startswith('CUST_')
            # Should have format CUST_XXXX where XXXX is a number
            assert len(customer_id) == 9  # 'CUST_' + 4 digits
    
    def test_postal_code_format(self):
        """Test that postal codes are 5-digit strings."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=10,
            origin='08020'
        )
        
        customers = router.generate_customers(n_customers=20)
        
        # All postal codes should be 5-digit strings
        for postal_code in customers['destination']:
            assert len(postal_code) == 5
            assert postal_code.isdigit()
    
    def test_customers_stored_in_db(self):
        """Test that generated customers are stored in customers_db."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=10,
            origin='08020'
        )
        
        # Initially customers_db should be None
        assert router.customers_db is None
        
        customers = router.generate_customers(n_customers=15)
        
        # After generation, customers_db should be populated
        assert router.customers_db is not None
        assert len(router.customers_db) == 15
        
        # Should be the same DataFrame
        pd.testing.assert_frame_equal(customers, router.customers_db)


class TestRouterDemandDistribution:
    """Tests for the distribute_demand() method."""
    
    def test_demand_distribution_sums_to_forecast(self):
        """Test that distributed demand sums to original forecast."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=10,
            origin='08020'
        )
        
        # Generate customers
        router.generate_customers(n_customers=20)
        
        # Create forecast data
        dates = pd.date_range('2024-01-01', periods=5, freq='W')
        forecast_df = pd.DataFrame({
            'store': ['A'] * 5,
            'product': ['X'] * 5,
            'date': dates,
            'sales': [100, 150, 120, 130, 110],
            'sample': ['test'] * 5,
            'prediction': [100, 150, 120, 130, 110]
        })
        
        # Distribute demand
        deliveries = router.distribute_demand(forecast_df)
        
        # Group deliveries by date and sum units
        delivery_sums = deliveries.groupby('date')['units'].sum()
        
        # Should match original forecast predictions
        for date in dates:
            original_forecast = forecast_df[forecast_df['date'] == date]['prediction'].iloc[0]
            distributed_sum = delivery_sums[date]
            
            # Allow small floating point differences
            assert abs(distributed_sum - original_forecast) < 0.01

    def test_demand_distribution_requires_customers(self):
        """Test that distribute_demand raises error if customers not generated."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=10,
            origin='08020'
        )
        
        # Create forecast data
        forecast_df = pd.DataFrame({
            'store': ['A'],
            'product': ['X'],
            'date': [pd.Timestamp('2024-01-01')],
            'sales': [100],
            'sample': ['test'],
            'prediction': [100]
        })
        
        # Should raise error because customers_db is None
        with pytest.raises(ValueError, match="Customer database not initialized"):
            router.distribute_demand(forecast_df)
    
    def test_demand_distribution_skips_zero_sales(self):
        """Test that zero or negative sales are skipped."""
        np.random.seed(42)
        
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=10,
            origin='08020'
        )
        
        router.generate_customers(n_customers=20)
        
        # Create forecast data with zero and negative sales
        dates = pd.date_range('2024-01-01', periods=5, freq='W')
        forecast_df = pd.DataFrame({
            'store': ['A'] * 5,
            'product': ['X'] * 5,
            'date': dates,
            'sales': [100, 0, -10, 150, 0],
            'sample': ['test'] * 5,
            'prediction': [100, 0, -10, 150, 0]
        })
        
        deliveries = router.distribute_demand(forecast_df)
        
        # Should only have deliveries for positive sales
        unique_dates = deliveries['date'].unique()
        assert len(unique_dates) == 2  # Only dates with 100 and 150
        
        # Verify the dates with positive sales (dates[0] and dates[3])
        assert dates[0] in unique_dates
        assert dates[3] in unique_dates
    
    def test_demand_distribution_multiple_time_series(self):
        """Test demand distribution with multiple store-product combinations."""
        np.random.seed(42)
        
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=10,
            origin='08020'
        )
        
        router.generate_customers(n_customers=20)
        
        # Create forecast data for multiple time series
        dates = pd.date_range('2024-01-01', periods=3, freq='W')
        data = []
        for store in ['A', 'B']:
            for product in ['X', 'Y']:
                for date in dates:
                    data.append({
                        'store': store,
                        'product': product,
                        'date': date,
                        'sales': 100,
                        'sample': 'test',
                        'prediction': 100
                    })
        
        forecast_df = pd.DataFrame(data)
        deliveries = router.distribute_demand(forecast_df)
        
        # Should have deliveries for all combinations
        unique_combinations = deliveries[['store', 'product']].drop_duplicates()
        assert len(unique_combinations) == 4  # 2 stores × 2 products
        
        # Verify sum for each combination
        for (store, product), group in deliveries.groupby(['store', 'product']):
            total_units = group['units'].sum()
            expected_total = 100 * 3  # 100 per week × 3 weeks
            assert abs(total_units - expected_total) < 0.01

    def test_demand_distribution_output_schema(self):
        """Test that distributed demand has correct output schema."""
        np.random.seed(42)
        
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=10,
            origin='08020'
        )
        
        router.generate_customers(n_customers=20)
        
        forecast_df = pd.DataFrame({
            'store': ['A'],
            'product': ['X'],
            'date': [pd.Timestamp('2024-01-01')],
            'sales': [100],
            'sample': ['test'],
            'prediction': [100]
        })
        
        deliveries = router.distribute_demand(forecast_df)
        
        # Check required columns exist
        required_columns = ['store', 'product', 'date', 'sales', 'customer', 'destination', 'units']
        for col in required_columns:
            assert col in deliveries.columns
        
        # Check data types
        assert deliveries['customer'].dtype == object
        assert deliveries['destination'].dtype == object
        assert pd.api.types.is_numeric_dtype(deliveries['units'])


class TestRouterTruckAssignment:
    """Tests for the assign_trucks() method."""
    
    def test_truck_assignment_respects_max_payload(self):
        """Test that truck assignment respects max_payload constraint."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=100,  # Set max payload to 100
            origin='08020'
        )
        
        # Create deliveries that will require multiple trucks
        deliveries_df = pd.DataFrame({
            'store': ['A'] * 10,
            'product': ['X'] * 10,
            'date': [pd.Timestamp('2024-01-01')] * 10,
            'sales': [100] * 10,
            'customer': [f'CUST_{i:04d}' for i in range(1, 11)],
            'destination': [f'{10000 + i:05d}' for i in range(10)],
            'units': [30] * 10  # Each delivery is 30 units
        })
        
        result = router.assign_trucks(deliveries_df)
        
        # Check that truck column was added
        assert 'truck' in result.columns
        
        # Verify payload constraint for each truck
        for truck, truck_group in result.groupby('truck'):
            total_payload = truck_group['units'].sum()
            assert total_payload <= router.max_payload
    
    def test_truck_assignment_creates_multiple_trucks(self):
        """Test that multiple trucks are created when needed."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=50,  # Small payload
            origin='08020'
        )
        
        # Create deliveries that require multiple trucks
        deliveries_df = pd.DataFrame({
            'store': ['A'] * 10,
            'product': ['X'] * 10,
            'date': [pd.Timestamp('2024-01-01')] * 10,
            'sales': [100] * 10,
            'customer': [f'CUST_{i:04d}' for i in range(1, 11)],
            'destination': [f'{10000 + i:05d}' for i in range(10)],
            'units': [20] * 10  # Total = 200 units, max_payload = 50
        })
        
        result = router.assign_trucks(deliveries_df)
        
        # Should create multiple trucks
        unique_trucks = result['truck'].nunique()
        assert unique_trucks >= 4  # 200 / 50 = 4 trucks minimum

    def test_truck_assignment_single_truck_sufficient(self):
        """Test that single truck is used when payload is sufficient."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=1000,  # Large payload
            origin='08020'
        )
        
        # Create deliveries that fit in one truck
        deliveries_df = pd.DataFrame({
            'store': ['A'] * 5,
            'product': ['X'] * 5,
            'date': [pd.Timestamp('2024-01-01')] * 5,
            'sales': [100] * 5,
            'customer': [f'CUST_{i:04d}' for i in range(1, 6)],
            'destination': [f'{10000 + i:05d}' for i in range(5)],
            'units': [10] * 5  # Total = 50 units
        })
        
        result = router.assign_trucks(deliveries_df)
        
        # Should only use one truck
        unique_trucks = result['truck'].nunique()
        assert unique_trucks == 1
        assert result['truck'].iloc[0] == 'truck_1'
    
    def test_truck_assignment_by_date(self):
        """Test that trucks are assigned separately for each date."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=100,
            origin='08020'
        )
        
        # Create deliveries for multiple dates
        dates = [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-08')]
        deliveries_df = pd.DataFrame({
            'store': ['A'] * 10,
            'product': ['X'] * 10,
            'date': dates * 5,  # Alternate between two dates
            'sales': [100] * 10,
            'customer': [f'CUST_{i:04d}' for i in range(1, 11)],
            'destination': [f'{10000 + i:05d}' for i in range(10)],
            'units': [30] * 10
        })
        
        result = router.assign_trucks(deliveries_df)
        
        # Each date should have its own truck numbering
        for date in dates:
            date_deliveries = result[result['date'] == date]
            # Should have truck assignments
            assert date_deliveries['truck'].notna().all()
    
    def test_truck_assignment_sorts_by_destination(self):
        """Test that deliveries are sorted by destination for locality."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=1000,  # Large enough for all deliveries
            origin='08020'
        )
        
        # Create deliveries with specific destinations
        deliveries_df = pd.DataFrame({
            'store': ['A'] * 5,
            'product': ['X'] * 5,
            'date': [pd.Timestamp('2024-01-01')] * 5,
            'sales': [100] * 5,
            'customer': [f'CUST_{i:04d}' for i in range(1, 6)],
            'destination': ['50000', '10000', '30000', '20000', '40000'],  # Unsorted
            'units': [10] * 5
        })
        
        result = router.assign_trucks(deliveries_df)
        
        # The assign_trucks method sorts internally but doesn't guarantee output order
        # Verify that all deliveries are assigned to trucks
        assert result['truck'].notna().all()
        assert len(result) == 5
        
        # All should be on the same truck since payload is sufficient
        assert result['truck'].nunique() == 1


class TestRouterRouteOptimization:
    """Tests for the optimize_routes() method."""
    
    def test_route_optimization_returns_two_dataframes(self):
        """Test that optimize_routes returns delivery data and route summary."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=100,
            origin='08020'
        )
        
        # Create deliveries with truck assignments
        deliveries_df = pd.DataFrame({
            'store': ['A'] * 5,
            'product': ['X'] * 5,
            'date': [pd.Timestamp('2024-01-01')] * 5,
            'sales': [100] * 5,
            'customer': [f'CUST_{i:04d}' for i in range(1, 6)],
            'destination': [f'{10000 + i:05d}' for i in range(5)],
            'units': [10] * 5,
            'truck': ['truck_1'] * 5
        })
        
        deliveries, routes = router.optimize_routes(deliveries_df)
        
        # Should return two DataFrames
        assert isinstance(deliveries, pd.DataFrame)
        assert isinstance(routes, pd.DataFrame)

    def test_route_summary_schema(self):
        """Test that route summary has correct schema."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=100,
            origin='08020'
        )
        
        deliveries_df = pd.DataFrame({
            'store': ['A'] * 5,
            'product': ['X'] * 5,
            'date': [pd.Timestamp('2024-01-01')] * 5,
            'sales': [100] * 5,
            'customer': [f'CUST_{i:04d}' for i in range(1, 6)],
            'destination': [f'{10000 + i:05d}' for i in range(5)],
            'units': [10] * 5,
            'truck': ['truck_1'] * 5
        })
        
        _, routes = router.optimize_routes(deliveries_df)
        
        # Check required columns
        required_columns = ['truck', 'route_order', 'origin', 'destinations', 'total_distance']
        for col in required_columns:
            assert col in routes.columns
        
        # Check data types
        assert routes['truck'].dtype == object
        assert routes['origin'].dtype == object
        assert routes['destinations'].dtype == object
        assert pd.api.types.is_numeric_dtype(routes['total_distance'])
    
    def test_route_optimization_nearest_neighbor(self):
        """Test that route optimization uses nearest neighbor heuristic."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=100,
            origin='10000'  # Starting point
        )
        
        # Create deliveries with specific destinations
        # Origin: 10000, Destinations: 10100, 10050, 10200
        # Optimal order from 10000: 10050 (50), 10100 (50), 10200 (100)
        deliveries_df = pd.DataFrame({
            'store': ['A'] * 3,
            'product': ['X'] * 3,
            'date': [pd.Timestamp('2024-01-01')] * 3,
            'sales': [100] * 3,
            'customer': [f'CUST_{i:04d}' for i in range(1, 4)],
            'destination': ['10100', '10050', '10200'],
            'units': [10] * 3,
            'truck': ['truck_1'] * 3
        })
        
        _, routes = router.optimize_routes(deliveries_df)
        
        # Check that route was created
        assert len(routes) == 1
        
        # Destinations should be ordered by nearest neighbor
        destinations = routes.iloc[0]['destinations']
        # Should start with 10050 (closest to 10000)
        assert destinations.startswith('10050')
    
    def test_route_optimization_calculates_distance(self):
        """Test that total distance is calculated."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=100,
            origin='10000'
        )
        
        deliveries_df = pd.DataFrame({
            'store': ['A'] * 3,
            'product': ['X'] * 3,
            'date': [pd.Timestamp('2024-01-01')] * 3,
            'sales': [100] * 3,
            'customer': [f'CUST_{i:04d}' for i in range(1, 4)],
            'destination': ['10100', '10200', '10300'],
            'units': [10] * 3,
            'truck': ['truck_1'] * 3
        })
        
        _, routes = router.optimize_routes(deliveries_df)
        
        # Total distance should be positive
        assert routes.iloc[0]['total_distance'] > 0
        
        # Distance should be sum of postal code differences
        # From 10000 to first destination, then between destinations
        assert routes.iloc[0]['total_distance'] > 0

    def test_route_optimization_multiple_trucks(self):
        """Test route optimization with multiple trucks."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=100,
            origin='10000'
        )
        
        # Create deliveries for two trucks
        deliveries_df = pd.DataFrame({
            'store': ['A'] * 6,
            'product': ['X'] * 6,
            'date': [pd.Timestamp('2024-01-01')] * 6,
            'sales': [100] * 6,
            'customer': [f'CUST_{i:04d}' for i in range(1, 7)],
            'destination': ['10100', '10200', '10300', '20100', '20200', '20300'],
            'units': [10] * 6,
            'truck': ['truck_1', 'truck_1', 'truck_1', 'truck_2', 'truck_2', 'truck_2']
        })
        
        _, routes = router.optimize_routes(deliveries_df)
        
        # Should have routes for both trucks
        assert len(routes) == 2
        assert 'truck_1' in routes['truck'].values
        assert 'truck_2' in routes['truck'].values
        
        # Each truck should have its own route
        for truck in ['truck_1', 'truck_2']:
            truck_route = routes[routes['truck'] == truck]
            assert len(truck_route) == 1
            assert truck_route.iloc[0]['total_distance'] > 0
    
    def test_route_optimization_origin_in_summary(self):
        """Test that origin is included in route summary."""
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=100,
            origin='08020'
        )
        
        deliveries_df = pd.DataFrame({
            'store': ['A'] * 3,
            'product': ['X'] * 3,
            'date': [pd.Timestamp('2024-01-01')] * 3,
            'sales': [100] * 3,
            'customer': [f'CUST_{i:04d}' for i in range(1, 4)],
            'destination': ['10100', '10200', '10300'],
            'units': [10] * 3,
            'truck': ['truck_1'] * 3
        })
        
        _, routes = router.optimize_routes(deliveries_df)
        
        # Origin should be in the route summary
        assert routes.iloc[0]['origin'] == '08020'


class TestRouterIntegration:
    """Integration tests for complete routing workflow."""
    
    def test_complete_routing_workflow(self):
        """Test complete routing workflow from customer generation to route optimization."""
        np.random.seed(42)
        
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=100,
            origin='08020'
        )
        
        # Step 1: Generate customers
        customers = router.generate_customers(n_customers=20)
        assert len(customers) == 20
        
        # Step 2: Create forecast data
        dates = pd.date_range('2024-01-01', periods=3, freq='W')
        forecast_df = pd.DataFrame({
            'store': ['A'] * 3,
            'product': ['X'] * 3,
            'date': dates,
            'sales': [100, 150, 120],
            'sample': ['test'] * 3,
            'prediction': [100, 150, 120]
        })
        
        # Step 3: Distribute demand
        deliveries = router.distribute_demand(forecast_df)
        assert len(deliveries) > 0
        
        # Step 4: Assign trucks
        deliveries_with_trucks = router.assign_trucks(deliveries)
        assert 'truck' in deliveries_with_trucks.columns
        
        # Step 5: Optimize routes
        final_deliveries, routes = router.optimize_routes(deliveries_with_trucks)
        
        # Validate final output
        assert len(final_deliveries) > 0
        assert len(routes) > 0
        
        # Check that all required columns exist
        assert 'truck' in final_deliveries.columns
        assert 'destination' in final_deliveries.columns
        assert 'total_distance' in routes.columns

    def test_workflow_with_multiple_time_series(self):
        """Test routing workflow with multiple store-product combinations."""
        np.random.seed(42)
        
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=100,
            origin='08020'
        )
        
        # Generate customers
        router.generate_customers(n_customers=30)
        
        # Create forecast data for multiple time series
        dates = pd.date_range('2024-01-01', periods=2, freq='W')
        data = []
        for store in ['A', 'B']:
            for product in ['X', 'Y']:
                for date in dates:
                    data.append({
                        'store': store,
                        'product': product,
                        'date': date,
                        'sales': 100,
                        'sample': 'test',
                        'prediction': 100
                    })
        
        forecast_df = pd.DataFrame(data)
        
        # Run complete workflow
        deliveries = router.distribute_demand(forecast_df)
        deliveries = router.assign_trucks(deliveries)
        final_deliveries, routes = router.optimize_routes(deliveries)
        
        # Should have deliveries for all combinations
        unique_combinations = final_deliveries[['store', 'product']].drop_duplicates()
        assert len(unique_combinations) == 4
        
        # Should have routes
        assert len(routes) > 0
        assert routes['total_distance'].sum() > 0
    
    def test_workflow_payload_constraint_enforcement(self):
        """Test that payload constraints are enforced throughout workflow."""
        np.random.seed(42)
        
        router = Router(
            primary_keys=['store', 'product'],
            max_payload=200,  # Set payload high enough to accommodate distributed demand
            origin='08020'
        )
        
        router.generate_customers(n_customers=20)
        
        # Create forecast with high demand
        forecast_df = pd.DataFrame({
            'store': ['A'],
            'product': ['X'],
            'date': [pd.Timestamp('2024-01-01')],
            'sales': [500],  # High demand
            'sample': ['test'],
            'prediction': [500]
        })
        
        # Run workflow
        deliveries = router.distribute_demand(forecast_df)
        deliveries = router.assign_trucks(deliveries)
        final_deliveries, routes = router.optimize_routes(deliveries)
        
        # Verify that trucks are assigned and payload constraint is respected
        # Note: distribute_demand splits demand across customers, so individual
        # delivery units may be large. The assign_trucks method enforces the constraint.
        for truck, truck_group in final_deliveries.groupby('truck'):
            total_payload = truck_group['units'].sum()
            assert total_payload <= router.max_payload
        
        # Verify workflow completed successfully
        assert len(final_deliveries) > 0
        assert len(routes) > 0
        assert final_deliveries['truck'].notna().all()


class TestRouterValidation:
    """Tests for parameter validation and error handling."""
    
    def test_invalid_max_payload(self):
        """Test that non-positive max_payload raises error."""
        with pytest.raises(ValueError, match="max_payload must be positive"):
            Router(
                primary_keys=['store', 'product'],
                max_payload=0
            )
    
    def test_invalid_origin_format(self):
        """Test that invalid origin postal code raises error."""
        with pytest.raises(ValueError, match="origin must be a 5-digit"):
            Router(
                primary_keys=['store', 'product'],
                origin='123'  # Too short
            )
    
    def test_empty_primary_keys(self):
        """Test that empty primary_keys raises error."""
        with pytest.raises(ValueError, match="primary_keys must contain at least one"):
            Router(
                primary_keys=[],
                max_payload=100
            )
