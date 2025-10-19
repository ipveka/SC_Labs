"""
Unit tests for the auxiliar module data generation functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from auxiliar.auxiliar import generate_data


class TestGenerateData:
    """Test suite for the generate_data function."""
    
    def test_output_schema_matches_specification(self):
        """Test that output DataFrame has the correct columns."""
        df = generate_data(n_stores=2, n_products=2, n_weeks=10)
        
        expected_columns = ['store', 'product', 'date', 'sales', 'inventory', 'customer_id', 'destination']
        assert list(df.columns) == expected_columns, f"Expected columns {expected_columns}, got {list(df.columns)}"
        
        # Verify column data types
        assert df['store'].dtype == object, "store column should be object (string)"
        assert df['product'].dtype == object, "product column should be object (string)"
        assert pd.api.types.is_datetime64_any_dtype(df['date']), "date column should be datetime64"
        assert pd.api.types.is_integer_dtype(df['sales']), "sales column should be integer"
        assert pd.api.types.is_integer_dtype(df['inventory']), "inventory column should be integer"
        assert df['customer_id'].dtype == object, "customer_id column should be object (string)"
        assert df['destination'].dtype == object, "destination column should be object (string)"
    
    def test_data_dimensions(self):
        """Test that output has correct number of rows (n_stores × n_products × n_weeks)."""
        # Test case 1: Small dataset
        n_stores, n_products, n_weeks = 2, 2, 10
        df = generate_data(n_stores=n_stores, n_products=n_products, n_weeks=n_weeks)
        expected_rows = n_stores * n_products * n_weeks
        assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
        
        # Test case 2: Different dimensions
        n_stores, n_products, n_weeks = 3, 4, 20
        df = generate_data(n_stores=n_stores, n_products=n_products, n_weeks=n_weeks)
        expected_rows = n_stores * n_products * n_weeks
        assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
        
        # Test case 3: Single store, single product
        n_stores, n_products, n_weeks = 1, 1, 5
        df = generate_data(n_stores=n_stores, n_products=n_products, n_weeks=n_weeks)
        expected_rows = n_stores * n_products * n_weeks
        assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    
    def test_reproducibility_with_same_seed(self):
        """Test that using the same seed produces identical results."""
        seed = 123
        
        # Generate data twice with the same seed
        df1 = generate_data(n_stores=3, n_products=2, n_weeks=20, seed=seed)
        df2 = generate_data(n_stores=3, n_products=2, n_weeks=20, seed=seed)
        
        # DataFrames should be identical
        pd.testing.assert_frame_equal(df1, df2, check_exact=True)
        
        # Verify specific numeric columns are identical
        assert np.array_equal(df1['sales'].values, df2['sales'].values), "Sales values should be identical with same seed"
        assert np.array_equal(df1['inventory'].values, df2['inventory'].values), "Inventory values should be identical with same seed"
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        df1 = generate_data(n_stores=2, n_products=2, n_weeks=10, seed=42)
        df2 = generate_data(n_stores=2, n_products=2, n_weeks=10, seed=99)
        
        # DataFrames should have same structure but different values
        assert list(df1.columns) == list(df2.columns), "Columns should be the same"
        assert len(df1) == len(df2), "Row counts should be the same"
        
        # Sales values should be different
        assert not np.array_equal(df1['sales'].values, df2['sales'].values), "Sales values should differ with different seeds"
    
    def test_non_negative_sales_values(self):
        """Test that all sales values are non-negative."""
        # Test with multiple different seeds to ensure robustness
        for seed in [42, 100, 200, 300]:
            df = generate_data(n_stores=3, n_products=3, n_weeks=52, seed=seed)
            
            assert (df['sales'] >= 0).all(), f"All sales values should be non-negative (seed={seed})"
            assert not df['sales'].isna().any(), f"Sales should not contain NaN values (seed={seed})"
    
    def test_date_range_continuity(self):
        """Test that dates are continuous with weekly frequency."""
        df = generate_data(n_stores=2, n_products=1, n_weeks=10, start_date='2024-01-01')
        
        # Get unique dates for one store-product combination
        dates = df[(df['store'] == 'A') & (df['product'] == 'A')]['date'].values
        
        # Check that we have the expected number of dates
        assert len(dates) == 10, "Should have 10 unique dates"
        
        # Check that dates are weekly intervals (7 days apart)
        for i in range(1, len(dates)):
            diff = pd.Timestamp(dates[i]) - pd.Timestamp(dates[i-1])
            assert diff.days == 7, f"Dates should be 7 days apart, got {diff.days}"
    
    def test_store_product_combinations(self):
        """Test that all store-product combinations are present."""
        n_stores, n_products = 3, 2
        df = generate_data(n_stores=n_stores, n_products=n_products, n_weeks=5)
        
        # Get unique combinations
        combinations = df[['store', 'product']].drop_duplicates()
        
        # Should have n_stores × n_products unique combinations
        expected_combinations = n_stores * n_products
        assert len(combinations) == expected_combinations, f"Expected {expected_combinations} combinations, got {len(combinations)}"
        
        # Verify store identifiers
        unique_stores = df['store'].unique()
        assert len(unique_stores) == n_stores, f"Expected {n_stores} unique stores"
        assert set(unique_stores) == {'A', 'B', 'C'}, "Store identifiers should be A, B, C"
        
        # Verify product identifiers
        unique_products = df['product'].unique()
        assert len(unique_products) == n_products, f"Expected {n_products} unique products"
        assert set(unique_products) == {'A', 'B'}, "Product identifiers should be A, B"
    
    def test_inventory_values_in_expected_range(self):
        """Test that inventory values are within the expected range (200-500)."""
        df = generate_data(n_stores=2, n_products=2, n_weeks=20, seed=42)
        
        # Inventory should be between 200 and 500 (with some tolerance for randomness)
        assert df['inventory'].min() >= 200, "Minimum inventory should be >= 200"
        assert df['inventory'].max() <= 500, "Maximum inventory should be <= 500"
    
    def test_customer_id_format(self):
        """Test that customer IDs follow the expected format."""
        df = generate_data(n_stores=1, n_products=1, n_weeks=5, seed=42)
        
        for customer_id in df['customer_id']:
            assert customer_id.startswith('CUST_'), f"Customer ID should start with 'CUST_', got {customer_id}"
            assert len(customer_id) == 9, f"Customer ID should be 9 characters long, got {len(customer_id)}"
            # Extract numeric part and verify it's a valid number
            numeric_part = customer_id.split('_')[1]
            assert numeric_part.isdigit(), f"Customer ID numeric part should be digits, got {numeric_part}"
            assert 1000 <= int(numeric_part) <= 9999, f"Customer ID number should be between 1000-9999, got {numeric_part}"
    
    def test_postal_code_format(self):
        """Test that postal codes follow the expected format (5-digit strings)."""
        df = generate_data(n_stores=1, n_products=1, n_weeks=5, seed=42)
        
        for postal_code in df['destination']:
            assert len(postal_code) == 5, f"Postal code should be 5 characters long, got {len(postal_code)}"
            assert postal_code.isdigit(), f"Postal code should be all digits, got {postal_code}"
            assert 10000 <= int(postal_code) <= 99999, f"Postal code should be between 10000-99999, got {postal_code}"
    
    def test_start_date_parameter(self):
        """Test that the start_date parameter is respected."""
        start_date = '2023-06-15'
        df = generate_data(n_stores=1, n_products=1, n_weeks=3, start_date=start_date)
        
        # Get the first date
        first_date = df['date'].min()
        expected_first_date = pd.Timestamp(start_date)
        
        # The first date should be on or after the start date (accounting for weekly frequency)
        assert first_date >= expected_first_date, f"First date should be >= {expected_first_date}, got {first_date}"
    
    def test_no_missing_values(self):
        """Test that the generated data has no missing values."""
        df = generate_data(n_stores=3, n_products=2, n_weeks=20, seed=42)
        
        # Check for any NaN values in the DataFrame
        assert not df.isna().any().any(), "Generated data should not contain any missing values"
    
    def test_sales_has_realistic_patterns(self):
        """Test that sales data shows expected patterns (trend and variability)."""
        df = generate_data(n_stores=1, n_products=1, n_weeks=52, seed=42)
        
        sales = df['sales'].values
        
        # Sales should have some variability (not constant)
        assert sales.std() > 0, "Sales should have variability"
        
        # Sales should have reasonable range (not all the same value)
        assert sales.max() - sales.min() > 10, "Sales should have reasonable range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
