"""
Unit tests for the Forecaster module.

Tests cover:
- prepare_data() with missing dates
- fit() with minimal valid dataset
- predict() output schema and dimensions
- handling of multiple time series
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from forecaster.forecaster import Forecaster


class TestForecasterPrepareData:
    """Tests for the prepare_data() method."""
    
    def test_prepare_data_fills_missing_dates(self):
        """Test that prepare_data() fills in missing dates in the time series."""
        # Create data with missing dates (use weekly-Sunday aligned dates)
        dates = pd.date_range('2024-01-07', periods=3, freq='2W')  # Jan 7, 21, Feb 4
        df = pd.DataFrame({
            'store': ['A', 'A', 'A'],
            'product': ['X', 'X', 'X'],
            'date': dates,
            'sales': [100, 150, 120]
        })
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W'
        )
        
        result = forecaster.prepare_data(df)
        
        # Should have 5 weeks of data (Jan 7, 14, 21, 28, Feb 4)
        assert len(result) == 5
        
        # Check that dates are continuous
        expected_dates = pd.date_range('2024-01-07', '2024-02-04', freq='W')
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(result['date']),
            expected_dates,
            check_names=False
        )
        
        # Check that missing values were filled
        assert result['sales'].notna().all()
        
        # Check forward fill behavior (week 2 should have value from week 1)
        assert result.loc[1, 'sales'] == 100  # Forward filled from first value
    
    def test_prepare_data_handles_multiple_time_series(self):
        """Test that prepare_data() correctly handles multiple time series."""
        # Create data for 2 stores and 2 products (4 time series)
        dates = pd.date_range('2024-01-01', periods=3, freq='W')
        df = pd.DataFrame({
            'store': ['A', 'A', 'A', 'B', 'B', 'B'],
            'product': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'date': list(dates) + list(dates),
            'sales': [100, 150, 120, 200, 250, 220]
        })
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W'
        )
        
        result = forecaster.prepare_data(df)
        
        # Should have 6 rows (3 dates × 2 time series)
        assert len(result) == 6
        
        # Check that both time series are present
        groups = result.groupby(['store', 'product'])
        assert len(groups) == 2
        
        # Check each group has correct number of periods
        for _, group in groups:
            assert len(group) == 3
    
    def test_prepare_data_aggregates_duplicates(self):
        """Test that prepare_data() aggregates duplicate date entries."""
        # Use Sunday-aligned dates for weekly frequency
        dates = pd.to_datetime(['2024-01-07', '2024-01-07', '2024-01-14'])
        df = pd.DataFrame({
            'store': ['A', 'A', 'A'],
            'product': ['X', 'X', 'X'],
            'date': dates,
            'sales': [100, 50, 150]
        })
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W'
        )
        
        result = forecaster.prepare_data(df)
        
        # Should have 2 rows (duplicates aggregated)
        assert len(result) == 2
        
        # First date should have sum of duplicates
        assert result.loc[0, 'sales'] == 150  # 100 + 50
    
    def test_prepare_data_empty_dataframe_raises_error(self):
        """Test that prepare_data() raises error for empty DataFrame."""
        df = pd.DataFrame(columns=['store', 'product', 'date', 'sales'])
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W'
        )
        
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            forecaster.prepare_data(df)
    
    def test_prepare_data_missing_columns_raises_error(self):
        """Test that prepare_data() raises error for missing required columns."""
        df = pd.DataFrame({
            'store': ['A'],
            'product': ['X'],
            'date': [datetime(2024, 1, 1)]
            # Missing 'sales' column
        })
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W'
        )
        
        with pytest.raises(ValueError, match="Missing required columns"):
            forecaster.prepare_data(df)
    
    def test_prepare_data_single_row(self):
        """Test that prepare_data() handles single row DataFrame."""
        df = pd.DataFrame({
            'store': ['A'],
            'product': ['X'],
            'date': [datetime(2024, 1, 1)],
            'sales': [100]
        })
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W'
        )
        
        result = forecaster.prepare_data(df)
        
        # Should return the single row unchanged
        assert len(result) == 1
        assert result.loc[0, 'sales'] == 100


class TestForecasterFit:
    """Tests for the fit() method."""
    
    def test_fit_with_minimal_valid_dataset(self):
        """Test that fit() works with minimal valid dataset."""
        # Create minimal dataset (10 periods for 1 time series)
        dates = pd.date_range('2024-01-01', periods=10, freq='W')
        df = pd.DataFrame({
            'store': ['A'] * 10,
            'product': ['X'] * 10,
            'date': dates,
            'sales': np.random.randint(50, 200, 10)
        })
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W',
            forecast_horizon=4
        )
        
        # Should not raise an error
        forecaster.fit(df)
        
        # Model should be trained
        assert forecaster.model is not None
    
    def test_fit_with_multiple_time_series(self):
        """Test that fit() handles multiple time series correctly."""
        # Create data for 2 stores × 2 products = 4 time series
        dates = pd.date_range('2024-01-01', periods=20, freq='W')
        
        data = []
        for store in ['A', 'B']:
            for product in ['X', 'Y']:
                for date in dates:
                    data.append({
                        'store': store,
                        'product': product,
                        'date': date,
                        'sales': np.random.randint(50, 200)
                    })
        
        df = pd.DataFrame(data)
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W',
            forecast_horizon=4
        )
        
        # Should not raise an error
        forecaster.fit(df)
        
        # Model should be trained
        assert forecaster.model is not None
    
    def test_fit_empty_dataframe_raises_error(self):
        """Test that fit() raises error for empty DataFrame."""
        df = pd.DataFrame(columns=['store', 'product', 'date', 'sales'])
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W'
        )
        
        with pytest.raises(ValueError, match="Training DataFrame is empty"):
            forecaster.fit(df)
    
    def test_fit_warns_insufficient_history(self, capsys):
        """Test that fit() warns when there's insufficient historical data."""
        # Create dataset with only 5 periods (less than 2x forecast_horizon)
        dates = pd.date_range('2024-01-01', periods=5, freq='W')
        df = pd.DataFrame({
            'store': ['A'] * 5,
            'product': ['X'] * 5,
            'date': dates,
            'sales': [100, 110, 120, 130, 140]
        })
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W',
            forecast_horizon=4
        )
        
        forecaster.fit(df)
        
        # Check that warning was printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Recommended minimum" in captured.out


class TestForecasterPredict:
    """Tests for the predict() method."""
    
    def test_predict_output_schema(self):
        """Test that predict() returns DataFrame with correct schema."""
        # Create and train model
        dates = pd.date_range('2024-01-01', periods=20, freq='W')
        df = pd.DataFrame({
            'store': ['A'] * 20,
            'product': ['X'] * 20,
            'date': dates,
            'sales': np.random.randint(50, 200, 20)
        })
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W',
            forecast_horizon=4
        )
        
        forecaster.fit(df)
        result = forecaster.predict(df)
        
        # Check columns
        expected_columns = ['store', 'product', 'date', 'sales', 'sample', 'prediction']
        assert list(result.columns) == expected_columns
        
        # Check data types
        assert result['sample'].dtype == object
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
        assert pd.api.types.is_numeric_dtype(result['sales'])
        assert pd.api.types.is_numeric_dtype(result['prediction'])
    
    def test_predict_output_dimensions(self):
        """Test that predict() returns correct number of rows."""
        # Create and train model
        dates = pd.date_range('2024-01-01', periods=20, freq='W')
        df = pd.DataFrame({
            'store': ['A'] * 20,
            'product': ['X'] * 20,
            'date': dates,
            'sales': np.random.randint(50, 200, 20)
        })
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W',
            forecast_horizon=4
        )
        
        forecaster.fit(df)
        result = forecaster.predict(df)
        
        # Should have historical (20) + forecast (4) = 24 rows
        assert len(result) == 24
        
        # Check train/test split
        train_data = result[result['sample'] == 'train']
        test_data = result[result['sample'] == 'test']
        
        assert len(train_data) == 20
        assert len(test_data) == 4
    
    def test_predict_train_test_split(self):
        """Test that predict() correctly splits train and test data."""
        # Create and train model
        dates = pd.date_range('2024-01-01', periods=20, freq='W')
        df = pd.DataFrame({
            'store': ['A'] * 20,
            'product': ['X'] * 20,
            'date': dates,
            'sales': np.random.randint(50, 200, 20)
        })
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W',
            forecast_horizon=4
        )
        
        forecaster.fit(df)
        result = forecaster.predict(df)
        
        train_data = result[result['sample'] == 'train']
        test_data = result[result['sample'] == 'test']
        
        # Train data should have NaN predictions
        assert train_data['prediction'].isna().all()
        
        # Test data should have non-NaN predictions
        assert test_data['prediction'].notna().all()
        
        # Test data should have NaN sales (future values unknown)
        assert test_data['sales'].isna().all()
        
        # Train data should have non-NaN sales
        assert train_data['sales'].notna().all()
    
    def test_predict_multiple_time_series(self):
        """Test that predict() handles multiple time series correctly."""
        # Create data for 2 stores × 2 products = 4 time series
        dates = pd.date_range('2024-01-01', periods=20, freq='W')
        
        data = []
        for store in ['A', 'B']:
            for product in ['X', 'Y']:
                for date in dates:
                    data.append({
                        'store': store,
                        'product': product,
                        'date': date,
                        'sales': np.random.randint(50, 200)
                    })
        
        df = pd.DataFrame(data)
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W',
            forecast_horizon=4
        )
        
        forecaster.fit(df)
        result = forecaster.predict(df)
        
        # Should have (20 historical + 4 forecast) × 4 time series = 96 rows
        assert len(result) == 96
        
        # Check each time series has correct dimensions
        for (store, product), group in result.groupby(['store', 'product']):
            assert len(group) == 24  # 20 train + 4 test
            
            train = group[group['sample'] == 'train']
            test = group[group['sample'] == 'test']
            
            assert len(train) == 20
            assert len(test) == 4
    
    def test_predict_without_fit_raises_error(self):
        """Test that predict() raises error if model not trained."""
        df = pd.DataFrame({
            'store': ['A'] * 10,
            'product': ['X'] * 10,
            'date': pd.date_range('2024-01-01', periods=10, freq='W'),
            'sales': np.random.randint(50, 200, 10)
        })
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W'
        )
        
        # Don't call fit()
        with pytest.raises(RuntimeError, match="Model has not been trained"):
            forecaster.predict(df)
    
    def test_predict_empty_dataframe_raises_error(self):
        """Test that predict() raises error for empty DataFrame."""
        # Train model first
        dates = pd.date_range('2024-01-01', periods=20, freq='W')
        df = pd.DataFrame({
            'store': ['A'] * 20,
            'product': ['X'] * 20,
            'date': dates,
            'sales': np.random.randint(50, 200, 20)
        })
        
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W'
        )
        
        forecaster.fit(df)
        
        # Try to predict with empty DataFrame
        empty_df = pd.DataFrame(columns=['store', 'product', 'date', 'sales'])
        
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            forecaster.predict(empty_df)


class TestForecasterIntegration:
    """Integration tests for complete forecasting workflow."""
    
    def test_complete_workflow_single_time_series(self):
        """Test complete workflow with single time series."""
        # Generate synthetic data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=30, freq='W')
        sales = 100 + np.arange(30) * 2 + np.random.normal(0, 10, 30)
        
        df = pd.DataFrame({
            'store': ['A'] * 30,
            'product': ['X'] * 30,
            'date': dates,
            'sales': sales
        })
        
        # Initialize forecaster
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W',
            forecast_horizon=4
        )
        
        # Prepare data
        prepared = forecaster.prepare_data(df)
        assert len(prepared) == 30
        
        # Fit model
        forecaster.fit(prepared)
        assert forecaster.model is not None
        
        # Generate predictions
        predictions = forecaster.predict(prepared)
        
        # Validate output
        assert len(predictions) == 34
        assert predictions['sample'].isin(['train', 'test']).all()
        assert predictions[predictions['sample'] == 'train']['prediction'].isna().all()
        assert predictions[predictions['sample'] == 'test']['prediction'].notna().all()
    
    def test_complete_workflow_multiple_time_series(self):
        """Test complete workflow with multiple time series."""
        # Generate synthetic data for multiple time series
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=30, freq='W')
        
        data = []
        for store in ['A', 'B']:
            for product in ['X', 'Y']:
                base = np.random.randint(80, 120)
                sales = base + np.arange(30) * 2 + np.random.normal(0, 10, 30)
                
                for i, date in enumerate(dates):
                    data.append({
                        'store': store,
                        'product': product,
                        'date': date,
                        'sales': sales[i]
                    })
        
        df = pd.DataFrame(data)
        
        # Initialize forecaster
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W',
            forecast_horizon=4
        )
        
        # Prepare data
        prepared = forecaster.prepare_data(df)
        assert len(prepared) == 120 
        
        # Fit model
        forecaster.fit(prepared)
        assert forecaster.model is not None
        
        # Generate predictions
        predictions = forecaster.predict(prepared)
        
        # Validate output
        assert len(predictions) == 136 
        
        # Check each time series
        for (store, product), group in predictions.groupby(['store', 'product']):
            assert len(group) == 34
            assert len(group[group['sample'] == 'train']) == 30
            assert len(group[group['sample'] == 'test']) == 4
