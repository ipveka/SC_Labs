"""
Forecaster module for demand forecasting using GluonTS.

This module provides time series forecasting capabilities for supply chain
demand planning using the GluonTS library with SimpleFeedForwardEstimator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Optional
import pandas as pd
import numpy as np
import warnings
import logging
import os

from gluonts.dataset.pandas import PandasDataset
from gluonts.torch import SimpleFeedForwardEstimator, DeepAREstimator, TemporalFusionTransformerEstimator

from utils.logger import Logger, suppress_warnings
from utils.config import get_config

# Suppress warnings and verbose logging
suppress_warnings()
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


class Forecaster:
    """
    A forecasting class for time series demand prediction.
    
    This class uses GluonTS SimpleFeedForwardEstimator to train models on
    historical time series data and generate forecasts for future periods.
    It supports multiple time series grouped by primary keys (e.g., store-product
    combinations).
    
    Attributes:
        primary_keys (List[str]): Column names used to group time series
            (e.g., ['store', 'product'])
        date_col (str): Name of the date/time column
        target_col (str): Name of the target variable to forecast
        frequency (str): Pandas frequency string ('W' for weekly, 'D' for daily)
        forecast_horizon (int): Number of periods to forecast into the future
        model: Trained GluonTS predictor (None until fit() is called)
    
    Example:
        >>> forecaster = Forecaster(
        ...     primary_keys=['store', 'product'],
        ...     date_col='date',
        ...     target_col='sales',
        ...     frequency='W',
        ...     forecast_horizon=4
        ... )
        >>> forecaster.fit(train_data)
        >>> forecasts = forecaster.predict(test_data)
    """
    
    def __init__(
        self,
        primary_keys: List[str],
        date_col: str = 'date',
        target_col: str = 'sales',
        frequency: str = 'W',
        forecast_horizon: int = 4,
        model_type: str = None,
        verbose: bool = None
    ) -> None:
        """
        Initialize the Forecaster with configuration parameters.
        
        Args:
            primary_keys: List of column names that uniquely identify each
                time series (e.g., ['store', 'product'])
            date_col: Name of the column containing dates/timestamps
            target_col: Name of the column containing values to forecast
            frequency: Pandas frequency string for the time series
                ('W' = weekly, 'D' = daily, 'M' = monthly, etc.)
            forecast_horizon: Number of future periods to forecast
            model_type: Type of model ('simple_feedforward', 'deepar', 'transformer')
                If None, uses config value
            verbose: Show detailed logs (None = use config)
        """
        self.primary_keys = primary_keys
        self.date_col = date_col
        self.target_col = target_col
        self.frequency = frequency
        self.forecast_horizon = forecast_horizon
        self.model: Optional[object] = None
        
        # Load config
        self.config = get_config()
        self.model_type = model_type if model_type is not None else self.config.get('forecasting', 'model', default='simple_feedforward')
        self.verbose = verbose if verbose is not None else self.config.get('forecasting', 'verbose', default=False)
        
        # Validate model type
        valid_models = ['simple_feedforward', 'deepar', 'transformer']
        if self.model_type not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}, got '{self.model_type}'")
        
        # Initialize logger
        self.logger = Logger('Forecaster', use_colors=self.config.get('logging', 'use_colors', default=True))

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and clean time series data for forecasting.
        
        This method aggregates data by primary keys and date, ensures a continuous
        date range, and fills missing values to create clean time series suitable
        for model training.
        
        Args:
            df: Input DataFrame containing time series data with columns
                matching primary_keys, date_col, and target_col
        
        Returns:
            Cleaned DataFrame with continuous date range and no missing values
        
        Raises:
            ValueError: If DataFrame is empty or missing required columns
        
        Example:
            >>> df = pd.DataFrame({
            ...     'store': ['A', 'A', 'A'],
            ...     'product': ['X', 'X', 'X'],
            ...     'date': pd.date_range('2024-01-01', periods=3, freq='W'),
            ...     'sales': [100, 150, 120]
            ... })
            >>> clean_df = forecaster.prepare_data(df)
        """
        # Handle edge case: empty DataFrame
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Validate required columns exist
        required_cols = self.primary_keys + [self.date_col, self.target_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Handle edge case: single row
        if len(df) == 1:
            return df.copy()
        
        # Ensure date column is datetime type
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Aggregate data by primary keys and date (in case of duplicates)
        group_cols = self.primary_keys + [self.date_col]
        df_agg = df.groupby(group_cols, as_index=False)[self.target_col].sum()
        
        # Create continuous date range for each primary key combination
        result_dfs = []
        
        for keys, group in df_agg.groupby(self.primary_keys):
            # Get min and max dates for this group
            min_date = group[self.date_col].min()
            max_date = group[self.date_col].max()
            
            # Create continuous date range
            date_range = pd.date_range(
                start=min_date,
                end=max_date,
                freq=self.frequency
            )
            
            # Create DataFrame with continuous dates
            continuous_df = pd.DataFrame({self.date_col: date_range})
            
            # Add primary key values
            if isinstance(keys, tuple):
                for i, key_col in enumerate(self.primary_keys):
                    continuous_df[key_col] = keys[i]
            else:
                continuous_df[self.primary_keys[0]] = keys
            
            # Merge with actual data
            merged = continuous_df.merge(
                group,
                on=self.primary_keys + [self.date_col],
                how='left'
            )
            
            # Fill missing values with forward fill
            merged[self.target_col] = merged[self.target_col].ffill()
            
            # If still NaN at the beginning, use backward fill
            merged[self.target_col] = merged[self.target_col].bfill()
            
            # If still NaN (all values were NaN), fill with 0
            merged[self.target_col] = merged[self.target_col].fillna(0)
            
            result_dfs.append(merged)
        
        # Combine all groups
        result = pd.concat(result_dfs, ignore_index=True)
        
        # Sort by primary keys and date
        result = result.sort_values(self.primary_keys + [self.date_col])
        
        return result.reset_index(drop=True)

    def fit(self, df: pd.DataFrame) -> None:
        """
        Train the forecasting model on historical time series data.
        
        This method converts the DataFrame to GluonTS format, groups data by
        primary keys to create multiple time series, and trains a
        SimpleFeedForwardEstimator model.
        
        Args:
            df: Training DataFrame with columns matching primary_keys,
                date_col, and target_col
        
        Raises:
            ValueError: If DataFrame has insufficient data for training
            RuntimeError: If model training fails
        
        Example:
            >>> train_df = pd.DataFrame({
            ...     'store': ['A'] * 20,
            ...     'product': ['X'] * 20,
            ...     'date': pd.date_range('2024-01-01', periods=20, freq='W'),
            ...     'sales': np.random.randint(50, 200, 20)
            ... })
            >>> forecaster.fit(train_df)
        """
        # Validate input
        if df.empty:
            raise ValueError("Training DataFrame is empty")
        
        # Log start
        if self.verbose:
            self.logger.info(f"Training forecaster on {len(df)} records")
        
        # Warn if insufficient history
        min_periods = self.forecast_horizon * 2
        for keys, group in df.groupby(self.primary_keys):
            if len(group) < min_periods:
                self.logger.warning(f"Time series {keys} has only {len(group)} periods (recommended: {min_periods})")
        
        # Prepare data
        df_clean = self.prepare_data(df)
        
        # Ensure date column is datetime
        df_clean[self.date_col] = pd.to_datetime(df_clean[self.date_col])
        
        try:
            # Convert to GluonTS PandasDataset format
            # GluonTS expects: item_id (for grouping), timestamp, target
            gluon_df = df_clean.copy()
            
            # Create item_id by combining primary keys
            if len(self.primary_keys) == 1:
                gluon_df['item_id'] = gluon_df[self.primary_keys[0]].astype(str)
            else:
                gluon_df['item_id'] = gluon_df[self.primary_keys].apply(
                    lambda row: '_'.join(row.astype(str)), axis=1
                )
            
            # Rename columns to GluonTS expected names
            gluon_df = gluon_df.rename(columns={
                self.date_col: 'timestamp',
                self.target_col: 'target'
            })
            
            # Ensure target is float32 type for GluonTS (PyTorch compatibility)
            gluon_df['target'] = gluon_df['target'].astype('float32')
            
            # Create PandasDataset
            dataset = PandasDataset.from_long_dataframe(
                gluon_df[['item_id', 'timestamp', 'target']],
                item_id='item_id',
                timestamp='timestamp',
                target='target',
                freq=self.frequency
            )
            
            # Get model config
            epochs = self.config.get('forecasting', 'epochs', default=10)
            lr = self.config.get('forecasting', 'learning_rate', default=0.001)
            
            # Initialize estimator based on model type
            trainer_kwargs = {
                "max_epochs": epochs,
                "enable_progress_bar": self.config.get('forecasting', 'show_progress', default=False),
                "enable_model_summary": False
            }
            
            if self.model_type == 'simple_feedforward':
                estimator = SimpleFeedForwardEstimator(
                    prediction_length=self.forecast_horizon,
                    lr=lr,
                    trainer_kwargs=trainer_kwargs
                )
            elif self.model_type == 'deepar':
                estimator = DeepAREstimator(
                    prediction_length=self.forecast_horizon,
                    lr=lr,
                    trainer_kwargs=trainer_kwargs
                )
            elif self.model_type == 'transformer':
                estimator = TemporalFusionTransformerEstimator(
                    prediction_length=self.forecast_horizon,
                    lr=lr,
                    trainer_kwargs=trainer_kwargs
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Suppress output during training
            if not self.verbose:
                import sys
                import io
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
            
            try:
                # Train the model
                self.model = estimator.train(dataset)
                
                if self.verbose:
                    self.logger.success("Model trained successfully")
            finally:
                # Restore output
                if not self.verbose:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
            
        except Exception as e:
            raise RuntimeError(f"Model training failed: {str(e)}") from e

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forecasts for future periods using the trained model.
        
        This method creates predictions for each primary key combination and
        returns a unified DataFrame containing both historical data (with
        sample='train' and prediction=NaN) and forecast data (with sample='test'
        and prediction values).
        
        Args:
            df: DataFrame containing historical data (same format as training data)
        
        Returns:
            DataFrame with columns: primary_keys, date_col, target_col,
            'sample' ('train' or 'test'), and 'prediction' (NaN for train,
            forecasted values for test)
        
        Raises:
            RuntimeError: If model has not been trained (fit() not called)
            ValueError: If input DataFrame is invalid
        
        Example:
            >>> forecasts = forecaster.predict(historical_data)
            >>> # forecasts contains both historical and predicted values
            >>> train_data = forecasts[forecasts['sample'] == 'train']
            >>> test_data = forecasts[forecasts['sample'] == 'test']
        """
        # Check if model is trained
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        
        # Validate input
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Prepare data
        df_clean = self.prepare_data(df)
        df_clean[self.date_col] = pd.to_datetime(df_clean[self.date_col])
        
        # Create historical data with sample='train' and prediction=NaN
        historical = df_clean.copy()
        historical['sample'] = 'train'
        historical['prediction'] = np.nan
        
        # Generate forecasts for each primary key combination
        forecast_dfs = []
        
        for keys, group in df_clean.groupby(self.primary_keys):
            try:
                # Prepare data for GluonTS prediction
                gluon_df = group.copy()
                
                # Create item_id
                if len(self.primary_keys) == 1:
                    item_id = str(keys)
                else:
                    item_id = '_'.join(str(k) for k in keys)
                
                gluon_df['item_id'] = item_id
                gluon_df = gluon_df.rename(columns={
                    self.date_col: 'timestamp',
                    self.target_col: 'target'
                })
                
                # Ensure target is float32 type for GluonTS (PyTorch compatibility)
                gluon_df['target'] = gluon_df['target'].astype('float32')
                
                # Create dataset for this time series
                dataset = PandasDataset.from_long_dataframe(
                    gluon_df[['item_id', 'timestamp', 'target']],
                    item_id='item_id',
                    timestamp='timestamp',
                    target='target',
                    freq=self.frequency
                )
                
                # Generate forecast
                forecast_iter = self.model.predict(dataset)
                forecast = next(forecast_iter)
                
                # Extract forecast values (mean prediction)
                forecast_values = forecast.mean
                
                # Create future dates
                last_date = group[self.date_col].max()
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(1, unit=self.frequency),
                    periods=self.forecast_horizon,
                    freq=self.frequency
                )
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    self.date_col: future_dates,
                    self.target_col: np.nan,  # No actual values for future
                    'sample': 'test',
                    'prediction': forecast_values
                })
                
                # Add primary key values
                if isinstance(keys, tuple):
                    for i, key_col in enumerate(self.primary_keys):
                        forecast_df[key_col] = keys[i]
                else:
                    forecast_df[self.primary_keys[0]] = keys
                
                forecast_dfs.append(forecast_df)
                
            except Exception as e:
                print(f"Warning: Failed to generate forecast for {keys}: {str(e)}")
                continue
        
        # Combine historical and forecast data
        if forecast_dfs:
            forecasts = pd.concat(forecast_dfs, ignore_index=True)
            result = pd.concat([historical, forecasts], ignore_index=True)
        else:
            # If no forecasts generated, return only historical
            result = historical
        
        # Reorder columns
        column_order = self.primary_keys + [self.date_col, self.target_col, 'sample', 'prediction']
        result = result[column_order]
        
        # Sort by primary keys and date
        result = result.sort_values(self.primary_keys + [self.date_col])
        
        return result.reset_index(drop=True)
