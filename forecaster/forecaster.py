"""
Forecaster module for demand forecasting using LightGBM.

This module provides time series forecasting capabilities for supply chain
demand planning using LightGBM with automated feature engineering.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import warnings

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("LightGBM is required. Install with: pip install lightgbm")

from utils.logger import Logger, suppress_warnings
from utils.config import get_config

# Suppress warnings
suppress_warnings()


class Forecaster:
    """
    A forecasting class for time series demand prediction using LightGBM.
    
    This class uses LightGBM with automated feature engineering to train models on
    historical time series data and generate forecasts for future periods.
    It supports multiple time series grouped by primary keys (e.g., store-product
    combinations).
    
    Features generated:
    - Temporal: day_of_week, week_of_year, month, quarter, year
    - Lag features: previous 1, 2, 3, 4 periods
    - Rolling statistics: mean, std, min, max over windows [3, 4, 8]
    - Categorical encodings: all primary keys
    
    Attributes:
        primary_keys (List[str]): Column names used to group time series
            (e.g., ['store', 'product'])
        date_col (str): Name of the date/time column
        target_col (str): Name of the target variable to forecast
        frequency (str): Pandas frequency string ('W' for weekly, 'D' for daily)
        forecast_horizon (int): Number of periods to forecast into the future
        model: Trained LightGBM model (None until fit() is called)
    
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
        model_type: str = None,  # Kept for compatibility, not used
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
            model_type: Kept for compatibility (not used with LightGBM)
            verbose: Show detailed logs (None = use config)
        """
        self.primary_keys = primary_keys
        self.date_col = date_col
        self.target_col = target_col
        self.frequency = frequency
        self.forecast_horizon = forecast_horizon
        self.model: Optional[lgb.Booster] = None
        self.feature_cols: List[str] = []
        self.categorical_features: List[str] = []
        
        # Load config
        self.config = get_config()
        self.verbose = verbose if verbose is not None else self.config.get('forecasting', 'verbose', default=False)
        
        # Initialize logger
        self.logger = Logger('Forecaster', use_colors=self.config.get('logging', 'use_colors', default=True))

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from date column.
        
        Args:
            df: DataFrame with date column
            
        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Temporal features
        df['day_of_week'] = df[self.date_col].dt.dayofweek
        df['week_of_year'] = df[self.date_col].dt.isocalendar().week.astype(int)
        df['month'] = df[self.date_col].dt.month
        df['quarter'] = df[self.date_col].dt.quarter
        df['year'] = df[self.date_col].dt.year
        df['day_of_month'] = df[self.date_col].dt.day
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 4]) -> pd.DataFrame:
        """
        Create lag features for each time series group.
        
        IMPORTANT: This must be called ONLY on training data to prevent leakage.
        For prediction, lags are computed from known historical values.
        
        Args:
            df: DataFrame with time series data
            lags: List of lag periods to create
            
        Returns:
            DataFrame with added lag features
        """
        df = df.copy()
        df = df.sort_values(self.primary_keys + [self.date_col])
        
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby(self.primary_keys)[self.target_col].shift(lag)
        
        return df
    
    def create_rolling_features(
        self, 
        df: pd.DataFrame, 
        windows: List[int] = [3, 4, 8]
    ) -> pd.DataFrame:
        """
        Create rolling window statistics for each time series group.
        
        IMPORTANT: Uses shift(1) to prevent leakage - only uses past data.
        
        Args:
            df: DataFrame with time series data
            windows: List of window sizes for rolling statistics
            
        Returns:
            DataFrame with added rolling features
        """
        df = df.copy()
        df = df.sort_values(self.primary_keys + [self.date_col])
        
        for window in windows:
            # Shift by 1 to ensure we only use past data (no leakage)
            rolling = df.groupby(self.primary_keys)[self.target_col].shift(1).rolling(window=window, min_periods=1)
            
            df[f'rolling_mean_{window}'] = rolling.mean().reset_index(level=0, drop=True)
            df[f'rolling_std_{window}'] = rolling.std().reset_index(level=0, drop=True)
            df[f'rolling_min_{window}'] = rolling.min().reset_index(level=0, drop=True)
            df[f'rolling_max_{window}'] = rolling.max().reset_index(level=0, drop=True)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Input DataFrame
            is_training: If True, this is training data. If False, this is prediction data.
            
        Returns:
            DataFrame with all engineered features
        """
        df = df.copy()
        
        # Temporal features
        df = self.create_temporal_features(df)
        
        # Lag features (safe - uses shift)
        df = self.create_lag_features(df)
        
        # Rolling features (safe - uses shift(1) internally)
        df = self.create_rolling_features(df)
        
        return df
    
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
        Train the LightGBM forecasting model on historical time series data.
        
        This method engineers features, prepares training data, and trains a
        LightGBM model with proper validation to prevent overfitting.
        
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
        min_periods = max(12, self.forecast_horizon * 2)
        for keys, group in df.groupby(self.primary_keys):
            if len(group) < min_periods:
                self.logger.warning(f"Time series {keys} has only {len(group)} periods (recommended: {min_periods})")
        
        # Prepare data
        df_clean = self.prepare_data(df)
        
        # Engineer features
        df_features = self.engineer_features(df_clean, is_training=True)
        
        # Identify feature columns
        temporal_features = ['day_of_week', 'week_of_year', 'month', 'quarter', 'year', 'day_of_month']
        lag_features = [col for col in df_features.columns if col.startswith('lag_')]
        rolling_features = [col for col in df_features.columns if col.startswith('rolling_')]
        
        # Categorical features (primary keys)
        self.categorical_features = self.primary_keys.copy()
        
        # All feature columns
        self.feature_cols = self.primary_keys + temporal_features + lag_features + rolling_features
        
        # Remove rows with NaN in features (due to lags/rolling windows)
        df_train = df_features.dropna(subset=self.feature_cols + [self.target_col]).copy()
        
        if len(df_train) == 0:
            raise ValueError("No valid training data after feature engineering. Need more historical data.")
        
        if self.verbose:
            self.logger.info(f"Training samples after feature engineering: {len(df_train)}")
            self.logger.info(f"Features: {len(self.feature_cols)} ({len(lag_features)} lags, {len(rolling_features)} rolling)")
        
        # Prepare training data
        X_train = df_train[self.feature_cols].copy()
        y_train = df_train[self.target_col].values
        
        # Encode categorical features
        for cat_col in self.categorical_features:
            X_train[cat_col] = X_train[cat_col].astype('category')
        
        # Create validation set (last 20% of data chronologically)
        split_idx = int(len(X_train) * 0.8)
        X_train_split = X_train.iloc[:split_idx]
        y_train_split = y_train[:split_idx]
        X_val = X_train.iloc[split_idx:]
        y_val = y_train[split_idx:]
        
        # LightGBM parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        # Create datasets
        train_data = lgb.Dataset(
            X_train_split, 
            label=y_train_split,
            categorical_feature=self.categorical_features
        )
        
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=self.categorical_features,
            reference=train_data
        )
        
        # Train model
        try:
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20, verbose=False),
                    lgb.log_evaluation(period=0 if not self.verbose else 50)
                ]
            )
            
            if self.verbose:
                self.logger.success(f"Model trained successfully (best iteration: {self.model.best_iteration})")
                
        except Exception as e:
            raise RuntimeError(f"Model training failed: {str(e)}") from e

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forecasts for future periods using the trained LightGBM model.
        
        This method creates predictions for each primary key combination using
        recursive forecasting (each prediction becomes input for the next).
        Returns a unified DataFrame containing both in-sample predictions (with
        sample='train' and prediction values) and out-of-sample forecasts (with 
        sample='test' and prediction values).
        
        Args:
            df: DataFrame containing historical data (same format as training data)
        
        Returns:
            DataFrame with columns: primary_keys, date_col, target_col,
            'sample' ('train' or 'test'), and 'prediction' (in-sample predictions
            for train, forecasted values for test)
        
        Raises:
            RuntimeError: If model has not been trained (fit() not called)
            ValueError: If input DataFrame is invalid
        
        Example:
            >>> forecasts = forecaster.predict(historical_data)
            >>> # forecasts contains both in-sample and out-of-sample predictions
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
        
        # Generate in-sample predictions for historical data
        historical = df_clean.copy()
        historical['sample'] = 'train'
        
        # Engineer features for in-sample prediction
        historical_features = self.engineer_features(historical, is_training=False)
        
        # Get features that exist in the data
        available_features = [f for f in self.feature_cols if f in historical_features.columns]
        
        if available_features:
            X_historical = historical_features[available_features].copy()
            
            # Encode categorical features
            for cat_col in self.categorical_features:
                if cat_col in X_historical.columns:
                    X_historical[cat_col] = X_historical[cat_col].astype('category')
            
            # Make in-sample predictions
            try:
                in_sample_preds = self.model.predict(X_historical, num_iteration=self.model.best_iteration)
                in_sample_preds = np.maximum(0, in_sample_preds)  # Ensure non-negative
                historical['prediction'] = in_sample_preds
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Failed to generate in-sample predictions: {str(e)}")
                historical['prediction'] = np.nan
        else:
            historical['prediction'] = np.nan
        
        # Generate forecasts for each primary key combination
        forecast_dfs = []
        
        for keys, group in df_clean.groupby(self.primary_keys):
            try:
                # Sort by date
                group = group.sort_values(self.date_col).copy()
                
                # Get last date
                last_date = group[self.date_col].max()
                
                # Create future dates
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(1, unit=self.frequency),
                    periods=self.forecast_horizon,
                    freq=self.frequency
                )
                
                # Initialize forecast storage
                forecast_values = []
                
                # Recursive forecasting: predict one step at a time
                for step, future_date in enumerate(future_dates):
                    # Create a row for this future period
                    future_row = pd.DataFrame({self.date_col: [future_date]})
                    
                    # Add primary key values
                    if isinstance(keys, tuple):
                        for i, key_col in enumerate(self.primary_keys):
                            future_row[key_col] = keys[i]
                    else:
                        future_row[self.primary_keys[0]] = keys
                    
                    # Add placeholder target (will be replaced with prediction)
                    future_row[self.target_col] = 0
                    
                    # Combine historical + previous predictions + current row
                    combined = pd.concat([group, future_row], ignore_index=True)
                    
                    # Engineer features for the combined data
                    combined_features = self.engineer_features(combined, is_training=False)
                    
                    # Get the last row (our prediction target)
                    pred_row = combined_features.iloc[[-1]]
                    
                    # Check if all required features are present
                    missing_features = [f for f in self.feature_cols if f not in pred_row.columns]
                    if missing_features:
                        # Fill missing features with 0
                        for feat in missing_features:
                            pred_row[feat] = 0
                    
                    # Prepare features for prediction
                    X_pred = pred_row[self.feature_cols].copy()
                    
                    # Encode categorical features
                    for cat_col in self.categorical_features:
                        X_pred[cat_col] = X_pred[cat_col].astype('category')
                    
                    # Make prediction
                    pred_value = self.model.predict(X_pred, num_iteration=self.model.best_iteration)[0]
                    pred_value = max(0, pred_value)  # Ensure non-negative
                    
                    forecast_values.append(pred_value)
                    
                    # Add this prediction to the group for next iteration
                    new_row = pd.DataFrame({
                        self.date_col: [future_date],
                        self.target_col: [pred_value]
                    })
                    
                    if isinstance(keys, tuple):
                        for i, key_col in enumerate(self.primary_keys):
                            new_row[key_col] = keys[i]
                    else:
                        new_row[self.primary_keys[0]] = keys
                    
                    group = pd.concat([group, new_row], ignore_index=True)
                
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
                if self.verbose:
                    self.logger.warning(f"Failed to generate forecast for {keys}: {str(e)}")
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
