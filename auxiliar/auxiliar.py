"""
Auxiliar module for synthetic supply chain data generation.

This module provides utilities to generate realistic synthetic data for
demonstrating supply chain optimization capabilities.
"""

import pandas as pd
import numpy as np
from typing import Optional
import string


def generate_data(
    n_stores: int = 3,
    n_products: int = 2,
    n_weeks: int = 52,
    start_date: str = '2024-01-01',
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate synthetic supply chain data with realistic patterns.
    
    Creates a dataset with sales, inventory, and customer information for
    multiple store-product combinations over a specified time period. The
    generated sales data includes trend, seasonality, and random noise
    components to simulate realistic demand patterns.
    
    Parameters
    ----------
    n_stores : int, default=3
        Number of stores to generate (labeled A, B, C, ...)
    n_products : int, default=2
        Number of products to generate (labeled X, Y, Z, ...)
    n_weeks : int, default=52
        Number of weeks of historical data to generate
    start_date : str, default='2024-01-01'
        Starting date for the time series (format: 'YYYY-MM-DD')
    seed : int or None, default=42
        Random seed for reproducibility. Set to None for random generation.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - store: Store identifier (str)
        - product: Product identifier (str)
        - date: Date of observation (datetime64)
        - sales: Units sold (float)
        - inventory: Current inventory level (float)
        - customer_id: Customer identifier (str)
        - destination: Postal code (str)
    
    Examples
    --------
    >>> data = generate_data(n_stores=2, n_products=2, n_weeks=20)
    >>> data.shape
    (80, 7)
    >>> data.columns.tolist()
    ['store', 'product', 'date', 'sales', 'inventory', 'customer_id', 'destination']
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Generate date range with weekly frequency
    dates = pd.date_range(start=start_date, periods=n_weeks, freq='W')
    
    # Generate store and product identifiers
    stores = [string.ascii_uppercase[i] for i in range(n_stores)]
    products = [string.ascii_uppercase[i] for i in range(n_products)]
    
    # Create all combinations of stores, products, and dates
    data_list = []
    
    for store in stores:
        for product in products:
            # Generate base demand level (random between 50-200)
            base_demand = np.random.uniform(50, 200)
            
            # Create time index for trend and seasonality
            time_index = np.arange(n_weeks)
            
            # Add linear trend component (small positive slope)
            trend = time_index * np.random.uniform(0.5, 2.0)
            
            # Add seasonal pattern (sine wave with 13-week period for quarterly seasonality)
            seasonality = 20 * np.sin(2 * np.pi * time_index / 13)
            
            # Add random noise (normal distribution)
            noise = np.random.normal(0, 15, n_weeks)
            
            # Combine components and ensure non-negative values
            sales = base_demand + trend + seasonality + noise
            sales = np.maximum(sales, 0)  # Ensure non-negative
            
            # Assign initial inventory (random between 200-500)
            inventory = np.random.uniform(200, 500, n_weeks)
            
            # Generate random customer IDs (format: CUST_XXXX)
            customer_ids = [f"CUST_{np.random.randint(1000, 9999)}" for _ in range(n_weeks)]
            
            # Generate random postal codes (5-digit strings)
            postal_codes = [f"{np.random.randint(10000, 99999)}" for _ in range(n_weeks)]
            
            # Create records for this store-product combination
            for i in range(n_weeks):
                data_list.append({
                    'store': store,
                    'product': product,
                    'date': dates[i],
                    'sales': sales[i],
                    'inventory': inventory[i],
                    'customer_id': customer_ids[i],
                    'destination': postal_codes[i]
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data_list)
    
    return df
