"""
Optimizer Module for Inventory Simulation

This module implements inventory optimization using a reorder point policy with safety stock.
The Optimizer simulates inventory flow over a planning horizon, placing orders when inventory
falls below the reorder point and tracking shipments with lead time delays.
"""

from typing import List
import pandas as pd
import numpy as np
from scipy.stats import norm


class Optimizer:
    """
    Inventory optimizer implementing (s, S) reorder point policy.
    
    This class simulates inventory management by:
    - Calculating safety stock based on demand variability and service level
    - Determining reorder points to trigger replenishment
    - Simulating period-by-period inventory flow with orders and shipments
    - Tracking lead time between order placement and receipt
    
    The inventory policy follows these rules:
    - Review inventory at specified review_period intervals
    - Place order when inventory < reorder_point
    - Order quantity = reorder_point - current_inventory
    - Shipments arrive after lead_time periods
    
    Parameters
    ----------
    primary_keys : List[str]
        Column names for grouping (e.g., ['store', 'product'])
    date_col : str, default='date'
        Name of the date/time column
    target_col : str, default='sales'
        Name of the demand/sales column to forecast
    inv_col : str, default='inventory'
        Name of the initial inventory column
    planning_horizon : int, default=8
        Number of periods to simulate into the future
    service_level : float, default=0.95
        Target service level (0.0 to 1.0), used for safety stock calculation
    review_period : int, default=1
        Frequency of inventory reviews (1 = every period, 2 = every other period)
    lead_time : int, default=2
        Number of periods between order placement and receipt
        
    Attributes
    ----------
    primary_keys : List[str]
        Grouping columns
    date_col : str
        Date column name
    target_col : str
        Demand column name
    inv_col : str
        Inventory column name
    planning_horizon : int
        Simulation horizon
    service_level : float
        Target service level
    review_period : int
        Review frequency
    lead_time : int
        Order lead time
        
    Examples
    --------
    >>> optimizer = Optimizer(
    ...     primary_keys=['store', 'product'],
    ...     planning_horizon=12,
    ...     service_level=0.95,
    ...     lead_time=2
    ... )
    >>> inventory_plan = optimizer.simulate(forecast_df)
    """
    
    def __init__(
        self,
        primary_keys: List[str],
        date_col: str = 'date',
        target_col: str = 'sales',
        inv_col: str = 'inventory',
        planning_horizon: int = 8,
        service_level: float = 0.95,
        review_period: int = 1,
        lead_time: int = 2
    ):
        # Validate parameters
        if not primary_keys or len(primary_keys) == 0:
            raise ValueError("primary_keys must contain at least one column name")
        
        if planning_horizon <= 0:
            raise ValueError("planning_horizon must be positive")
        
        if not (0.5 <= service_level < 1.0):
            raise ValueError("service_level must be between 0.5 and 0.999")
        
        if review_period <= 0:
            raise ValueError("review_period must be positive")
        
        if lead_time < 0:
            raise ValueError("lead_time must be non-negative")
        
        if lead_time > planning_horizon:
            import warnings
            warnings.warn(
                f"lead_time ({lead_time}) exceeds planning_horizon ({planning_horizon}). "
                "Some orders may not be received within the simulation period."
            )
        
        # Store configuration
        self.primary_keys = primary_keys
        self.date_col = date_col
        self.target_col = target_col
        self.inv_col = inv_col
        self.planning_horizon = planning_horizon
        self.service_level = service_level
        self.review_period = review_period
        self.lead_time = lead_time

    def calculate_safety_stock(self, demand_series: pd.Series) -> float:
        """
        Calculate safety stock based on demand variability and service level.
        
        Safety stock provides a buffer to protect against demand uncertainty and
        supply variability. It is calculated using the formula:
        
        safety_stock = z_score * std_dev * sqrt(lead_time)
        
        where:
        - z_score is derived from the target service level
        - std_dev is the standard deviation of historical demand
        - lead_time is the replenishment lead time
        
        Parameters
        ----------
        demand_series : pd.Series
            Historical demand values for a single product-location combination
            
        Returns
        -------
        float
            Calculated safety stock quantity
            
        Notes
        -----
        - If demand has zero variance, returns a minimum safety stock of 1.0
        - Uses normal distribution assumption for demand uncertainty
        - Higher service levels result in higher safety stock
        
        Examples
        --------
        >>> demand = pd.Series([100, 110, 95, 105, 98])
        >>> safety_stock = optimizer.calculate_safety_stock(demand)
        """
        # Calculate standard deviation of demand
        std_dev = demand_series.std()
        
        # Handle edge case of zero variance
        if std_dev == 0 or pd.isna(std_dev):
            return 1.0  # Minimum safety stock
        
        # Calculate z-score from service level using inverse normal CDF
        z_score = norm.ppf(self.service_level)
        
        # Calculate safety stock: z * σ * sqrt(L)
        safety_stock = z_score * std_dev * np.sqrt(self.lead_time)
        
        return max(safety_stock, 0.0)  # Ensure non-negative

    def calculate_reorder_point(self, avg_demand: float, safety_stock: float) -> float:
        """
        Calculate the reorder point for inventory replenishment.
        
        The reorder point is the inventory level at which a new order should be placed.
        It accounts for expected demand during lead time plus safety stock buffer:
        
        reorder_point = (average_demand * lead_time) + safety_stock
        
        Parameters
        ----------
        avg_demand : float
            Average demand per period
        safety_stock : float
            Safety stock quantity (from calculate_safety_stock)
            
        Returns
        -------
        float
            Reorder point quantity
            
        Notes
        -----
        - The first term covers expected demand during lead time
        - Safety stock provides buffer against uncertainty
        - When inventory falls below this level, an order is triggered
        
        Examples
        --------
        >>> reorder_point = optimizer.calculate_reorder_point(
        ...     avg_demand=100,
        ...     safety_stock=50
        ... )
        """
        # Reorder point = expected demand during lead time + safety stock
        reorder_point = (avg_demand * self.lead_time) + safety_stock
        
        return max(reorder_point, 0.0)  # Ensure non-negative

    def simulate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate inventory flow over the planning horizon.
        
        This method performs a period-by-period simulation of inventory management:
        1. Groups data by primary_keys (e.g., store-product combinations)
        2. Calculates safety stock and reorder point for each group
        3. Simulates inventory flow:
           - Deducts forecasted demand each period
           - Checks inventory against reorder point (on review periods)
           - Places orders when inventory falls below reorder point
           - Tracks orders in transit with lead time
           - Adds shipments when orders arrive
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing forecasted demand with columns:
            - primary_keys columns (e.g., 'store', 'product')
            - date_col: date/time column
            - target_col: forecasted demand (prediction column)
            - 'sample': indicator of 'train' or 'test' data
            
        Returns
        -------
        pd.DataFrame
            Simulation results with columns:
            - All original columns from input
            - safety_stock: calculated safety stock level
            - reorder_point: calculated reorder point
            - inventory: end-of-period inventory level
            - order: quantity ordered in this period
            - shipment: quantity received in this period
            
        Notes
        -----
        - Initial inventory is taken from the last historical period
        - Orders are placed when inventory < reorder_point on review periods
        - Order quantity = reorder_point - current_inventory
        - Negative inventory indicates stockout situation
        - Only simulates for 'test' (forecast) periods
        
        Examples
        --------
        >>> forecast_df = forecaster.predict(data)
        >>> inventory_plan = optimizer.simulate(forecast_df)
        >>> print(inventory_plan[['store', 'product', 'date', 'inventory', 'order']])
        """
        results = []
        
        # Group by primary keys (e.g., store-product combinations)
        for group_keys, group_df in df.groupby(self.primary_keys):
            # Sort by date to ensure chronological order
            group_df = group_df.sort_values(self.date_col).copy()
            
            # Separate historical (train) and forecast (test) data
            train_data = group_df[group_df['sample'] == 'train'].copy()
            test_data = group_df[group_df['sample'] == 'test'].copy()
            
            # Calculate safety stock from historical demand
            if len(train_data) > 0:
                historical_demand = train_data[self.target_col].dropna()
                if len(historical_demand) > 0:
                    avg_demand = historical_demand.mean()
                    safety_stock = self.calculate_safety_stock(historical_demand)
                    reorder_point = self.calculate_reorder_point(avg_demand, safety_stock)
                else:
                    avg_demand = 0
                    safety_stock = 1.0
                    reorder_point = 1.0
            else:
                avg_demand = 0
                safety_stock = 1.0
                reorder_point = 1.0
            
            # Initialize inventory from last historical period
            if len(train_data) > 0 and self.inv_col in train_data.columns:
                current_inventory = train_data[self.inv_col].iloc[-1]
            else:
                current_inventory = reorder_point * 2  # Start with 2x reorder point
            
            # Track orders in transit: list of (arrival_period, quantity) tuples
            orders_in_transit = []
            
            # Add safety stock and reorder point to historical data
            for idx in train_data.index:
                train_data.loc[idx, 'safety_stock'] = safety_stock
                train_data.loc[idx, 'reorder_point'] = reorder_point
                train_data.loc[idx, 'inventory'] = np.nan  # Not simulated for historical
                train_data.loc[idx, 'order'] = 0.0
                train_data.loc[idx, 'shipment'] = 0.0
            
            results.append(train_data)
            
            # Simulate forecast periods
            if len(test_data) > 0:
                test_data = test_data.head(self.planning_horizon).copy()
                
                for period_idx, (idx, row) in enumerate(test_data.iterrows()):
                    # Get forecasted demand (use prediction column if available, else target_col)
                    if 'prediction' in row and not pd.isna(row['prediction']):
                        demand = row['prediction']
                    else:
                        demand = row[self.target_col] if not pd.isna(row[self.target_col]) else 0
                    
                    # Process arriving shipments
                    shipment = 0.0
                    remaining_orders = []
                    for arrival_period, quantity in orders_in_transit:
                        if arrival_period <= period_idx:
                            shipment += quantity
                        else:
                            remaining_orders.append((arrival_period, quantity))
                    orders_in_transit = remaining_orders
                    
                    # Add shipments to inventory
                    current_inventory += shipment
                    
                    # Deduct demand from inventory
                    current_inventory -= demand
                    
                    # Check if we should review inventory and place order
                    order = 0.0
                    if (period_idx % self.review_period == 0) and (current_inventory < reorder_point):
                        # Place order to bring inventory back to reorder point
                        order = reorder_point - current_inventory
                        # Add order to transit queue (arrives after lead_time periods)
                        arrival_period = period_idx + self.lead_time
                        orders_in_transit.append((arrival_period, order))
                    
                    # Record simulation results
                    test_data.loc[idx, 'safety_stock'] = safety_stock
                    test_data.loc[idx, 'reorder_point'] = reorder_point
                    test_data.loc[idx, 'inventory'] = current_inventory
                    test_data.loc[idx, 'order'] = order
                    test_data.loc[idx, 'shipment'] = shipment
                
                results.append(test_data)
        
        # Combine all results
        if len(results) == 0:
            return pd.DataFrame()
        
        result_df = pd.concat(results, ignore_index=True)
        
        # Ensure proper column order
        base_cols = self.primary_keys + [self.date_col, self.target_col, 'sample']
        if 'prediction' in result_df.columns:
            base_cols.append('prediction')
        
        sim_cols = ['safety_stock', 'reorder_point', 'inventory', 'order', 'shipment']
        
        # Add any remaining columns
        other_cols = [col for col in result_df.columns if col not in base_cols + sim_cols]
        
        final_cols = base_cols + sim_cols + other_cols
        final_cols = [col for col in final_cols if col in result_df.columns]
        
        return result_df[final_cols]
