"""
Router module for delivery routing and truck assignment.

This module provides functionality to distribute forecasted demand among customers,
assign deliveries to trucks based on payload constraints, and optimize delivery routes
using a nearest neighbor heuristic.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np


class Router:
    """
    Router class for managing delivery routing and truck assignment.
    
    This class handles the distribution of forecasted sales to customers,
    assigns deliveries to trucks while respecting payload constraints,
    and optimizes routes using a simple nearest neighbor heuristic based
    on postal code proximity.
    
    Attributes:
        primary_keys (List[str]): Grouping columns (e.g., ['store', 'product'])
        date_col (str): Name of the date column
        target_col (str): Name of the target variable column (e.g., 'sales')
        max_payload (int): Maximum units per truck
        origin (str): Starting postal code for all routes
        customers_db (pd.DataFrame): Generated customer database
    """
    
    def __init__(
        self,
        primary_keys: List[str],
        date_col: str = 'date',
        target_col: str = 'sales',
        max_payload: int = 10,
        origin: str = '08020'
    ):
        """
        Initialize the Router with configuration parameters.
        
        Args:
            primary_keys: List of column names to group by (e.g., ['store', 'product'])
            date_col: Name of the date column (default: 'date')
            target_col: Name of the target variable column (default: 'sales')
            max_payload: Maximum units that can be loaded on a single truck (default: 10)
            origin: Starting postal code for all delivery routes (default: '08020')
        
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate primary_keys
        if not primary_keys or len(primary_keys) == 0:
            raise ValueError("primary_keys must contain at least one column name")
        
        # Validate max_payload
        if max_payload <= 0:
            raise ValueError("max_payload must be positive")
        
        # Validate origin format
        if not (isinstance(origin, str) and len(origin) == 5 and origin.isdigit()):
            raise ValueError("origin must be a 5-digit postal code string")
        
        self.primary_keys = primary_keys
        self.date_col = date_col
        self.target_col = target_col
        self.max_payload = max_payload
        self.origin = origin
        self.customers_db = None

    def generate_customers(self, n_customers: int = 20) -> pd.DataFrame:
        """
        Generate a database of sample customers with unique IDs and postal codes.
        
        Creates customer records with unique customer IDs and random 5-digit postal codes.
        The generated customer database is stored in self.customers_db for later use.
        
        Args:
            n_customers: Number of customers to generate (default: 20)
            
        Returns:
            pd.DataFrame: Customer database with columns ['customer_id', 'destination']
        """
        # Generate unique customer IDs
        customer_ids = [f'CUST_{i:04d}' for i in range(1, n_customers + 1)]
        
        # Generate random 5-digit postal codes
        postal_codes = [f'{np.random.randint(10000, 99999):05d}' for _ in range(n_customers)]
        
        # Create customer database
        self.customers_db = pd.DataFrame({
            'customer_id': customer_ids,
            'destination': postal_codes
        })
        
        return self.customers_db

    def distribute_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Distribute forecasted sales among customers for delivery planning.
        
        For each (store, product, date) combination in the forecast data, this method
        randomly selects customers and splits the sales quantity evenly across them.
        
        Args:
            df: DataFrame with forecast data containing primary_keys, date_col, and target_col
            
        Returns:
            pd.DataFrame: Delivery records with columns [primary_keys, date, sales, customer, destination, units]
        """
        if self.customers_db is None:
            raise ValueError("Customer database not initialized. Call generate_customers() first.")
        
        # Filter to forecast data only (sample == 'test')
        if 'sample' in df.columns:
            forecast_df = df[df['sample'] == 'test'].copy()
        else:
            forecast_df = df.copy()
        
        # Filter out rows with zero or negative sales/predictions
        # Use prediction column if available (for forecasts), otherwise use target_col
        if 'prediction' in forecast_df.columns:
            forecast_df = forecast_df[forecast_df['prediction'] > 0].copy()
        else:
            forecast_df = forecast_df[forecast_df[self.target_col] > 0].copy()
        
        deliveries = []
        
        # Group by primary keys and date
        group_cols = self.primary_keys + [self.date_col]
        
        for group_key, group_data in forecast_df.groupby(group_cols):
            # Get sales value (use prediction if available, otherwise target_col)
            if 'prediction' in group_data.columns:
                sales = group_data['prediction'].iloc[0]
            else:
                sales = group_data[self.target_col].iloc[0]
            
            # Skip if sales is NaN or zero
            if pd.isna(sales) or sales <= 0:
                continue
            
            # Calculate minimum customers needed to respect max_payload
            # Ensure each delivery is at most max_payload
            min_customers_needed = int(np.ceil(sales / self.max_payload))
            
            # Randomly select customers (at least min_customers_needed, up to 5 or available)
            min_cust = max(2, min_customers_needed)
            max_cust = min(6, len(self.customers_db) + 1)
            
            # Ensure min_cust doesn't exceed max_cust
            if min_cust >= max_cust:
                n_customers_for_delivery = min(min_customers_needed, len(self.customers_db))
            else:
                n_customers_for_delivery = np.random.randint(min_cust, max_cust)
            selected_customers = self.customers_db.sample(n=n_customers_for_delivery, replace=False)
            
            # Split sales evenly across customers
            units_per_customer = sales / n_customers_for_delivery
            
            # Create delivery record for each customer
            for _, customer in selected_customers.iterrows():
                delivery_record = {}
                
                # Add primary key values
                for i, pk in enumerate(self.primary_keys):
                    if isinstance(group_key, tuple):
                        delivery_record[pk] = group_key[i]
                    else:
                        delivery_record[pk] = group_key
                
                # Add date
                if isinstance(group_key, tuple):
                    delivery_record[self.date_col] = group_key[-1]
                else:
                    delivery_record[self.date_col] = group_key
                
                # Add delivery details
                delivery_record[self.target_col] = sales
                delivery_record['customer'] = customer['customer_id']
                delivery_record['destination'] = customer['destination']
                delivery_record['units'] = units_per_customer
                
                deliveries.append(delivery_record)
        
        return pd.DataFrame(deliveries)

    def assign_trucks(self, deliveries_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign deliveries to trucks while respecting payload constraints.
        
        Groups deliveries by date, sorts by destination for locality, and assigns
        deliveries to trucks sequentially. Creates a new truck when the current
        truck's payload would exceed max_payload.
        
        Args:
            deliveries_df: DataFrame with delivery records containing 'units' and 'destination'
            
        Returns:
            pd.DataFrame: Updated deliveries DataFrame with 'truck' column added
        """
        deliveries_df = deliveries_df.copy()
        deliveries_df['truck'] = None
        
        # Global truck counter across all dates
        truck_num = 1
        
        # Group by date
        for date, date_group in deliveries_df.groupby(self.date_col):
            # Sort by destination for locality
            date_group = date_group.sort_values('destination').copy()
            
            # Assign trucks for this date
            current_payload = 0
            
            for idx, row in date_group.iterrows():
                units = row['units']
                
                # Check if adding this delivery would exceed max_payload
                if current_payload + units > self.max_payload and current_payload > 0:
                    # Start a new truck (only if current truck has something)
                    truck_num += 1
                    current_payload = 0
                
                # Assign to current truck
                deliveries_df.loc[idx, 'truck'] = f'truck_{truck_num}'
                current_payload += units
            
            # Move to next truck for next date (each date gets fresh trucks)
            if current_payload > 0:
                truck_num += 1
        
        return deliveries_df

    def optimize_routes(self, deliveries_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Optimize delivery routes using a nearest neighbor heuristic.
        
        For each truck, extracts unique destinations, calculates distances based on
        postal code numeric difference, and orders destinations to minimize total
        distance using a nearest neighbor approach.
        
        Args:
            deliveries_df: DataFrame with delivery records including 'truck' and 'destination'
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                - Delivery-level data with route information
                - Route summary with columns [truck, route_order, origin, destinations, total_distance]
        """
        deliveries_df = deliveries_df.copy()
        route_summaries = []
        
        # Group by truck
        for truck, truck_group in deliveries_df.groupby('truck'):
            # Get unique destinations for this truck
            destinations = truck_group['destination'].unique().tolist()
            
            if len(destinations) == 0:
                continue
            
            # Calculate route using nearest neighbor heuristic
            route_order = []
            current_location = self.origin
            remaining_destinations = destinations.copy()
            total_distance = 0
            
            while remaining_destinations:
                # Find nearest destination
                min_distance = float('inf')
                nearest_dest = None
                
                for dest in remaining_destinations:
                    # Calculate distance as numeric difference between postal codes
                    distance = abs(int(current_location) - int(dest))
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_dest = dest
                
                # Move to nearest destination
                route_order.append(nearest_dest)
                total_distance += min_distance
                current_location = nearest_dest
                remaining_destinations.remove(nearest_dest)
            
            # Create route summary record
            route_summary = {
                'truck': truck,
                'route_order': ', '.join([str(i+1) for i in range(len(route_order))]),
                'origin': self.origin,
                'destinations': ', '.join(route_order),
                'total_distance': total_distance
            }
            route_summaries.append(route_summary)
        
        route_summary_df = pd.DataFrame(route_summaries)
        
        return deliveries_df, route_summary_df
