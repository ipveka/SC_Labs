"""
SC Labs - Supply Chain Optimization Main Orchestration Script

This script orchestrates the complete supply chain optimization workflow:
1. Generate synthetic data
2. Forecast demand using GluonTS
3. Optimize inventory with reorder point policy
4. Route deliveries with truck assignment

Usage:
    python main.py [options]
    
Examples:
    python main.py
    python main.py --n_stores 5 --n_products 3 --save
    python main.py --verbose --save
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Import modules
from auxiliar.auxiliar import generate_data
from forecaster.forecaster import Forecaster
from optimizer.optimizer import Optimizer
from router.router import Router


def print_summary(
    forecasts: pd.DataFrame,
    inventory_plan: pd.DataFrame,
    deliveries: pd.DataFrame,
    routes: pd.DataFrame,
    verbose: bool = False
) -> None:
    """
    Display summary statistics for all pipeline stages.
    
    Args:
        forecasts: Forecast DataFrame from Forecaster
        inventory_plan: Inventory simulation DataFrame from Optimizer
        deliveries: Delivery DataFrame from Router
        routes: Route summary DataFrame from Router
        verbose: If True, display detailed information
    """
    print("\n" + "="*80)
    print("SC LABS - SUPPLY CHAIN OPTIMIZATION SUMMARY")
    print("="*80)
    
    # Forecast Summary
    print("\n📊 FORECAST SUMMARY")
    print("-" * 80)
    
    forecast_test = forecasts[forecasts['sample'] == 'test']
    
    if len(forecast_test) > 0:
        print(f"Forecast periods: {len(forecast_test)}")
        print(f"Total forecasted demand: {forecast_test['prediction'].sum():.2f} units")
        print(f"Average forecasted demand: {forecast_test['prediction'].mean():.2f} units/period")
        
        if verbose:
            print("\nForecast by Store-Product:")
            summary = forecast_test.groupby(['store', 'product'])['prediction'].agg(['sum', 'mean', 'std'])
            summary.columns = ['Total', 'Mean', 'Std Dev']
            print(summary.round(2))
    else:
        print("No forecast data available")
    
    # Inventory Summary
    print("\n📦 INVENTORY SUMMARY")
    print("-" * 80)
    
    inv_test = inventory_plan[inventory_plan['sample'] == 'test']
    
    if len(inv_test) > 0:
        avg_inventory = inv_test['inventory'].mean()
        min_inventory = inv_test['inventory'].min()
        max_inventory = inv_test['inventory'].max()
        total_orders = inv_test['order'].sum()
        total_shipments = inv_test['shipment'].sum()
        stockouts = (inv_test['inventory'] < 0).sum()
        
        print(f"Average inventory level: {avg_inventory:.2f} units")
        print(f"Min inventory level: {min_inventory:.2f} units")
        print(f"Max inventory level: {max_inventory:.2f} units")
        print(f"Total orders placed: {total_orders:.2f} units")
        print(f"Total shipments received: {total_shipments:.2f} units")
        print(f"Stockout periods: {stockouts}")
        
        if verbose:
            print("\nInventory by Store-Product:")
            summary = inv_test.groupby(['store', 'product']).agg({
                'inventory': 'mean',
                'order': 'sum',
                'safety_stock': 'first',
                'reorder_point': 'first'
            })
            summary.columns = ['Avg Inventory', 'Total Orders', 'Safety Stock', 'Reorder Point']
            print(summary.round(2))
    else:
        print("No inventory simulation data available")
    
    # Routing Summary
    print("\n🚚 ROUTING SUMMARY")
    print("-" * 80)
    
    if len(deliveries) > 0 and len(routes) > 0:
        total_trucks = routes['truck'].nunique()
        total_distance = routes['total_distance'].sum()
        avg_distance = routes['total_distance'].mean()
        total_deliveries = len(deliveries)
        avg_payload = deliveries.groupby('truck')['units'].sum().mean()
        
        print(f"Total trucks required: {total_trucks}")
        print(f"Total deliveries: {total_deliveries}")
        print(f"Total route distance: {total_distance:.2f} units")
        print(f"Average route distance: {avg_distance:.2f} units/truck")
        print(f"Average truck payload: {avg_payload:.2f} units")
        
        if verbose:
            print("\nRoute Details:")
            print(routes.to_string(index=False))
    else:
        print("No routing data available")
    
    print("\n" + "="*80)


def save_outputs(
    forecasts: pd.DataFrame,
    inventory_plan: pd.DataFrame,
    deliveries: pd.DataFrame,
    routes: pd.DataFrame,
    output_dir: str = 'outputs'
) -> None:
    """
    Save all output DataFrames to CSV files with timestamps.
    
    Args:
        forecasts: Forecast DataFrame
        inventory_plan: Inventory simulation DataFrame
        deliveries: Delivery DataFrame
        routes: Route summary DataFrame
        output_dir: Directory to save outputs (default: 'outputs')
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate timestamp for versioning
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save files
    forecasts.to_csv(output_path / f'forecasts_{timestamp}.csv', index=False)
    inventory_plan.to_csv(output_path / f'inventory_plan_{timestamp}.csv', index=False)
    deliveries.to_csv(output_path / f'deliveries_{timestamp}.csv', index=False)
    routes.to_csv(output_path / f'routes_{timestamp}.csv', index=False)
    
    print(f"\n✅ Outputs saved to '{output_dir}/' directory with timestamp {timestamp}")


def main(
    n_stores: int = 3,
    n_products: int = 2,
    n_weeks: int = 52,
    start_date: str = '2024-01-01',
    forecast_horizon: int = 4,
    model_type: str = 'simple_feedforward',
    planning_horizon: int = 8,
    service_level: float = 0.95,
    lead_time: int = 2,
    max_payload: int = 10,
    n_customers: int = 20,
    save: bool = False,
    verbose: bool = False
) -> dict:
    """
    Main orchestration function for the supply chain optimization workflow.
    
    Args:
        n_stores: Number of stores to simulate
        n_products: Number of products to simulate
        n_weeks: Number of weeks of historical data
        start_date: Start date for data generation
        forecast_horizon: Number of periods to forecast
        model_type: Forecasting model ('simple_feedforward', 'deepar', 'transformer')
        planning_horizon: Number of periods for inventory simulation
        service_level: Target service level for safety stock
        lead_time: Order lead time in periods
        max_payload: Maximum units per truck
        n_customers: Number of customers to generate
        save: If True, save outputs to files
        verbose: If True, display detailed information
        
    Returns:
        dict: Dictionary containing all output DataFrames
    """
    print("\n" + "="*80)
    print("SC LABS - SUPPLY CHAIN OPTIMIZATION PIPELINE")
    print("="*80)
    
    try:
        # Stage 1: Generate synthetic data
        print("\n[1/4] Generating synthetic data...")
        data = generate_data(
            n_stores=n_stores,
            n_products=n_products,
            n_weeks=n_weeks,
            start_date=start_date
        )
        print(f"✓ Generated {len(data)} records for {n_stores} stores × {n_products} products × {n_weeks} weeks")
        
        if verbose:
            print(f"\nData shape: {data.shape}")
            print(f"Date range: {data['date'].min()} to {data['date'].max()}")
            print(f"\nSample data:\n{data.head()}")
        
        # Stage 2: Demand forecasting
        print("\n[2/4] Forecasting demand...")
        
        # Split data into train/test
        # Use all but last forecast_horizon periods for training
        split_date = data['date'].max() - pd.Timedelta(weeks=forecast_horizon)
        train_data = data[data['date'] <= split_date]
        
        # Initialize and train forecaster
        forecaster = Forecaster(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            frequency='W',
            forecast_horizon=forecast_horizon,
            model_type=model_type
        )
        
        forecaster.fit(train_data)
        forecasts = forecaster.predict(data)
        
        forecast_test = forecasts[forecasts['sample'] == 'test']
        print(f"✓ Generated forecasts for {len(forecast_test)} periods")
        
        if verbose:
            print(f"\nForecast shape: {forecasts.shape}")
            print(f"\nSample forecasts:\n{forecast_test.head(10)}")
        
        # Stage 3: Inventory optimization
        print("\n[3/4] Optimizing inventory...")
        
        optimizer = Optimizer(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            inv_col='inventory',
            planning_horizon=planning_horizon,
            service_level=service_level,
            review_period=1,
            lead_time=lead_time
        )
        
        inventory_plan = optimizer.simulate(forecasts)
        
        inv_test = inventory_plan[inventory_plan['sample'] == 'test']
        print(f"✓ Simulated inventory for {len(inv_test)} periods")
        
        if verbose:
            print(f"\nInventory plan shape: {inventory_plan.shape}")
            print(f"\nSample inventory plan:\n{inv_test.head(10)}")
        
        # Stage 4: Delivery routing
        print("\n[4/4] Optimizing delivery routes...")
        
        router = Router(
            primary_keys=['store', 'product'],
            date_col='date',
            target_col='sales',
            max_payload=max_payload,
            origin='08020'
        )
        
        # Generate customers
        router.generate_customers(n_customers=n_customers)
        print(f"✓ Generated {n_customers} customers")
        
        # Distribute demand to customers
        deliveries = router.distribute_demand(forecasts)
        print(f"✓ Created {len(deliveries)} delivery records")
        
        # Assign trucks
        deliveries = router.assign_trucks(deliveries)
        n_trucks = deliveries['truck'].nunique()
        print(f"✓ Assigned deliveries to {n_trucks} trucks")
        
        # Optimize routes
        deliveries, routes = router.optimize_routes(deliveries)
        print(f"✓ Optimized {len(routes)} routes")
        
        if verbose:
            print(f"\nDeliveries shape: {deliveries.shape}")
            print(f"Routes shape: {routes.shape}")
            print(f"\nSample deliveries:\n{deliveries.head(10)}")
            print(f"\nSample routes:\n{routes.head()}")
        
        # Display summary
        print_summary(forecasts, inventory_plan, deliveries, routes, verbose=verbose)
        
        # Save outputs if requested
        if save:
            save_outputs(forecasts, inventory_plan, deliveries, routes)
        
        print("\n✅ Pipeline completed successfully!")
        
        return {
            'data': data,
            'forecasts': forecasts,
            'inventory_plan': inventory_plan,
            'deliveries': deliveries,
            'routes': routes
        }
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SC Labs - Supply Chain Optimization Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data generation parameters
    parser.add_argument('--n_stores', type=int, default=3,
                        help='Number of stores to simulate')
    parser.add_argument('--n_products', type=int, default=2,
                        help='Number of products to simulate')
    parser.add_argument('--n_weeks', type=int, default=52,
                        help='Number of weeks of historical data')
    parser.add_argument('--start_date', type=str, default='2024-01-01',
                        help='Start date for data generation (YYYY-MM-DD)')
    
    # Forecasting parameters
    parser.add_argument('--forecast_horizon', type=int, default=4,
                        help='Number of periods to forecast')
    parser.add_argument('--model_type', type=str, default='simple_feedforward',
                        choices=['simple_feedforward', 'deepar', 'transformer'],
                        help='Forecasting model type')
    
    # Inventory optimization parameters
    parser.add_argument('--planning_horizon', type=int, default=8,
                        help='Number of periods for inventory simulation')
    parser.add_argument('--service_level', type=float, default=0.95,
                        help='Target service level (0.0-1.0)')
    parser.add_argument('--lead_time', type=int, default=2,
                        help='Order lead time in periods')
    
    # Routing parameters
    parser.add_argument('--max_payload', type=int, default=10,
                        help='Maximum units per truck')
    parser.add_argument('--n_customers', type=int, default=20,
                        help='Number of customers to generate')
    
    # Output options
    parser.add_argument('--save', action='store_true',
                        help='Save outputs to CSV files')
    parser.add_argument('--verbose', action='store_true',
                        help='Display detailed information')
    
    args = parser.parse_args()
    
    # Run main workflow
    main(
        n_stores=args.n_stores,
        n_products=args.n_products,
        n_weeks=args.n_weeks,
        start_date=args.start_date,
        forecast_horizon=args.forecast_horizon,
        model_type=args.model_type,
        planning_horizon=args.planning_horizon,
        service_level=args.service_level,
        lead_time=args.lead_time,
        max_payload=args.max_payload,
        n_customers=args.n_customers,
        save=args.save,
        verbose=args.verbose
    )
