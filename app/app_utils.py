"""
Utility functions for SC Labs Streamlit Dashboard
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_sales_trend_chart(data):
    """Create sales trend line chart."""
    fig = px.line(
        data,
        x='date',
        y='sales',
        color='store',
        line_dash='product',
        title='Sales Trends by Store and Product',
        labels={'sales': 'Sales (units)', 'date': 'Date'}
    )
    fig.update_layout(height=400)
    return fig


def create_sales_histogram(data):
    """Create sales distribution histogram."""
    fig = px.histogram(
        data,
        x='sales',
        nbins=30,
        title='Sales Distribution',
        labels={'sales': 'Sales (units)'}
    )
    return fig


def create_sales_boxplot(data):
    """Create sales box plot by store and product."""
    fig = px.box(
        data,
        x='store',
        y='sales',
        color='product',
        title='Sales by Store and Product',
        labels={'sales': 'Sales (units)'}
    )
    return fig


def create_forecast_chart(subset):
    """Create forecast vs actual comparison chart."""
    fig = go.Figure()
    
    # Historical
    train = subset[subset['sample'] == 'train']
    fig.add_trace(go.Scatter(
        x=train['date'],
        y=train['sales'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Actual test
    test = subset[subset['sample'] == 'test']
    fig.add_trace(go.Scatter(
        x=test['date'],
        y=test['sales'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='green')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=test['date'],
        y=test['prediction'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Sales (units)',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_inventory_chart(subset):
    """Create inventory level chart with reorder point."""
    fig = go.Figure()
    
    # Inventory level
    fig.add_trace(go.Scatter(
        x=subset['date'],
        y=subset['inventory'],
        mode='lines+markers',
        name='Inventory',
        line=dict(color='blue', width=2)
    ))
    
    # Reorder point
    fig.add_trace(go.Scatter(
        x=subset['date'],
        y=[subset['reorder_point'].iloc[0]] * len(subset),
        mode='lines',
        name='Reorder Point',
        line=dict(color='red', dash='dash')
    ))
    
    # Safety stock
    fig.add_trace(go.Scatter(
        x=subset['date'],
        y=[subset['safety_stock'].iloc[0]] * len(subset),
        mode='lines',
        name='Safety Stock',
        line=dict(color='orange', dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Units',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_orders_chart(subset):
    """Create orders bar chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=subset['date'],
        y=subset['order'],
        name='Orders',
        marker_color='lightblue'
    ))
    fig.update_layout(
        title='Orders Placed',
        xaxis_title='Date',
        yaxis_title='Units',
        height=300
    )
    return fig


def create_shipments_chart(subset):
    """Create shipments bar chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=subset['date'],
        y=subset['shipment'],
        name='Shipments',
        marker_color='lightgreen'
    ))
    fig.update_layout(
        title='Shipments Received',
        xaxis_title='Date',
        yaxis_title='Units',
        height=300
    )
    return fig


def create_truck_utilization_chart(deliveries, max_payload):
    """Create truck utilization bar chart."""
    truck_loads = deliveries.groupby('truck')['units'].sum().sort_values(ascending=False)
    utilization = (truck_loads / max_payload * 100).round(1)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(truck_loads))),
        y=truck_loads.values,
        name='Load',
        marker_color='lightblue',
        text=utilization.values,
        texttemplate='%{text}%',
        textposition='outside'
    ))
    fig.add_hline(
        y=max_payload,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Max Payload ({max_payload})"
    )
    fig.update_layout(
        xaxis_title='Truck',
        yaxis_title='Load (units)',
        height=350,
        showlegend=False
    )
    return fig


def create_route_distance_chart(routes):
    """Create route distance bar chart."""
    fig = px.bar(
        routes.sort_values('total_distance', ascending=False),
        x='truck',
        y='total_distance',
        title='Route Distances',
        labels={'total_distance': 'Distance', 'truck': 'Truck'},
        color='total_distance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=350, showlegend=False)
    return fig


def create_distance_vs_stops_chart(routes, deliveries):
    """Create distance vs stops scatter plot."""
    stops_per_truck = deliveries.groupby('truck').size()
    route_data = routes.merge(stops_per_truck.rename('stops'), left_on='truck', right_index=True)
    
    fig = px.scatter(
        route_data,
        x='stops',
        y='total_distance',
        title='Distance vs Stops',
        labels={'stops': 'Number of Stops', 'total_distance': 'Distance'},
        trendline='ols',
        color='stops',
        size='stops'
    )
    fig.update_layout(height=350, showlegend=False)
    return fig


def calculate_forecast_metrics(test_forecasts):
    """Calculate forecast accuracy metrics."""
    mae = abs(test_forecasts['sales'] - test_forecasts['prediction']).mean()
    mape = (abs((test_forecasts['sales'] - test_forecasts['prediction']) / test_forecasts['sales']) * 100).mean()
    rmse = np.sqrt(((test_forecasts['sales'] - test_forecasts['prediction']) ** 2).mean())
    return mae, mape, rmse


def calculate_inventory_metrics(test_inventory):
    """Calculate inventory performance metrics."""
    avg_inventory = test_inventory['inventory'].mean()
    total_orders = test_inventory['order'].sum()
    stockouts = (test_inventory['inventory'] < 0).sum()
    fill_rate = (test_inventory['inventory'] >= 0).sum() / len(test_inventory) * 100
    return avg_inventory, total_orders, stockouts, fill_rate


def calculate_routing_metrics(routes, deliveries, max_payload):
    """Calculate routing efficiency metrics."""
    n_trucks = routes['truck'].nunique()
    total_distance = routes['total_distance'].sum()
    avg_utilization = (deliveries.groupby('truck')['units'].sum() / max_payload * 100).mean()
    total_deliveries = len(deliveries)
    return n_trucks, total_distance, avg_utilization, total_deliveries
