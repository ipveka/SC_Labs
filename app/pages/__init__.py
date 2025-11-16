"""
Page modules for SC Labs dashboard
"""
from .landing import show_landing_page
from .data_settings import show_data_page
from .forecasting import show_forecast_page
from .inventory import show_inventory_page
from .routing import show_routing_page

__all__ = [
    'show_landing_page',
    'show_data_page',
    'show_forecast_page',
    'show_inventory_page',
    'show_routing_page'
]
