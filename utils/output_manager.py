"""
Output Manager - Centralized output file handling for SC Labs

This module provides consistent file naming and saving across the application.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd


class OutputManager:
    """
    Manages output file saving with consistent naming and structure.
    
    File naming convention: {module}_{timestamp}.csv
    Example: forecasts_20241112_143022.csv
    """
    
    def __init__(self, output_dir: str = 'output'):
        """
        Initialize the OutputManager.
        
        Args:
            output_dir: Directory to save output files (default: 'output')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def save_forecasts(self, df: pd.DataFrame, custom_name: Optional[str] = None) -> Path:
        """
        Save forecast data to CSV.
        
        Args:
            df: Forecast DataFrame
            custom_name: Optional custom filename (without extension)
            
        Returns:
            Path to saved file
        """
        filename = f"{custom_name or 'forecasts'}_{self.timestamp}.csv"
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        return filepath
    
    def save_inventory(self, df: pd.DataFrame, custom_name: Optional[str] = None) -> Path:
        """
        Save inventory plan data to CSV.
        
        Args:
            df: Inventory plan DataFrame
            custom_name: Optional custom filename (without extension)
            
        Returns:
            Path to saved file
        """
        filename = f"{custom_name or 'inventory_plan'}_{self.timestamp}.csv"
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        return filepath
    
    def save_deliveries(self, df: pd.DataFrame, custom_name: Optional[str] = None) -> Path:
        """
        Save delivery data to CSV.
        
        Args:
            df: Deliveries DataFrame
            custom_name: Optional custom filename (without extension)
            
        Returns:
            Path to saved file
        """
        filename = f"{custom_name or 'deliveries'}_{self.timestamp}.csv"
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        return filepath
    
    def save_routes(self, df: pd.DataFrame, custom_name: Optional[str] = None) -> Path:
        """
        Save route summary data to CSV.
        
        Args:
            df: Routes DataFrame
            custom_name: Optional custom filename (without extension)
            
        Returns:
            Path to saved file
        """
        filename = f"{custom_name or 'routes'}_{self.timestamp}.csv"
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        return filepath
    
    def save_all(
        self,
        forecasts: pd.DataFrame,
        inventory_plan: pd.DataFrame,
        deliveries: pd.DataFrame,
        routes: pd.DataFrame
    ) -> dict:
        """
        Save all output DataFrames at once.
        
        Args:
            forecasts: Forecast DataFrame
            inventory_plan: Inventory plan DataFrame
            deliveries: Deliveries DataFrame
            routes: Routes DataFrame
            
        Returns:
            Dictionary mapping output type to file path
        """
        saved_files = {
            'forecasts': self.save_forecasts(forecasts),
            'inventory_plan': self.save_inventory(inventory_plan),
            'deliveries': self.save_deliveries(deliveries),
            'routes': self.save_routes(routes)
        }
        return saved_files
    
    def get_output_dir(self) -> Path:
        """Get the output directory path."""
        return self.output_dir
    
    def get_timestamp(self) -> str:
        """Get the current timestamp string."""
        return self.timestamp
