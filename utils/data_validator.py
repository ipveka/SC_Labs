"""
Data validation utilities for SC Labs
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def validate_data_quality(data: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data quality and return warnings/recommendations
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with columns: store, product, date, sales
    
    Returns:
    --------
    dict : Validation results with warnings and recommendations
    """
    results = {
        'is_valid': True,
        'warnings': [],
        'recommendations': [],
        'stats': {}
    }
    
    # Check minimum data requirements
    n_periods = len(data['date'].unique())
    results['stats']['n_periods'] = n_periods
    
    if n_periods < 20:
        results['warnings'].append(
            f"⚠️ Only {n_periods} time periods found. Recommend at least 20-30 periods for reliable forecasts."
        )
        results['recommendations'].append(
            "Consider collecting more historical data or using a shorter forecast horizon."
        )
    
    # Check for missing dates (gaps in time series)
    data_sorted = data.sort_values('date')
    date_diffs = data_sorted['date'].diff()
    
    # Detect frequency (most common difference)
    if len(date_diffs) > 1:
        freq_days = date_diffs.mode()[0].days if not date_diffs.mode().empty else 7
        results['stats']['frequency_days'] = freq_days
        
        # Check for gaps larger than expected frequency
        gaps = date_diffs[date_diffs > pd.Timedelta(days=freq_days * 1.5)]
        if len(gaps) > 0:
            results['warnings'].append(
                f"⚠️ Found {len(gaps)} gaps in time series data (missing dates)."
            )
            results['recommendations'].append(
                "Fill missing dates with interpolated values or zero sales for better forecast accuracy."
            )
    
    # Check for zero/negative sales
    zero_sales = (data['sales'] == 0).sum()
    negative_sales = (data['sales'] < 0).sum()
    
    if zero_sales > len(data) * 0.1:  # More than 10% zeros
        results['warnings'].append(
            f"⚠️ {zero_sales} records ({zero_sales/len(data)*100:.1f}%) have zero sales."
        )
        results['recommendations'].append(
            "High number of zero sales may indicate sparse demand or data quality issues."
        )
    
    if negative_sales > 0:
        results['warnings'].append(
            f"❌ {negative_sales} records have negative sales values."
        )
        results['is_valid'] = False
    
    # Check for outliers (sales > 3 std deviations from mean)
    sales_mean = data['sales'].mean()
    sales_std = data['sales'].std()
    outliers = data[np.abs(data['sales'] - sales_mean) > 3 * sales_std]
    
    results['stats']['outliers'] = len(outliers)
    
    if len(outliers) > len(data) * 0.05:  # More than 5% outliers
        results['warnings'].append(
            f"⚠️ {len(outliers)} potential outliers detected ({len(outliers)/len(data)*100:.1f}%)."
        )
        results['recommendations'].append(
            "Review extreme values - they may be data errors or special events (promotions, holidays)."
        )
    
    # Check data variability
    cv = sales_std / sales_mean if sales_mean > 0 else 0  # Coefficient of variation
    results['stats']['coefficient_of_variation'] = cv
    
    if cv > 1.0:
        results['warnings'].append(
            f"⚠️ High sales variability detected (CV: {cv:.2f})."
        )
        results['recommendations'].append(
            "High variability may require higher safety stock levels or shorter forecast horizons."
        )
    
    # Check for multiple time series
    n_series = len(data.groupby(['store', 'product']))
    results['stats']['n_time_series'] = n_series
    
    # Check if each time series has enough data
    series_counts = data.groupby(['store', 'product']).size()
    short_series = series_counts[series_counts < 10]
    
    if len(short_series) > 0:
        results['warnings'].append(
            f"⚠️ {len(short_series)} store-product combinations have fewer than 10 periods."
        )
        results['recommendations'].append(
            "Time series with limited history may produce unreliable forecasts."
        )
    
    # Check date range
    date_range_days = (data['date'].max() - data['date'].min()).days
    results['stats']['date_range_days'] = date_range_days
    results['stats']['date_range_weeks'] = date_range_days // 7
    
    if date_range_days < 90:  # Less than ~3 months
        results['warnings'].append(
            f"⚠️ Short date range: {date_range_days} days ({date_range_days//7} weeks)."
        )
        results['recommendations'].append(
            "Longer historical periods (6+ months) typically improve forecast accuracy."
        )
    
    return results


def get_recommended_forecast_horizon(data: pd.DataFrame) -> int:
    """
    Recommend forecast horizon based on data characteristics
    
    Parameters:
    -----------
    data : pd.DataFrame
        Historical data
    
    Returns:
    --------
    int : Recommended forecast horizon in periods
    """
    n_periods = len(data['date'].unique())
    
    # Rule of thumb: forecast horizon should be ~20-25% of historical data
    # but capped at reasonable limits
    recommended = max(4, min(12, n_periods // 4))
    
    return recommended


def format_validation_report(results: Dict) -> str:
    """
    Format validation results as a readable report
    
    Parameters:
    -----------
    results : dict
        Validation results from validate_data_quality
    
    Returns:
    --------
    str : Formatted report
    """
    report = []
    
    if results['warnings']:
        report.append("**Data Quality Warnings:**")
        for warning in results['warnings']:
            report.append(f"- {warning}")
        report.append("")
    
    if results['recommendations']:
        report.append("**Recommendations:**")
        for rec in results['recommendations']:
            report.append(f"- {rec}")
        report.append("")
    
    if not results['warnings'] and not results['recommendations']:
        report.append("✅ **Data quality looks good!**")
    
    return "\n".join(report)
