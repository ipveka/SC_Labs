"""
Generate sample dataset for SC Labs
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from auxiliar.auxiliar import generate_data

# Generate sample data
print("Generating sample dataset...")
data = generate_data(
    n_stores=3,
    n_products=2,
    n_weeks=52,
    start_date='2024-01-01',
    seed=42
)

# Save to data folder
output_path = project_root / 'data' / 'sample_sales_data.csv'
data.to_csv(output_path, index=False)

print(f"✓ Sample dataset saved to: {output_path}")
print(f"  - Records: {len(data)}")
print(f"  - Stores: {data['store'].nunique()}")
print(f"  - Products: {data['product'].nunique()}")
print(f"  - Date range: {data['date'].min()} to {data['date'].max()}")
print("\nColumns:", list(data.columns))
print("\nFirst few rows:")
print(data.head(10))
