"""
Clean __pycache__ directories and .pyc files from the project.

Usage:
    python clean_cache.py
"""

import os
import shutil
from pathlib import Path


def clean_pycache(root_dir='.'):
    """
    Remove all __pycache__ directories and .pyc files.
    
    Args:
        root_dir: Root directory to start cleaning from
    """
    root_path = Path(root_dir).resolve()
    removed_dirs = []
    removed_files = []
    
    # Find and remove __pycache__ directories
    for pycache_dir in root_path.rglob('__pycache__'):
        try:
            shutil.rmtree(pycache_dir)
            removed_dirs.append(str(pycache_dir.relative_to(root_path)))
        except Exception as e:
            print(f"Error removing {pycache_dir}: {e}")
    
    # Find and remove .pyc files
    for pyc_file in root_path.rglob('*.pyc'):
        try:
            pyc_file.unlink()
            removed_files.append(str(pyc_file.relative_to(root_path)))
        except Exception as e:
            print(f"Error removing {pyc_file}: {e}")
    
    # Print summary
    print("=" * 60)
    print("CACHE CLEANUP SUMMARY")
    print("=" * 60)
    
    if removed_dirs:
        print(f"\n✓ Removed {len(removed_dirs)} __pycache__ directories:")
        for dir_path in removed_dirs[:10]:  # Show first 10
            print(f"  - {dir_path}")
        if len(removed_dirs) > 10:
            print(f"  ... and {len(removed_dirs) - 10} more")
    else:
        print("\n✓ No __pycache__ directories found")
    
    if removed_files:
        print(f"\n✓ Removed {len(removed_files)} .pyc files:")
        for file_path in removed_files[:10]:  # Show first 10
            print(f"  - {file_path}")
        if len(removed_files) > 10:
            print(f"  ... and {len(removed_files) - 10} more")
    else:
        print("\n✓ No .pyc files found")
    
    print("\n" + "=" * 60)
    print("✅ Cleanup complete!")
    print("\nTo prevent future __pycache__ creation:")
    print("  1. Set environment variable: PYTHONDONTWRITEBYTECODE=1")
    print("  2. Or run Python with -B flag: python -B script.py")
    print("=" * 60)


if __name__ == "__main__":
    clean_pycache()
