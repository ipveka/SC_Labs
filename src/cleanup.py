"""
SC Labs - Cleanup Script
Remove old output files, cache directories, and logs
"""
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import argparse


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def cleanup_lightning_logs(dry_run=False):
    """Remove lightning_logs directories"""
    project_root = get_project_root()
    removed_count = 0
    
    # Find all lightning_logs directories
    lightning_dirs = list(project_root.rglob('lightning_logs'))
    
    if not lightning_dirs:
        print("✓ No lightning_logs directories found")
        return 0
    
    for log_dir in lightning_dirs:
        if log_dir.is_dir():
            if dry_run:
                print(f"  Would remove: {log_dir.relative_to(project_root)}")
            else:
                try:
                    shutil.rmtree(log_dir)
                    print(f"  ✓ Removed: {log_dir.relative_to(project_root)}")
                    removed_count += 1
                except Exception as e:
                    print(f"  ✗ Error removing {log_dir}: {e}")
    
    return removed_count


def cleanup_pycache(dry_run=False):
    """Remove __pycache__ directories"""
    project_root = get_project_root()
    removed_count = 0
    
    # Find all __pycache__ directories
    cache_dirs = list(project_root.rglob('__pycache__'))
    
    if not cache_dirs:
        print("✓ No __pycache__ directories found")
        return 0
    
    for cache_dir in cache_dirs:
        if cache_dir.is_dir():
            if dry_run:
                print(f"  Would remove: {cache_dir.relative_to(project_root)}")
            else:
                try:
                    shutil.rmtree(cache_dir)
                    print(f"  ✓ Removed: {cache_dir.relative_to(project_root)}")
                    removed_count += 1
                except Exception as e:
                    print(f"  ✗ Error removing {cache_dir}: {e}")
    
    return removed_count


def cleanup_old_outputs(days=30, dry_run=False):
    """Remove output files older than specified days"""
    project_root = get_project_root()
    output_dir = project_root / 'output'
    
    if not output_dir.exists():
        print("✓ No output directory found")
        return 0
    
    cutoff_date = datetime.now() - timedelta(days=days)
    removed_count = 0
    total_size = 0
    
    # Find all CSV files in output directory
    csv_files = list(output_dir.glob('*.csv'))
    
    if not csv_files:
        print("✓ No output files found")
        return 0
    
    for csv_file in csv_files:
        if csv_file.name == '.gitkeep':
            continue
            
        # Get file modification time
        file_time = datetime.fromtimestamp(csv_file.stat().st_mtime)
        
        if file_time < cutoff_date:
            file_size = csv_file.stat().st_size
            if dry_run:
                print(f"  Would remove: {csv_file.name} ({file_size / 1024:.1f} KB, {file_time.strftime('%Y-%m-%d')})")
                total_size += file_size
            else:
                try:
                    csv_file.unlink()
                    print(f"  ✓ Removed: {csv_file.name} ({file_size / 1024:.1f} KB, {file_time.strftime('%Y-%m-%d')})")
                    removed_count += 1
                    total_size += file_size
                except Exception as e:
                    print(f"  ✗ Error removing {csv_file.name}: {e}")
    
    if removed_count > 0 or (dry_run and total_size > 0):
        print(f"  Total space: {total_size / (1024 * 1024):.2f} MB")
    
    return removed_count


def cleanup_pyc_files(dry_run=False):
    """Remove .pyc files"""
    project_root = get_project_root()
    removed_count = 0
    
    # Find all .pyc files
    pyc_files = list(project_root.rglob('*.pyc'))
    
    if not pyc_files:
        print("✓ No .pyc files found")
        return 0
    
    for pyc_file in pyc_files:
        if dry_run:
            print(f"  Would remove: {pyc_file.relative_to(project_root)}")
        else:
            try:
                pyc_file.unlink()
                removed_count += 1
            except Exception as e:
                print(f"  ✗ Error removing {pyc_file}: {e}")
    
    if not dry_run and removed_count > 0:
        print(f"  ✓ Removed {removed_count} .pyc files")
    
    return removed_count


def main():
    parser = argparse.ArgumentParser(
        description='Clean up SC Labs project files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/cleanup.py --all                    # Clean everything
  python src/cleanup.py --logs --cache           # Clean logs and cache only
  python src/cleanup.py --outputs --days 60      # Clean outputs older than 60 days
  python src/cleanup.py --all --dry-run          # Preview what would be removed
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Clean everything')
    parser.add_argument('--logs', action='store_true', help='Remove lightning_logs directories')
    parser.add_argument('--cache', action='store_true', help='Remove __pycache__ directories')
    parser.add_argument('--pyc', action='store_true', help='Remove .pyc files')
    parser.add_argument('--outputs', action='store_true', help='Remove old output files')
    parser.add_argument('--days', type=int, default=30, help='Days to keep output files (default: 30)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without deleting')
    
    args = parser.parse_args()
    
    # If no specific flags, show help
    if not any([args.all, args.logs, args.cache, args.pyc, args.outputs]):
        parser.print_help()
        return
    
    print("=" * 70)
    print("SC Labs - Cleanup Script")
    print("=" * 70)
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be deleted\n")
    
    total_removed = 0
    
    # Lightning logs
    if args.all or args.logs:
        print("\n📁 Cleaning lightning_logs directories...")
        count = cleanup_lightning_logs(dry_run=args.dry_run)
        total_removed += count
        if not args.dry_run and count > 0:
            print(f"   Removed {count} directories")
    
    # Cache directories
    if args.all or args.cache:
        print("\n📁 Cleaning __pycache__ directories...")
        count = cleanup_pycache(dry_run=args.dry_run)
        total_removed += count
        if not args.dry_run and count > 0:
            print(f"   Removed {count} directories")
    
    # .pyc files
    if args.all or args.pyc:
        print("\n📄 Cleaning .pyc files...")
        count = cleanup_pyc_files(dry_run=args.dry_run)
        total_removed += count
    
    # Old output files
    if args.all or args.outputs:
        print(f"\n📊 Cleaning output files older than {args.days} days...")
        count = cleanup_old_outputs(days=args.days, dry_run=args.dry_run)
        total_removed += count
        if not args.dry_run and count > 0:
            print(f"   Removed {count} files")
    
    print("\n" + "=" * 70)
    if args.dry_run:
        print("✓ Dry run complete - no files were deleted")
    else:
        print(f"✓ Cleanup complete - removed {total_removed} items")
    print("=" * 70)


if __name__ == "__main__":
    main()
