"""
SC Labs - Main Script Redirect

This file redirects to the new location: src/main.py

For backward compatibility, this script runs the main module from src directory.
"""

import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    print("⚠️  Note: main.py has been moved to src/main.py")
    print("   Please use: python src/main.py [options]\n")
    print("   Redirecting to src/main.py...\n")
    
    # Run src/main.py with all arguments (with -B to prevent __pycache__)
    src_main = Path(__file__).parent / 'src' / 'main.py'
    args = [sys.executable, '-B', str(src_main)] + sys.argv[1:]
    
    result = subprocess.run(args)
    sys.exit(result.returncode)
