"""
SC Labs - Streamlit App Runner
Launches the Streamlit application for supply chain optimization
"""
import subprocess
import sys
import os
from pathlib import Path


def main():
    """Run the Streamlit application"""
    # Get the app directory
    app_path = Path(__file__).parent / "app" / "app.py"
    
    # Check if app exists
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("Starting SC Labs Streamlit Application...")
    print(f"App location: {app_path}")
    print("-" * 50)
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down Streamlit app...")
        sys.exit(0)


if __name__ == "__main__":
    main()
