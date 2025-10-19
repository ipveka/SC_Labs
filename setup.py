"""
SC Labs - Setup Script
Prepares the environment and installs all required dependencies
"""
import subprocess
import sys
import os
from pathlib import Path


def create_directories():
    """Create necessary directories for the project"""
    print("Creating project directories...")
    
    directories = [
        "output",
        "output/data",
        "output/forecasts",
        "output/inventory",
        "output/routes",
        "lightning_logs",
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {directory}")
        else:
            print(f"  ✓ Exists: {directory}")


def install_requirements():
    """Install Python dependencies from requirements.txt"""
    print("\nInstalling Python dependencies...")
    
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("  ✗ Error: requirements.txt not found")
        return False
    
    try:
        # Upgrade pip first
        print("  → Upgrading pip...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True, capture_output=True)
        
        # Install requirements
        print("  → Installing packages from requirements.txt...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        # Install streamlit for the app
        print("  → Installing Streamlit...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "streamlit>=1.28.0"
        ], check=True)
        
        print("  ✓ All dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error installing dependencies: {e}")
        return False


def verify_installation():
    """Verify that key packages are installed"""
    print("\nVerifying installation...")
    
    required_packages = [
        "pandas",
        "numpy",
        "gluonts",
        "torch",
        "scipy",
        "streamlit"
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT FOUND")
            all_installed = False
    
    return all_installed


def main():
    """Main setup function"""
    print("=" * 60)
    print("SC Labs - Supply Chain Optimization Setup")
    print("=" * 60)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Install requirements
    if not install_requirements():
        print("\n✗ Setup failed during dependency installation")
        sys.exit(1)
    
    # Step 3: Verify installation
    if not verify_installation():
        print("\n✗ Setup completed with errors - some packages missing")
        sys.exit(1)
    
    # Success
    print("\n" + "=" * 60)
    print("✓ Setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run 'python run_app.py' to start the Streamlit app")
    print("  2. Or explore the notebooks in the 'notebooks/' directory")
    print("  3. Check 'README.md' for more information")
    print()


if __name__ == "__main__":
    main()
