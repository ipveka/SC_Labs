"""
SC Labs - Streamlit App Runner
Launches the Streamlit application for supply chain optimization
"""
import subprocess
import sys
import os
from pathlib import Path


def check_auth_setup():
    """Check if auth table exists, offer to create if not"""
    from dotenv import load_dotenv
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    supabase_schema = os.getenv("SUPABASE_SCHEMA", "public")
    auth_table = os.getenv("AUTH_TABLE", "auth")
    
    if not supabase_url or not supabase_key:
        print("⚠️  Warning: Supabase credentials not found in .env")
        print("    The app will not work without authentication setup.")
        print()
        return False
    
    try:
        from supabase import create_client
        client = create_client(supabase_url, supabase_key)
        
        # Try to query the auth table
        result = client.schema(supabase_schema).table(auth_table).select("username").limit(1).execute()
        print(f"✓ Auth table found: {supabase_schema}.{auth_table}")
        return True
        
    except Exception as e:
        print(f"⚠️  Auth table not found: {supabase_schema}.{auth_table}")
        print()
        print("Would you like to create it now? (y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            print()
            print("Running setup script...")
            setup_script = Path(__file__).parent / "src" / "setup_auth.py"
            result = subprocess.run([sys.executable, str(setup_script)])
            return result.returncode == 0
        else:
            print()
            print("⚠️  You can create it later by running:")
            print("    python src/setup_auth.py")
            print()
            return False


def main():
    """Run the Streamlit application"""
    # Get the app directory
    app_path = Path(__file__).parent / "app" / "app.py"
    
    # Check if app exists
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("SC Labs - Streamlit Application")
    print("=" * 60)
    print()
    
    # Check auth setup
    check_auth_setup()
    
    print()
    print("Starting Streamlit server...")
    print(f"App location: {app_path}")
    print("-" * 60)
    
    # Set environment variable to prevent __pycache__
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-B", "-m", "streamlit", "run",
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down Streamlit app...")
        sys.exit(0)


if __name__ == "__main__":
    main()
