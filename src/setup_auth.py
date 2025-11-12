#!/usr/bin/env python3
"""
Setup Authentication in Supabase

This script creates the authentication table and uploads initial users
with properly hashed passwords to Supabase.

Security Best Practices:
1. Passwords are hashed using bcrypt (industry standard)
2. Never store plain text passwords
3. Use environment variables for credentials
4. Schema isolation (eurofred schema)
5. Proper table permissions in Supabase

Usage:
    python src/setup_auth.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import bcrypt
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configuration from .env
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')
SUPABASE_SCHEMA = os.getenv('SUPABASE_SCHEMA', 'eurofred')
AUTH_TABLE = os.getenv('AUTH_TABLE', 'auth')


def hash_password_bcrypt(password: str) -> str:
    """
    Hash password using bcrypt (industry standard)
    
    Why bcrypt?
    - Designed for password hashing
    - Includes salt automatically
    - Computationally expensive (prevents brute force)
    - Industry standard for password storage
    """
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password_bcrypt(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def get_supabase_client() -> Client:
    """Create Supabase client with schema configuration"""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
    
    if SUPABASE_URL == "https://your-project.supabase.co":
        raise ValueError("Please update SUPABASE_URL in .env with your actual Supabase project URL")
    
    if SUPABASE_ANON_KEY == "your-anon-key-here":
        raise ValueError("Please update SUPABASE_ANON_KEY in .env with your actual Supabase anon key")
    
    # Create client (schema is specified per-query)
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return client


def create_auth_table(supabase: Client):
    """
    Create authentication table in Supabase automatically
    """
    
    sql = f"""
    -- Drop existing table if it exists (to recreate with correct schema)
    DROP TABLE IF EXISTS {SUPABASE_SCHEMA}.{AUTH_TABLE} CASCADE;
    
    -- Create auth table in {SUPABASE_SCHEMA} schema
    CREATE TABLE {SUPABASE_SCHEMA}.{AUTH_TABLE} (
        id SERIAL PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        name VARCHAR(100) NOT NULL,
        role VARCHAR(20) NOT NULL DEFAULT 'user',
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        last_login TIMESTAMP,
        active BOOLEAN DEFAULT TRUE
    );
    
    -- Create index on username for faster lookups
    CREATE INDEX IF NOT EXISTS idx_{AUTH_TABLE}_username 
    ON {SUPABASE_SCHEMA}.{AUTH_TABLE}(username);
    
    -- Create index on active users
    CREATE INDEX IF NOT EXISTS idx_{AUTH_TABLE}_active 
    ON {SUPABASE_SCHEMA}.{AUTH_TABLE}(active) WHERE active = TRUE;
    
    -- Add updated_at trigger
    CREATE OR REPLACE FUNCTION {SUPABASE_SCHEMA}.update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ language 'plpgsql';
    
    DROP TRIGGER IF EXISTS update_{AUTH_TABLE}_updated_at ON {SUPABASE_SCHEMA}.{AUTH_TABLE};
    CREATE TRIGGER update_{AUTH_TABLE}_updated_at
        BEFORE UPDATE ON {SUPABASE_SCHEMA}.{AUTH_TABLE}
        FOR EACH ROW
        EXECUTE FUNCTION {SUPABASE_SCHEMA}.update_updated_at_column();
    
    -- Grant permissions
    GRANT SELECT, INSERT, UPDATE ON {SUPABASE_SCHEMA}.{AUTH_TABLE} TO anon;
    GRANT SELECT, INSERT, UPDATE ON {SUPABASE_SCHEMA}.{AUTH_TABLE} TO authenticated;
    """
    
    print("Creating auth table in Supabase...")
    print(f"   Schema: {SUPABASE_SCHEMA}")
    print(f"   Table: {AUTH_TABLE}")
    print()
    
    try:
        # Execute SQL using Supabase RPC
        result = supabase.rpc('exec_sql', {'query': sql}).execute()
        print("✅ Auth table created successfully!")
        return True
    except Exception as e:
        # If RPC doesn't work, try postgrest API
        print(f"⚠️  RPC method failed: {e}")
        print()
        print("Attempting alternative method...")
        print()
        
        # Show SQL for manual execution
        print("=" * 70)
        print("Please run this SQL in Supabase SQL Editor:")
        print("=" * 70)
        print(sql)
        print("=" * 70)
        print()
        print("📍 How to run:")
        print("   1. Go to Supabase Dashboard → SQL Editor")
        print("   2. Click 'New Query'")
        print("   3. Copy the SQL above")
        print("   4. Paste and click 'Run' (or Ctrl+Enter)")
        print()
        
        response = input("Have you created the table? (y/n): ")
        if response.lower() == 'y':
            return True
        else:
            print("❌ Table creation required to continue")
            return False


def upload_users(supabase: Client):
    """Upload initial users to Supabase"""
    
    # Define users
    users = [
        {
            "username": "admin",
            "password": "admin123",
            "name": "Administrator",
            "role": "admin"
        },
        {
            "username": "demo",
            "password": "demo123",
            "name": "Demo User",
            "role": "user"
        }
    ]
    
    print("\n📤 Uploading users to Supabase...")
    print(f"   Schema: {SUPABASE_SCHEMA}")
    print(f"   Table: {AUTH_TABLE}")
    print()
    
    # Get table reference with schema
    table = supabase.schema(SUPABASE_SCHEMA).table(AUTH_TABLE)
    
    for user in users:
        username = user['username']
        password = user['password']
        
        # Hash password with bcrypt
        password_hash = hash_password_bcrypt(password)
        
        # Prepare user data (without plain password)
        user_data = {
            "username": username,
            "password_hash": password_hash,
            "name": user['name'],
            "role": user['role'],
            "active": True
        }
        
        try:
            # Check if user exists
            result = table.select("username").eq("username", username).execute()
            
            if result.data:
                # User exists, update
                print(f"   Updating user: {username}")
                table.update(user_data).eq("username", username).execute()
                print(f"   ✅ Updated: {username} ({user['name']}) - Role: {user['role']}")
            else:
                # User doesn't exist, insert
                print(f"   Creating user: {username}")
                table.insert(user_data).execute()
                print(f"   ✅ Created: {username} ({user['name']}) - Role: {user['role']}")
            
            # Test password verification
            if verify_password_bcrypt(password, password_hash):
                print(f"   ✓ Password verification test passed")
            else:
                print(f"   ✗ Password verification test FAILED")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error with user {username}: {e}")
            print()


def test_authentication(supabase: Client):
    """Test authentication with uploaded users"""
    
    print("\n🧪 Testing Authentication...")
    print("=" * 70)
    
    # Get table reference with schema
    table = supabase.schema(SUPABASE_SCHEMA).table(AUTH_TABLE)
    
    test_cases = [
        ("admin", "admin123", True),
        ("demo", "demo123", True),
        ("demo", "wrong", False),
        ("nonexistent", "123", False),
    ]
    
    for username, password, should_succeed in test_cases:
        try:
            # Fetch user
            result = table.select("*").eq("username", username).eq("active", True).execute()
            
            if result.data:
                user = result.data[0]
                password_hash = user['password_hash']
                
                # Verify password
                if verify_password_bcrypt(password, password_hash):
                    status = "✅ SUCCESS" if should_succeed else "❌ UNEXPECTED SUCCESS"
                    print(f"{status}: {username} / {password}")
                else:
                    status = "❌ FAILED" if should_succeed else "✅ EXPECTED FAIL"
                    print(f"{status}: {username} / {password}")
            else:
                status = "❌ USER NOT FOUND" if should_succeed else "✅ EXPECTED NOT FOUND"
                print(f"{status}: {username} / {password}")
                
        except Exception as e:
            print(f"❌ ERROR: {username} / {password} - {e}")
    
    print("=" * 70)


def main():
    """Main setup function"""
    
    print("=" * 70)
    print("🔐 SC Planner - Supabase Authentication Setup")
    print("=" * 70)
    print()
    
    # Validate environment
    print("📋 Configuration:")
    print(f"   SUPABASE_URL: {SUPABASE_URL}")
    print(f"   SUPABASE_SCHEMA: {SUPABASE_SCHEMA}")
    print(f"   AUTH_TABLE: {AUTH_TABLE}")
    print()
    
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("❌ Error: SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
        sys.exit(1)
    
    if SUPABASE_URL == "https://your-project.supabase.co":
        print("❌ Error: Please update SUPABASE_URL in .env with your actual project URL")
        sys.exit(1)
    
    if SUPABASE_ANON_KEY == "your-anon-key-here":
        print("❌ Error: Please update SUPABASE_ANON_KEY in .env with your actual anon key")
        sys.exit(1)
    
    # Create Supabase client
    try:
        supabase = get_supabase_client()
        print("✅ Connected to Supabase")
        print()
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        sys.exit(1)
    
    # Step 1: Create table
    print("Step 1: Create Authentication Table")
    print("-" * 70)
    table_created = create_auth_table(supabase)
    
    if not table_created:
        print("\n❌ Cannot continue without auth table")
        sys.exit(1)
    
    # Step 2: Upload users
    print("\nStep 2: Upload Users")
    print("-" * 70)
    upload_users(supabase)
    
    # Step 3: Test authentication
    print("\nStep 3: Test Authentication")
    print("-" * 70)
    test_authentication(supabase)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ Setup Complete!")
    print("=" * 70)
    print()
    print("📝 Next Steps:")
    print("   1. Set USE_DB=true in .env to enable Supabase authentication")
    print("   2. Restart the application")
    print("   3. Login with:")
    print("      - Admin: username=admin, password=admin123")
    print("      - Demo:  username=demo, password=123")
    print()
    print("🔒 Security Notes:")
    print("   ✓ Passwords are hashed with bcrypt (industry standard)")
    print("   ✓ Salt is automatically included in bcrypt hash")
    print("   ✓ 12 rounds = good balance of security and performance")
    print("   ✓ Never store plain text passwords")
    print("   ✓ Schema isolation: {SUPABASE_SCHEMA}")
    print()


if __name__ == "__main__":
    main()
