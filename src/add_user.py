#!/usr/bin/env python3
"""
Add User to Supabase Authentication

This script adds a new user to the authentication table in Supabase.
It checks for duplicate usernames and hashes passwords with bcrypt.

Security:
- Passwords are hashed using bcrypt
- Checks for duplicate usernames
- Uses environment variables for credentials

Usage:
    python src/add_user.py
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
from getpass import getpass

# Load environment variables
load_dotenv()

# Configuration from .env
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')
SUPABASE_SCHEMA = os.getenv('SUPABASE_SCHEMA', 'eurofred')
AUTH_TABLE = os.getenv('AUTH_TABLE', 'auth')


def hash_password_bcrypt(password: str) -> str:
    """Hash password using bcrypt (industry standard)"""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def get_supabase_client() -> Client:
    """Create Supabase client"""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
    
    if SUPABASE_URL == "https://your-project.supabase.co":
        raise ValueError("Please update SUPABASE_URL in .env with your actual Supabase project URL")
    
    if SUPABASE_ANON_KEY == "your-anon-key-here":
        raise ValueError("Please update SUPABASE_ANON_KEY in .env with your actual Supabase anon key")
    
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return client


def check_username_exists(supabase: Client, username: str) -> bool:
    """Check if username already exists in database"""
    try:
        response = supabase.schema(SUPABASE_SCHEMA).table(AUTH_TABLE).select("username").eq("username", username).execute()
        return len(response.data) > 0
    except Exception as e:
        print(f"❌ Error checking username: {e}")
        return False


def add_user(supabase: Client, username: str, password: str, name: str, role: str = "user"):
    """
    Add a new user to the authentication table
    
    Args:
        supabase: Supabase client
        username: Unique username
        password: Plain text password (will be hashed)
        name: Full name
        role: User role ('admin' or 'user')
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check for duplicate username
        if check_username_exists(supabase, username):
            print(f"❌ Username '{username}' already exists!")
            return False
        
        # Hash password
        password_hash = hash_password_bcrypt(password)
        
        # Insert user
        user_data = {
            "username": username,
            "password_hash": password_hash,
            "name": name,
            "role": role,
            "active": True
        }
        
        response = supabase.schema(SUPABASE_SCHEMA).table(AUTH_TABLE).insert(user_data).execute()
        
        if response.data:
            print(f"✅ User '{username}' added successfully!")
            print(f"   Name: {name}")
            print(f"   Role: {role}")
            return True
        else:
            print(f"❌ Failed to add user")
            return False
            
    except Exception as e:
        print(f"❌ Error adding user: {e}")
        return False


def main():
    """Main execution"""
    print("=" * 60)
    print("Add User to Supabase Authentication")
    print("=" * 60)
    print()
    
    try:
        # Connect to Supabase
        print("🔌 Connecting to Supabase...")
        supabase = get_supabase_client()
        print(f"✅ Connected to Supabase")
        print(f"   Schema: {SUPABASE_SCHEMA}")
        print(f"   Table: {AUTH_TABLE}")
        print()
        
        # Get user input
        print("Enter user details:")
        print("-" * 60)
        
        username = input("Username: ").strip()
        if not username:
            print("❌ Username cannot be empty")
            return
        
        # Check if username exists
        if check_username_exists(supabase, username):
            print(f"❌ Username '{username}' already exists!")
            return
        
        name = input("Full Name: ").strip()
        if not name:
            print("❌ Name cannot be empty")
            return
        
        # Get password securely
        password = getpass("Password: ")
        if not password:
            print("❌ Password cannot be empty")
            return
        
        password_confirm = getpass("Confirm Password: ")
        if password != password_confirm:
            print("❌ Passwords do not match")
            return
        
        # Get role
        role = input("Role (admin/user) [user]: ").strip().lower() or "user"
        if role not in ["admin", "user"]:
            print("❌ Role must be 'admin' or 'user'")
            return
        
        print()
        print("-" * 60)
        print("Creating user...")
        
        # Add user
        success = add_user(supabase, username, password, name, role)
        
        if success:
            print()
            print("=" * 60)
            print("✅ User created successfully!")
            print("=" * 60)
        else:
            print()
            print("=" * 60)
            print("❌ Failed to create user")
            print("=" * 60)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
