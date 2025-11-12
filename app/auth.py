"""
Authentication module for SC Labs Streamlit app

Uses Supabase with custom auth table for authentication.
Passwords are hashed with bcrypt (industry standard).
"""

import streamlit as st
import os
from pathlib import Path
from typing import Optional, Dict
import bcrypt
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env file
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)


class AuthManager:
    """
    Manages user authentication using Supabase.
    
    Uses custom auth table (configured via SUPABASE_SCHEMA.AUTH_TABLE) with bcrypt password hashing.
    Table structure:
    - id: SERIAL PRIMARY KEY
    - username: VARCHAR(50) UNIQUE NOT NULL
    - password_hash: VARCHAR(255) NOT NULL (bcrypt)
    - name: VARCHAR(100) NOT NULL
    - role: VARCHAR(20) NOT NULL DEFAULT 'user'
    - created_at: TIMESTAMP DEFAULT NOW()
    - updated_at: TIMESTAMP DEFAULT NOW()
    - last_login: TIMESTAMP
    - active: BOOLEAN DEFAULT TRUE
    """
    
    def __init__(self):
        """Initialize Supabase authentication manager"""
        self._init_supabase()
        self._load_settings()
    
    def _load_settings(self):
        """Load authentication settings"""
        try:
            allow_signup = st.secrets.get("ALLOW_SIGNUP", "true")
        except:
            allow_signup = os.getenv("ALLOW_SIGNUP", "true")
        
        self.allow_signup = allow_signup.lower() in ['true', '1', 'yes']
    
    def _init_supabase(self):
        """Initialize Supabase client with custom auth table"""
        # Get credentials from environment or secrets
        try:
            supabase_url = st.secrets.get("SUPABASE_URL")
            supabase_key = st.secrets.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_ANON_KEY")
            supabase_schema = st.secrets.get("SUPABASE_SCHEMA", "public")
            auth_table = st.secrets.get("AUTH_TABLE", "auth")
        except:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
            supabase_schema = os.getenv("SUPABASE_SCHEMA", "public")
            auth_table = os.getenv("AUTH_TABLE", "auth")
        
        # Validate credentials
        if not supabase_url or not supabase_key:
            st.error("❌ Supabase credentials not configured")
            st.error("Please set SUPABASE_URL and SUPABASE_KEY in .env file")
            st.info("Run: python src/setup_auth.py to create the auth table")
            st.stop()
        
        if supabase_url == "https://your-project.supabase.co":
            st.error("❌ Please update SUPABASE_URL in .env with your actual Supabase project URL")
            st.stop()
        
        if "your-anon-key" in supabase_key or "your-key" in supabase_key:
            st.error("❌ Please update SUPABASE_KEY in .env with your actual Supabase anon key")
            st.stop()
        
        # Create Supabase client
        try:
            self.supabase: Client = create_client(supabase_url, supabase_key)
            self.supabase_schema = supabase_schema
            self.auth_table = auth_table
            # Connection successful (silent)
        except Exception as e:
            st.error(f"❌ Failed to connect to Supabase: {str(e)}")
            st.error("Check your SUPABASE_URL and SUPABASE_KEY in .env")
            st.stop()
    
    def _hash_password_bcrypt(self, password: str) -> str:
        """Hash password using bcrypt (industry standard)"""
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def _verify_password_bcrypt(self, password: str, hashed: str) -> bool:
        """Verify password against bcrypt hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False
    
    def login(self, username: str, password: str) -> bool:
        """
        Authenticate user with Supabase eurofred.auth table.
        
        Args:
            username: Username
            password: Password (will be verified against bcrypt hash)
            
        Returns:
            True if login successful, False otherwise
        """
        try:
            # Query user from eurofred.auth table
            table = self.supabase.schema(self.supabase_schema).table(self.auth_table)
            response = table.select("*")\
                .eq("username", username)\
                .eq("active", True)\
                .execute()
            
            if not response.data or len(response.data) == 0:
                st.error("❌ Invalid username or password")
                return False
            
            user_data = response.data[0]
            password_hash = user_data.get('password_hash')
            
            # Verify password with bcrypt
            if not password_hash or not self._verify_password_bcrypt(password, password_hash):
                st.error("❌ Invalid username or password")
                return False
            
            # Login successful - create session
            st.session_state['authenticated'] = True
            st.session_state['user'] = {
                'id': user_data.get('id'),
                'username': user_data.get('username'),
                'name': user_data.get('name'),
                'role': user_data.get('role'),
                'email': f"{username}@planner.local"
            }
            
            # Update last_login timestamp
            try:
                from datetime import datetime
                table.update({'last_login': datetime.utcnow().isoformat()})\
                    .eq('username', username)\
                    .execute()
            except Exception as e:
                print(f"Warning: Could not update last_login: {e}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Login failed: {str(e)}")
            return False
    
    def signup(self, username: str, name: str, password: str) -> bool:
        """
        Register new user in Supabase eurofred.auth table.
        
        Args:
            username: Username (must be unique)
            name: User's full name
            password: Password (will be hashed with bcrypt)
            
        Returns:
            True if signup successful, False otherwise
        """
        try:
            # Hash the password with bcrypt
            password_hash = self._hash_password_bcrypt(password)
            
            # Get table reference
            table = self.supabase.schema(self.supabase_schema).table(self.auth_table)
            
            # Check if username already exists
            existing = table.select("username").eq("username", username).execute()
            
            if existing.data and len(existing.data) > 0:
                st.error("❌ Username already exists")
                return False
            
            # Insert new user
            response = table.insert({
                "username": username,
                "name": name,
                "password_hash": password_hash,
                "role": "user",
                "active": True
            }).execute()
            
            if response.data:
                st.success("✅ Account created successfully! You can now login.")
                return True
            else:
                st.error("❌ Failed to create account")
                return False
                
        except Exception as e:
            st.error(f"❌ Signup failed: {str(e)}")
            return False
    
    def logout(self):
        """Logout current user"""
        st.session_state['authenticated'] = False
        st.session_state['user'] = None
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return st.session_state.get('authenticated', False)
    
    def get_user(self) -> Optional[Dict]:
        """Get current user info"""
        return st.session_state.get('user')
    
    def require_auth(self):
        """
        Require authentication to access page.
        Shows login page if not authenticated.
        """
        if not self.is_authenticated():
            self.show_login_page()
            st.stop()
    
    def show_login_page(self):
        """Display login/signup page"""
        # Custom CSS for login page matching app style
        st.markdown("""
        <style>
        /* Hide sidebar and default elements */
        [data-testid="stSidebar"] {
            display: none;
        }
        #MainMenu {
            visibility: hidden;
        }
        footer {
            visibility: hidden;
        }
        
        /* Full page gradient background */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Login container */
        .login-hero {
            text-align: center;
            padding: 3rem 2rem 2rem;
            color: white;
        }
        
        .login-title {
            font-size: 4rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            animation: fadeInDown 0.6s ease-out;
        }
        
        .login-subtitle {
            font-size: 1.3rem;
            font-weight: 300;
            margin-bottom: 1rem;
            opacity: 0.95;
            animation: fadeInUp 0.6s ease-out;
        }
        
        .login-description {
            font-size: 1rem;
            opacity: 0.85;
            max-width: 500px;
            margin: 0 auto 2rem;
            line-height: 1.6;
        }
        
        /* Form container */
        .stTabs {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 450px;
            margin: 0 auto;
        }
        
        /* Tab styling - bigger tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f7fafc;
            border-radius: 10px;
            padding: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 1.1rem;
            color: #4a5568;
            flex: 1;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
        }
        
        /* Input fields */
        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 2px solid #e2e8f0;
            padding: 12px 16px;
            font-size: 1rem;
            transition: all 0.3s;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Buttons */
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s;
            margin-top: 0.5rem;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        /* Info box */
        .stAlert {
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        /* Animations */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Hero section
        st.markdown("""
        <div class="login-hero">
            <div class="login-title" style="color: #2d3748; text-shadow: none;">📦 SC Planner</div>
            <div class="login-subtitle">Supply Chain Optimization Platform</div>
            <div class="login-description">
                Forecast demand • Optimize inventory • Route deliveries
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Center the form - wider tabs
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            if self.allow_signup:
                # Show both login and signup tabs
                tab1, tab2 = st.tabs(["🔐  Login", "📝  Sign Up"])
                
                with tab1:
                    self._show_login_form()
                
                with tab2:
                    self._show_signup_form()
            else:
                # Show only login form (no tabs)
                st.markdown("<br>", unsafe_allow_html=True)
                self._show_login_form()
    
    def _show_login_form(self):
        """Show login form"""
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username", label_visibility="collapsed")
            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
            password = st.text_input("Password", type="password", placeholder="Enter your password", label_visibility="collapsed")
            st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
            submit = st.form_submit_button("🚀 Login", width="stretch")
            
            if submit:
                if not username or not password:
                    st.warning("⚠️ Please enter both username and password")
                elif self.login(username, password):
                    st.success("✅ Login successful!")
                    st.rerun()
    
    def _show_signup_form(self):
        """Show signup form"""
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form("signup_form"):
            username = st.text_input("Username", placeholder="Choose a username", key="signup_username", label_visibility="collapsed")
            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
            name = st.text_input("Full Name", placeholder="Your full name", key="signup_name", label_visibility="collapsed")
            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
            password = st.text_input("Password", type="password", placeholder="Min 6 characters", key="signup_password", label_visibility="collapsed")
            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
            password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", label_visibility="collapsed")
            st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
            submit = st.form_submit_button("✨ Create Account", width="stretch")
            
            if submit:
                if not username or not name or not password:
                    st.warning("⚠️ Please fill in all fields")
                elif password != password_confirm:
                    st.error("❌ Passwords don't match")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                else:
                    if self.signup(username, name, password):
                        st.balloons()


def show_user_menu(auth_manager: AuthManager):
    """
    Display user menu in sidebar.
    
    Args:
        auth_manager: AuthManager instance
    """
    user = auth_manager.get_user()
    
    if user:
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"👤 **{user.get('name', user.get('username', 'User'))}**")
            st.caption(f"@{user.get('username')} • {user.get('role', 'user')}")
            
            if st.button("🚪 Logout", width="stretch"):
                auth_manager.logout()
                st.rerun()

