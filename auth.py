import streamlit as st
import hashlib
import pickle
import os.path
from pathlib import Path

# User authentication functions
def init_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'users_db_path' not in st.session_state:
        st.session_state.users_db_path = os.path.join(os.path.dirname(__file__), 'users.pkl')

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def save_users(users):
    with open(st.session_state.users_db_path, 'wb') as f:
        pickle.dump(users, f)

def load_users():
    """Load users dictionary from file"""
    if os.path.exists(st.session_state.users_db_path):
        with open(st.session_state.users_db_path, 'rb') as f:
            return pickle.load(f)
    default_users = {
        'admin': {
            'password': hash_password('admin123'),
            'is_admin': True
        }
    }
    save_users(default_users)
    return default_users

def login_user(username, password):
    users = load_users()
    if username in users and users[username]['password'] == hash_password(password):
        st.session_state.authenticated = True
        st.session_state.username = username
        return True
    return False

def register_user(username, password, is_admin=False):
    users = load_users()
    if username in users:
        return False
    users[username] = {
        'password': hash_password(password),
        'is_admin': is_admin
    }
    save_users(users)
    return True

def logout_user():
    st.session_state.authenticated = False
    st.session_state.username = ""

def show_login_page():
    st.markdown("""
    <div class="login-container">
        <div class="login-header">
            <h1>GitHub Activity Analyzer</h1>
            <p>Please log in to continue</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if login_user(username, password):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        with st.expander("New User? Register here"):
            with st.form("register_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register_button = st.form_submit_button("Register")
                
                if register_button:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif not new_username or not new_password:
                        st.error("Username and password cannot be empty")
                    else:
                        if register_user(new_username, new_password):
                            st.success("Registration successful! You can now log in.")
                        else:
                            st.error("Username already exists")