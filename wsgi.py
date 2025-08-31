#!/usr/bin/python3.9
"""
WSGI configuration for CRYPTIX-ML on PythonAnywhere - SIMPLIFIED VERSION

This file exposes the WSGI callable as a module-level variable named ``application``.
"""

import sys
import os

# Add your project directory to the Python path
# IMPORTANT: Update this path to match your actual PythonAnywhere location
project_home = '/home/cryptix/CRYPTIX-ML-02'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Change working directory to project directory
os.chdir(project_home)

# Set basic environment variables if not already set
os.environ.setdefault('FLASK_DEBUG', '0')

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(project_home, '.env'))
except ImportError:
    pass  # python-dotenv not installed, skip

# Import your Flask application
try:
    from web_bot import app as application
    print("✅ CRYPTIX-ML Flask application loaded successfully")
except Exception as e:
    print(f"❌ Error importing Flask app: {e}")
    # Create a simple error app to show what went wrong
    from flask import Flask
    application = Flask(__name__)
    
    @application.route('/')
    def error():
        return f"""
        <h1>CRYPTIX-ML Import Error</h1>
        <p>Error: {str(e)}</p>
        <p>Check the error logs in PythonAnywhere for more details.</p>
        <p>Project path: {project_home}</p>
        <p>Python path: {sys.path}</p>
        """, 500

# For debugging - this won't run on PythonAnywhere but useful for local testing
if __name__ == "__main__":
    application.run(debug=True)
