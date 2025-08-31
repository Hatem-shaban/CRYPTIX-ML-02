#!/usr/bin/python3.9
"""
WSGI configuration for CRYPTIX-ML on PythonAnywhere

This file exposes the WSGI callable as a module-level variable named ``application``.
For more information on this file, see the PythonAnywhere help pages.
"""

import sys
import os
from pathlib import Path

# Add your project directory to the Python path
# UPDATE THIS PATH to match your actual project location on PythonAnywhere
project_home = '/home/cryptix/CRYPTIX-ML-02'
if project_home not in sys.path:
    sys.path.append(project_home)

# Set up environment variables for your app
# You can either set them here or use a .env file with python-dotenv
# IMPORTANT: Replace these with your actual values or use .env file
os.environ.setdefault('API_KEY', 'your_binance_api_key_here')
os.environ.setdefault('API_SECRET', 'your_binance_secret_here')
os.environ.setdefault('TELEGRAM_BOT_TOKEN', 'your_telegram_bot_token')
os.environ.setdefault('TELEGRAM_CHAT_ID', 'your_telegram_chat_id')
os.environ.setdefault('FLASK_DEBUG', '0')
os.environ.setdefault('FLASK_HOST', '0.0.0.0')

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(project_home) / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, using environment variables")

# Import your Flask application
try:
    from web_bot import app as application
    print("✅ Flask application imported successfully")
except ImportError as e:
    print(f"❌ Error importing Flask app: {e}")
    # Create a simple error application
    from flask import Flask
    application = Flask(__name__)
    
    @application.route('/')
    def error():
        return f"Import Error: {e}", 500

# For PythonAnywhere, the application object must be named 'application'
if __name__ == "__main__":
    # This won't be called on PythonAnywhere, but useful for local testing
    application.run(debug=False)
