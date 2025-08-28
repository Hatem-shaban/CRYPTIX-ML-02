from flask import Flask, render_template_string, jsonify, redirect, send_file
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import config  # Import trading configuration
import os, time, threading, subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from textblob import TextBlob
import requests  # Added for Coinbase API calls
import pytz
import csv
from pathlib import Path
import io
import zipfile
# from keep_alive import keep_alive  # Disabled to avoid Flask conflicts
import sys
import json
from datetime import datetime, timedelta

# Import enhanced trading modules
try:
    from position_manager import get_position_manager
    from signal_filter import get_signal_filter
    from risk_manager import get_risk_manager
    from market_intelligence import get_market_intelligence
    from ml_predictor import EnhancedMLPredictor
    
    ENHANCED_MODULES_AVAILABLE = True
    print("✅ Enhanced trading modules loaded successfully")
    print("🧠 ML Intelligence and Market Analytics enabled")
    
    # Initialize enhanced modules
    position_manager = get_position_manager()
    signal_filter = get_signal_filter()
    risk_manager = get_risk_manager()
    market_intelligence = get_market_intelligence(lookback_days=config.ML_LOOKBACK_DAYS)
    ml_predictor = EnhancedMLPredictor()
    
except ImportError as e:
    print(f"⚠️ Enhanced modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False
    position_manager = None
    signal_filter = None
    risk_manager = None
    market_intelligence = None
    ml_predictor = None

# Import Telegram notifications
try:
    from telegram_notify import (
        notify_signal, notify_trade, notify_error, notify_bot_status, 
        notify_daily_summary, notify_market_update, process_queued_notifications,
        get_telegram_stats, telegram_notifier
    )
    TELEGRAM_AVAILABLE = True
    print("✅ Telegram notifications module loaded successfully")
except ImportError as e:
    print(f"⚠️ Telegram notifications not available: {e}")
    TELEGRAM_AVAILABLE = False
    # Create dummy functions to prevent errors
    def notify_signal(*args, **kwargs): return False
    def notify_trade(*args, **kwargs): return False
    def notify_error(*args, **kwargs): return False
    def notify_bot_status(*args, **kwargs): return False
    def notify_daily_summary(*args, **kwargs): return False
    def notify_market_update(*args, **kwargs): return False
    def process_queued_notifications(): pass
    def get_telegram_stats(): return {}
    telegram_notifier = None

# Install psutil if not present
try:
    import psutil
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

# Watchdog and auto-restart functionality has been removed

# Load environment variables
load_dotenv()

# Verbosity helper (set VERBOSE_LOGS=1 to enable extra logs)
def _verbose() -> bool:
    try:
        return str(os.getenv("VERBOSE_LOGS", "")).strip().lower() in {"1", "true", "yes", "on", "debug"}
    except Exception:
        return False

# Cairo timezone
CAIRO_TZ = pytz.timezone('Africa/Cairo')

def get_cairo_time():
    """Get current time in Cairo, Egypt timezone"""
    return datetime.now(CAIRO_TZ)

def format_cairo_time(dt=None):
    """Format datetime to Cairo timezone string"""
    if dt is None:
        dt = get_cairo_time()
    elif dt.tzinfo is None:
        # If naive datetime, assume it's UTC and convert to Cairo
        dt = pytz.UTC.localize(dt).astimezone(CAIRO_TZ)
    elif dt.tzinfo != CAIRO_TZ:
        # Convert to Cairo timezone
        dt = dt.astimezone(CAIRO_TZ)
    
    return dt.strftime('%Y-%m-%d %H:%M:%S %Z')

def get_time_remaining_for_next_signal():
    """Calculate time remaining until next signal in a human-readable format"""
    try:
        if not bot_status.get('next_signal_time') or not bot_status.get('running'):
            return "Not scheduled"
        
        next_signal = bot_status['next_signal_time']
        current_time = get_cairo_time()
        
        # If next_signal is naive datetime, make it timezone-aware
        if next_signal.tzinfo is None:
            next_signal = CAIRO_TZ.localize(next_signal)
        
        time_diff = next_signal - current_time
        
        if time_diff.total_seconds() <= 0:
            return "Signal due now"
        
        # Convert to minutes and seconds
        total_seconds = int(time_diff.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except Exception as e:
        return "Unknown"

# CSV Trade History Logging
def setup_csv_logging():
    """Initialize CSV logging directories and files while preserving existing data"""
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Define CSV file paths
    csv_files = {
        'trades': logs_dir / 'trade_history.csv',
        'signals': logs_dir / 'signal_history.csv',
        'performance': logs_dir / 'daily_performance.csv',
        'errors': logs_dir / 'error_log.csv'
    }
    
    # Define headers for each file type
    trade_headers = [
        'timestamp', 'cairo_time', 'signal', 'symbol', 'quantity', 'price', 
        'value', 'fee', 'status', 'order_id', 'rsi', 'macd_trend', 'sentiment',
        'balance_before', 'balance_after', 'profit_loss'
    ]
    
    signal_headers = [
        'timestamp', 'cairo_time', 'signal', 'symbol', 'price', 'rsi', 'macd', 'macd_trend',
        'sentiment', 'sma5', 'sma20', 'reason'
    ]
    
    performance_headers = [
        'date', 'total_trades', 'successful_trades', 'failed_trades', 'win_rate',
        'total_revenue', 'daily_pnl', 'total_volume', 'max_drawdown'
    ]
    
    error_headers = [
        'timestamp', 'cairo_time', 'error_type', 'error_message', 'function_name',
        'severity', 'bot_status'
    ]
    
    headers_map = {
        'trades': trade_headers,
        'signals': signal_headers,
        'performance': performance_headers,
        'errors': error_headers
    }
    
    # Initialize CSV files while preserving existing data
    for file_type, file_path in csv_files.items():
        if not file_path.exists():
            # Create new file with headers if it doesn't exist
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers_map[file_type])
        else:
            # File exists - verify headers
            try:
                with open(file_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    existing_headers = next(reader, None)
                    
                    # If file is empty or headers don't match, initialize with headers while preserving data
                    if not existing_headers or existing_headers != headers_map[file_type]:
                        # Read existing data
                        f.seek(0)
                        existing_data = list(reader)
                        
                        # Rewrite file with correct headers and preserved data
                        with open(file_path, 'w', newline='', encoding='utf-8') as f_write:
                            writer = csv.writer(f_write)
                            writer.writerow(headers_map[file_type])
                            writer.writerows(existing_data)
            except Exception as e:
                print(f"Error verifying {file_type} log file: {e}")
                # If there's an error, backup the existing file and create a new one
                backup_path = file_path.with_suffix('.csv.bak')
                try:
                    if file_path.exists():
                        file_path.rename(backup_path)
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers_map[file_type])
                except Exception as be:
                    print(f"Error creating backup of {file_type} log file: {be}")
    
    return csv_files

def log_trade_to_csv(trade_info, additional_data=None):
    """Log trade information to CSV file"""
    try:
        csv_files = setup_csv_logging()
        
        # Prepare trade data
        trade_data = [
            trade_info.get('timestamp', ''),
            format_cairo_time(),
            trade_info.get('signal', ''),
            trade_info.get('symbol', ''),
            trade_info.get('quantity', 0),
            trade_info.get('price', 0),
            trade_info.get('value', 0),
            trade_info.get('fee', 0),
            trade_info.get('status', ''),
            trade_info.get('order_id', ''),
            additional_data.get('rsi', 0) if additional_data else 0,
            additional_data.get('macd_trend', '') if additional_data else '',
            additional_data.get('sentiment', '') if additional_data else '',
            additional_data.get('balance_before', 0) if additional_data else 0,
            additional_data.get('balance_after', 0) if additional_data else 0,
            additional_data.get('profit_loss', 0) if additional_data else 0
        ]
        
        # Write to CSV with most recent at top
        import tempfile
        temp_file = csv_files['trades'].with_suffix('.tmp')
        
        # Read existing data
        existing_data = []
        if csv_files['trades'].exists():
            with open(csv_files['trades'], 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_data = list(reader)
        
        # Write new data at top
        with open(temp_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if existing_data and existing_data[0]:  # Write header if exists
                writer.writerow(existing_data[0])
                writer.writerow(trade_data)  # New entry at top
                writer.writerows(existing_data[1:])  # Rest of data
            else:
                writer.writerow(trade_data)
        
        # Replace original file
        temp_file.replace(csv_files['trades'])
            
        print(f"Trade logged to CSV: {trade_info.get('signal', 'UNKNOWN')} at {trade_info.get('price', 0)}")
        
    except Exception as e:
        print(f"Error logging trade to CSV: {e}")

# Global signal tracking to prevent duplicates
last_signals = {}
last_signal_time = None  # Track when ANY signal was last generated globally

def log_signal_to_csv(signal, price, indicators, reason=""):
    """Log trading signal to CSV file with enhanced duplicate prevention"""
    global last_signals, last_signal_time
    try:
        symbol = indicators.get('symbol', 'UNKNOWN')
        current_time = datetime.now()
        
        print(f"🔍 Attempting to log signal: {signal} for {symbol} at ${price:.4f}")  # Debug

        # Reduce noise: only log HOLD if it's a transition or reason is important
        try:
            last_logged = bot_status.get('last_logged_signal', {}).get(symbol)
            important = signal in ("BUY", "SELL") or (isinstance(reason, str) and ("blocked" in reason.lower() or "error" in reason.lower()))
            if signal == "HOLD" and last_logged == "HOLD" and not important:
                print(f"ℹ️ Skipping HOLD log for {symbol} (no transition)")
                return
            bot_status.setdefault('last_logged_signal', {})[symbol] = signal
        except Exception:
            pass
        
        # GLOBAL rate limiting - prevent ANY signal generation too frequently
        if last_signal_time is not None:
            global_time_diff = (current_time - last_signal_time).total_seconds()
            if global_time_diff < 45:  # Minimum 45 seconds between ANY signals globally
                print(f"🛑 GLOBAL rate limit: Any signal suppressed (last signal {global_time_diff:.1f}s ago, need 45s gap)")
                return
        
        # Enhanced duplicate prevention - prevent ANY signal for same symbol within 90 seconds
        # This prevents rapid-fire signal generation regardless of signal type
        symbol_key = f"{symbol}"  # Just symbol, not signal type
        
        if symbol_key in last_signals:
            time_diff = (current_time - last_signals[symbol_key]).total_seconds()
            if time_diff < 90:  # 90 seconds cooldown for ANY signal on this symbol
                print(f"⚠️ Symbol rate limit: {signal} for {symbol} suppressed (last signal {time_diff:.1f}s ago, need 90s gap)")
                return
        
        # Also check for the specific signal type (additional safety)
        signal_key = f"{symbol}_{signal}"
        if signal_key in last_signals:
            time_diff = (current_time - last_signals[signal_key]).total_seconds()
            if time_diff < 180:  # 3 minutes cooldown for same signal type
                print(f"⚠️ Signal type rate limit: {signal} for {symbol} suppressed (same signal {time_diff:.1f}s ago)")
                return
        
        # Update all tracking variables
        last_signal_time = current_time
        last_signals[symbol_key] = current_time
        last_signals[signal_key] = current_time
        
        csv_files = setup_csv_logging()
        
        signal_data = [
            datetime.now().isoformat(),
            format_cairo_time(),
            signal,
            symbol,
            price,
            indicators.get('rsi', 0),
            indicators.get('macd', 0),
            indicators.get('macd_trend', ''),
            indicators.get('sentiment', ''),
            indicators.get('sma5', 0),
            indicators.get('sma20', 0),
            reason
        ]
        
        with open(csv_files['signals'], 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(signal_data)
            
        print(f"✅ Signal logged: {signal} for {symbol} at ${price:.4f} - {reason}")  # Debug confirmation
            
    except Exception as e:
        print(f"❌ Error logging signal to CSV: {e}")
        # Log the error for debugging
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")

def log_daily_performance(date_dt: datetime | None = None):
    """Compute and log daily performance for the given Cairo date.
    If date_dt is None, uses current Cairo date. Avoids duplicate rows for the same date.
    Metrics are computed from logs/trade_history.csv to ensure per-day accuracy.
    """
    try:
        csv_files = setup_csv_logging()

        # Determine which date to log (Cairo date string YYYY-MM-DD)
        day_dt = date_dt or get_cairo_time()
        if day_dt.tzinfo is None:
            day_dt = pytz.UTC.localize(day_dt).astimezone(CAIRO_TZ)
        elif day_dt.tzinfo != CAIRO_TZ:
            day_dt = day_dt.astimezone(CAIRO_TZ)
        day_str = day_dt.strftime('%Y-%m-%d')

        # Check if already logged for this date
        already_logged = False
        if csv_files['performance'].exists():
            try:
                pdf = pd.read_csv(csv_files['performance'])
                if not pdf.empty and 'date' in pdf.columns:
                    already_logged = (pdf['date'].astype(str) == day_str).any()
            except Exception:
                pass
        if already_logged:
            return True

        # Compute metrics from trade history
        total_trades = successful_trades = failed_trades = 0
        win_rate = 0.0
        total_revenue = 0.0
        daily_pnl = 0.0
        total_volume = 0.0
        max_drawdown = 0.0  # Placeholder

        if csv_files['trades'].exists():
            try:
                tdf = pd.read_csv(csv_files['trades'])
                if not tdf.empty:
                    # Ensure columns exist
                    if 'cairo_time' in tdf.columns:
                        # Extract Cairo date portion
                        tdf['cairo_date'] = tdf['cairo_time'].astype(str).str[:10]
                        ddf = tdf[tdf['cairo_date'] == day_str]
                    else:
                        ddf = pd.DataFrame()

                    if not ddf.empty:
                        total_trades = len(ddf)
                        successful_trades = int((ddf.get('status', pd.Series(dtype=str)) == 'success').sum())
                        failed_trades = total_trades - successful_trades
                        # Sum numeric fields safely
                        if 'profit_loss' in ddf.columns:
                            daily_pnl = pd.to_numeric(ddf['profit_loss'], errors='coerce').fillna(0).sum()
                        if 'value' in ddf.columns:
                            total_volume = pd.to_numeric(ddf['value'], errors='coerce').fillna(0).sum()
                        # For total_revenue, reuse daily_pnl as realized result
                        total_revenue = float(daily_pnl)
                        win_rate = (successful_trades / total_trades * 100.0) if total_trades > 0 else 0.0
            except Exception as e:
                print(f"Error computing daily metrics for {day_str}: {e}")

        # Prepare row
        performance_data = [
            day_str,
            total_trades,
            successful_trades,
            failed_trades,
            win_rate,
            total_revenue,
            daily_pnl,
            total_volume,
            max_drawdown
        ]

        # Append row
        with open(csv_files['performance'], 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(performance_data)
        return True
    except Exception as e:
        print(f"Error logging daily performance to CSV: {e}")
        return False

def log_error_to_csv(error_message, error_type="GENERAL", function_name="", severity="ERROR"):
    """Log errors to CSV file"""
    try:
        csv_files = setup_csv_logging()
        
        error_data = [
            datetime.now().isoformat(),
            format_cairo_time(),
            error_type,
            str(error_message),
            function_name,
            severity,
            bot_status.get('running', False)
        ]
        
        # Write to CSV with most recent at top
        import tempfile
        temp_file = csv_files['errors'].with_suffix('.tmp')
        
        # Read existing data
        existing_data = []
        if csv_files['errors'].exists():
            with open(csv_files['errors'], 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_data = list(reader)
        
        # Write new data at top
        with open(temp_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if existing_data and existing_data[0]:  # Write header if exists
                writer.writerow(existing_data[0])
                writer.writerow(error_data)  # New entry at top
                writer.writerows(existing_data[1:])  # Rest of data
            else:
                writer.writerow(error_data)
        
        # Replace original file
        temp_file.replace(csv_files['errors'])
            
        print(f"Error logged to CSV: {error_type} - {error_message}")
        
        # Send Telegram notification for critical errors
        if TELEGRAM_AVAILABLE and severity in ['ERROR', 'CRITICAL']:
            try:
                notify_error(str(error_message), error_type, function_name, severity)
            except Exception as telegram_error:
                print(f"Telegram error notification failed: {telegram_error}")
            
    except Exception as e:
        print(f"Error logging error to CSV: {e}")

def get_csv_trade_history(days=30):
    """Read and return trade history from CSV"""
    try:
        csv_files = setup_csv_logging()
        
        if not csv_files['trades'].exists():
            return []
        
        # Read CSV file
        df = pd.read_csv(csv_files['trades'])
        
        # Filter by date if needed
        if days > 0 and not df.empty:
            try:
                # Handle different timestamp formats
                if 'timestamp' in df.columns:
                    # Try to parse the timestamp column, removing timezone info to avoid warnings
                    df['timestamp'] = pd.to_datetime(df['timestamp'].str.replace(r' [A-Z]{3,4}$', '', regex=True), errors='coerce')
                    
                    # Remove rows where timestamp parsing failed
                    df = df.dropna(subset=['timestamp'])
                    
                    if not df.empty:
                        # Create cutoff date with timezone awareness
                        cutoff_date = get_cairo_time() - pd.Timedelta(days=days)
                        
                        # If timestamps are timezone-naive, make them timezone-aware for comparison
                        if df['timestamp'].dt.tz is None:
                            # Assume UTC if no timezone info
                            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                        
                        # Convert cutoff_date to the same timezone as df timestamps
                        if cutoff_date.tzinfo is None:
                            cutoff_date = CAIRO_TZ.localize(cutoff_date)
                        
                        # Filter by date
                        df = df[df['timestamp'] >= cutoff_date]
                        
            except Exception as date_error:
                log_error_to_csv(f"Date filtering error in CSV read: {date_error}", 
                               "CSV_DATE_ERROR", "get_csv_trade_history", "WARNING")
                # Continue without date filtering if there's an error
        
        # Sort by timestamp to show newest first (similar to Signal History)
        if not df.empty and 'timestamp' in df.columns:
            try:
                # Ensure timestamp column is properly parsed
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                # Sort by timestamp, newest first
                df = df.sort_values('timestamp', ascending=False)
            except Exception as sort_error:
                # Fallback: reverse the order to show newest first
                df = df.iloc[::-1]
        else:
            # Fallback: reverse the order to show newest first
            df = df.iloc[::-1]
        
        # Convert to list of dictionaries
        return df.to_dict('records')
        
    except Exception as e:
        log_error_to_csv(f"Error reading CSV trade history: {e}", 
                       "CSV_READ_ERROR", "get_csv_trade_history", "ERROR")
        return []

# Global bot status
bot_status = {
    'running': False,
    'signal_scanning_active': False,  # Track signal scanning status
    'last_signal': 'UNKNOWN',
    'last_scan_time': None,  # Track when last scan occurred
    'current_symbol': 'BTCUSDT',  # Track currently analyzed symbol
    'last_price': 0,
    'last_update': None,
    'api_connected': False,
    'total_trades': 0,
    'errors': [],
    'start_time': get_cairo_time(),
    'consecutive_errors': 0,
    'rsi': 50,
    'macd': {'macd': 0, 'signal': 0, 'trend': 'NEUTRAL'},
    'sentiment': 'neutral',
    'monitored_pairs': {},  # Track all monitored pairs' status
    'trading_strategy': 'ADAPTIVE',  # Current trading strategy (STRICT, MODERATE, ADAPTIVE)
    'next_signal_time': None,  # Track when next signal will be generated
    'signal_interval': 300,  # Base signal generation interval in seconds (5 minutes - adaptive)
    'market_regime': 'NORMAL',  # Current market regime (QUIET, NORMAL, VOLATILE, EXTREME)
    'hunting_mode': False,  # Aggressive opportunity hunting mode
    'last_volatility_check': None,  # Track when we last checked volatility
    'adaptive_intervals': {
        'QUIET': 600,       # 10 minutes during quiet markets (increased from 30min)
        'NORMAL': 600,      # 10 minutes during normal markets (increased from 5min)
        'VOLATILE': 900,     # 15 minutes during volatile markets (increased from 3min)
        'EXTREME': 600,      # 10 minutes during extreme volatility (increased from 1min)
        'HUNTING': 300       # 5 minutes when hunting opportunities (increased from 30s)
    },
    'trading_summary': {
        'total_revenue': 0.0,
        'successful_trades': 0,
        'failed_trades': 0,
        'total_buy_volume': 0.0,
        'total_sell_volume': 0.0,
        'average_trade_size': 0.0,
        'win_rate': 0.0,
        'trades_history': []  # Last 10 trades for display
    },
    # Caches
    'exchange_info_cache': None,   # {'time': datetime, 'data': {...}}
    'coinbase_cache': {},          # {'BTC-USD': {'time': dt, 'data': {...}}}
    # Logging deduplication
    'last_logged_signal': {}       # per-symbol last logged signal value
}

app = Flask(__name__)

# Initialize CSV logging on startup
setup_csv_logging()

# Initialize API credentials with multiple fallback methods for Render deployment
print("🚀 CRYPTIX Bot Starting...")

api_key = None
api_secret = None
client = None

# Try multiple methods to get environment variables without noisy prints
try:
    # Method 1: os.getenv (standard)
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    # Method 2: Direct os.environ access (backup)
    if not api_key:
        api_key = os.environ.get("API_KEY")
    if not api_secret:
        api_secret = os.environ.get("API_SECRET")
    if _verbose():
        print(f"🔑 Initial credential check:")
        print(f"   API_KEY loaded: {'✓' if api_key else '✗'}")
        print(f"   API_SECRET loaded: {'✓' if api_secret else '✗'}")
        if api_key and api_secret:
            print(f"   API_KEY format: {len(api_key)} chars, preview: {api_key[:8]}...{api_key[-4:]}")
            print("✅ Credentials loaded successfully at startup")
        else:
            print("⚠️  Credentials not found at startup - will retry during initialization")
except Exception as e:
    if _verbose():
        print(f"⚠️  Error loading credentials at startup: {e}")
        print("   Will attempt to load during client initialization")

# Lightweight sentiment analysis function
def get_sentiment_score(text):
    """Enhanced sentiment scoring with crypto-specific keyword weighting"""
    try:
        blob = TextBlob(text)
        base_sentiment = blob.sentiment.polarity
        
        # Crypto-specific keywords for better sentiment analysis
        bullish_keywords = ['moon', 'bullish', 'buy', 'hodl', 'pump', 'rally', 'breakout', 'surge', 'gains', 'profit']
        bearish_keywords = ['dump', 'crash', 'sell', 'bearish', 'drop', 'fall', 'loss', 'decline', 'dip', 'correction']
        
        text_lower = text.lower()
        keyword_boost = 0
        
        # Apply keyword boosting
        for keyword in bullish_keywords:
            if keyword in text_lower:
                keyword_boost += 0.1
                
        for keyword in bearish_keywords:
            if keyword in text_lower:
                keyword_boost -= 0.1
        
        # Combine base sentiment with keyword boost
        enhanced_sentiment = base_sentiment + keyword_boost
        
        # Ensure sentiment stays within bounds [-1, 1]
        return max(-1, min(1, enhanced_sentiment))
    except Exception as e:
        print(f"Sentiment scoring error: {e}")
        return 0

def initialize_client():
    global client, bot_status, api_key, api_secret
    try:
        # Skip if already connected and client exists
        if client and bot_status.get('api_connected', False):
            print("✅ API client already connected")
            return True
            
        # Reload environment variables to ensure we have latest values
        load_dotenv()
        
        # Get API credentials with multiple fallback methods for Render
        api_key = (
            os.getenv("API_KEY") or 
            os.environ.get("API_KEY") or 
            os.getenv("BINANCE_API_KEY") or
            os.environ.get("BINANCE_API_KEY") or
            None
        )
        api_secret = (
            os.getenv("API_SECRET") or 
            os.environ.get("API_SECRET") or 
            os.getenv("BINANCE_API_SECRET") or
            os.environ.get("BINANCE_API_SECRET") or
            None
        )
        
        # Detailed logging for debugging (verbose only)
        if _verbose():
            print(f"🔍 Environment check:")
            print(f"   API_KEY found: {'Yes' if api_key else 'No'}")
            print(f"   API_SECRET found: {'Yes' if api_secret else 'No'}")
            if api_key:
                print(f"   API_KEY length: {len(api_key)}")
                print(f"   API_KEY preview: {api_key[:8]}...{api_key[-4:]}")
        
        if not api_key or not api_secret:
            error_msg = f"API credentials missing - API_KEY: {'✓' if api_key else '✗'}, API_SECRET: {'✓' if api_secret else '✗'}"
            print(f"❌ {error_msg}")
            bot_status['errors'].append(error_msg)
            log_error_to_csv(error_msg, "CREDENTIALS_ERROR", "initialize_client", "ERROR")
            return False
        
        # Determine whether to use Binance Testnet (via env or config)
        def _truthy(v):
            return str(v).strip().lower() in {"1", "true", "yes", "on"}
        env_flag = os.getenv("BINANCE_TESTNET") or os.getenv("USE_TESTNET")
        use_testnet = _truthy(env_flag) if env_flag is not None else getattr(config, 'USE_TESTNET', False)

        # Validate credential format (less strict for testnet); allow variation in lengths on LIVE
        if not use_testnet and len(api_key) < 32:
            error_msg = f"Invalid API key format - too short for LIVE (len={len(api_key)})"
            print(f"❌ {error_msg}")
            bot_status['errors'].append(error_msg)
            log_error_to_csv(error_msg, "CREDENTIALS_ERROR", "initialize_client", "ERROR")
            return False
        if not use_testnet and len(api_secret) < 32:
            error_msg = f"Invalid API secret format - too short for LIVE (len={len(api_secret)})"
            print(f"❌ {error_msg}")
            bot_status['errors'].append(error_msg)
            log_error_to_csv(error_msg, "CREDENTIALS_ERROR", "initialize_client", "ERROR")
            return False
        if use_testnet:
            # Basic sanity check only
            if len(api_key) < 24 or len(api_secret) < 24:
                error_msg = f"Testnet credentials look too short (key {len(api_key)}, secret {len(api_secret)})"
                print(f"❌ {error_msg}")
                bot_status['errors'].append(error_msg)
                log_error_to_csv(error_msg, "CREDENTIALS_ERROR", "initialize_client", "ERROR")
                return False

        print(f"🔗 Initializing Binance client for {'TESTNET' if use_testnet else 'LIVE'} trading...")
        client = Client(api_key, api_secret, testnet=use_testnet)
        # Ensure Spot Testnet base URL when requested
        if use_testnet:
            try:
                client.API_URL = 'https://testnet.binance.vision/api'
            except Exception:
                pass
        try:
            base_url = getattr(client, 'API_URL', None) or getattr(client, 'BASE_URL', None)
            if base_url:
                print(f"   Base URL: {base_url}")
        except Exception:
            pass
        
        # Test API connection with minimal call
        if _verbose():
            print("📊 Testing API connection...")
        server_time = client.get_server_time()
        
        # Only get account info if server connection is successful
        account = client.get_account()
        
        if _verbose():
            print("✅ API connection successful!")
            print(f"   Account Type: {account.get('accountType', 'Unknown')}")
            print(f"   Can Trade: {account.get('canTrade', 'Unknown')}")
            perms = account.get('permissions', [])
            try:
                perms_str = ", ".join(perms)
            except Exception:
                perms_str = str(perms)
            print(f"   Permissions: {perms_str}")
        
        bot_status['api_connected'] = True
        bot_status['account_type'] = account.get('accountType', 'Unknown')
        bot_status['can_trade'] = account.get('canTrade', False)
        
        return True
        
    except BinanceAPIException as e:
        error_msg = f"Binance API Error {e.code}: {e.message}"
        print(f"❌ {error_msg}")
        bot_status['errors'].append(error_msg)
        bot_status['api_connected'] = False
        client = None
        
        # Log specific error solutions
        if e.code == -2015:
            solution_msg = "Error -2015: Check API key/secret format, IP restrictions, or regenerate API key"
            print(f"💡 {solution_msg}")
            log_error_to_csv(f"{error_msg} | {solution_msg}", "API_ERROR", "initialize_client", "ERROR")
        else:
            log_error_to_csv(error_msg, "API_ERROR", "initialize_client", "ERROR")
        
        return False
        
    except Exception as e:
        error_msg = f"Unexpected error initializing client: {str(e)}"
        print(f"❌ {error_msg}")
        bot_status['errors'].append(error_msg)
        bot_status['api_connected'] = False
        client = None
        log_error_to_csv(error_msg, "CLIENT_ERROR", "initialize_client", "ERROR")
        return False

# Market data based sentiment analysis is used instead of social sentiment

def fetch_coinbase_data(product: str = "BTC-USD", ttl_seconds: int = 30):
    """Fetch Coinbase public market data with simple TTL cache and backoff.
    Returns dict with order_book, recent_trades, timestamp or None on error.
    """
    try:
        # TTL cache
        cache = bot_status.get('coinbase_cache') or {}
        entry = cache.get(product)
        now = get_cairo_time()
        if entry and (now - entry['time']).total_seconds() < ttl_seconds:
            return entry['data']

        base_url = "https://api.exchange.coinbase.com"
        headers = {
            'User-Agent': 'CRYPTIX-ML/1.0',
            'Accept': 'application/json'
        }

        if _verbose():
            print(f"Fetching Coinbase order book for {product}...")

        # Helper for GET with backoff
        def get_with_backoff(url, max_retries=3):
            delay = 0.35
            for attempt in range(max_retries):
                try:
                    resp = requests.get(url, headers=headers, timeout=5)
                except Exception as req_err:
                    if attempt == max_retries - 1:
                        raise req_err
                    time.sleep(delay)
                    delay = min(delay * 2, 2.0)
                    continue
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 429:
                    retry_after = resp.headers.get('Retry-After')
                    wait_s = float(retry_after) if retry_after else delay
                    log_error_to_csv("Coinbase rate limit exceeded", "API_RATE_LIMIT", "fetch_coinbase_data", "WARNING")
                    time.sleep(wait_s)
                    delay = min(delay * 2, 2.0)
                    continue
                # Other errors: raise after final attempt
                if attempt == max_retries - 1:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(delay)
                delay = min(delay * 2, 2.0)
            return None

        # Requests
        order_book_resp = get_with_backoff(f"{base_url}/products/{product}/book?level=2")
        order_book = order_book_resp.json() if order_book_resp is not None else None
        if not isinstance(order_book, dict) or 'bids' not in order_book or 'asks' not in order_book:
            log_error_to_csv(f"Invalid Coinbase order book response for {product}", "COINBASE_ERROR", "fetch_coinbase_data", "ERROR")
            return None

        time.sleep(0.2)  # slight pacing between requests
        trades_resp = get_with_backoff(f"{base_url}/products/{product}/trades")
        trades = trades_resp.json() if trades_resp is not None else []

        data = {
            'order_book': order_book,
            'recent_trades': trades,
            'timestamp': datetime.now().timestamp()
        }
        # Save in cache
        cache[product] = {'time': now, 'data': data}
        bot_status['coinbase_cache'] = cache
        return data
    except Exception as e:
        print(f"Coinbase data fetch error: {e}")
        return None

def analyze_market_sentiment():
    """Analyze market sentiment from multiple sources"""
    try:
        # Initialize sentiment components
        order_book_sentiment = 0
        trade_flow_sentiment = 0
        print("\nAnalyzing market sentiment from order book and trade data...")  # Debug log
        
        # 1. Order Book Analysis
        cb_data = fetch_coinbase_data("BTC-USD")
        if cb_data:
            order_book = cb_data['order_book']
            if 'bids' in order_book and 'asks' in order_book:
                # Calculate buy/sell pressure
                bid_volume = sum(float(bid[1]) for bid in order_book['bids'][:10])
                ask_volume = sum(float(ask[1]) for ask in order_book['asks'][:10])
                
                # Normalize order book sentiment
                total_volume = bid_volume + ask_volume
                if total_volume > 0:
                    order_book_sentiment = (bid_volume - ask_volume) / total_volume
        
            # 3. Recent Trade Flow Analysis
            if 'recent_trades' in cb_data:
                recent_trades = cb_data['recent_trades']
                buy_volume = sum(float(trade['size']) for trade in recent_trades if trade['side'] == 'buy')
                sell_volume = sum(float(trade['size']) for trade in recent_trades if trade['side'] == 'sell')
                
                total_trade_volume = buy_volume + sell_volume
                if total_trade_volume > 0:
                    trade_flow_sentiment = (buy_volume - sell_volume) / total_trade_volume
        
        # Market data based sentiment weights
        weights = {
            'order_book': 0.6,  # Order book pressure weight
            'trade_flow': 0.4   # Recent trade flow weight
        }
        
        # Calculate combined sentiment using market data
        combined_sentiment = (
            weights['order_book'] * order_book_sentiment +
            weights['trade_flow'] * trade_flow_sentiment
        )
        
        # Advanced sentiment thresholds with confidence levels
        sentiment_data = {
            'value': combined_sentiment,
            'components': {
                'order_book_sentiment': order_book_sentiment,
                'trade_flow_sentiment': trade_flow_sentiment
            },
            'confidence': min(1.0, abs(combined_sentiment) * 2)  # Confidence score 0-1
        }
        # Determine sentiment with confidence threshold
        if abs(combined_sentiment) < 0.1:
            return "neutral"
        elif combined_sentiment > 0:
            return "bullish" if sentiment_data['confidence'] > 0.5 else "neutral"
        else:
            return "bearish" if sentiment_data['confidence'] > 0.5 else "neutral"
            
    except Exception as e:
        bot_status['errors'].append(f"Market sentiment analysis failed: {e}")
        return "neutral"

def get_exchange_info_cached(ttl_seconds: int = 300):
    """Return Binance exchange_info using a simple TTL cache to reduce API calls."""
    if not client:
        raise RuntimeError("Client not initialized")
    try:
        cache = bot_status.get('exchange_info_cache')
        now = get_cairo_time()
        if cache and (now - cache['time']).total_seconds() < ttl_seconds:
            return cache['data']
        data = client.get_exchange_info()
        bot_status['exchange_info_cache'] = {'time': now, 'data': data}
        return data
    except Exception as e:
        log_error_to_csv(f"exchange_info cache error: {e}", "CACHE_ERROR", "get_exchange_info_cached", "WARNING")
        return client.get_exchange_info()

def calculate_rsi(prices, period=None):
    """Calculate RSI using proper Wilder's smoothing method"""
    period = period or config.RSI_PERIOD
    try:
        # Handle different input types - ensure we have a numpy array of floats
        if hasattr(prices, 'values'):  # pandas Series
            prices = prices.values
        elif isinstance(prices, list):
            prices = np.array(prices)
        elif isinstance(prices, (int, float)):  # Single value
            return 50  # Can't calculate RSI for single value
        
        # Convert to float and handle any string values
        try:
            prices = np.array([float(p) for p in prices])
        except (ValueError, TypeError) as e:
            log_error_to_csv(f"Price conversion error in RSI: {e}, prices type: {type(prices)}", 
                           "DATA_TYPE_ERROR", "calculate_rsi", "ERROR")
            return 50
        
        if len(prices) < period + 1:
            return 50  # Neutral RSI when insufficient data
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Use Wilder's smoothing (similar to EMA) for more accurate RSI
        alpha = 1.0 / period
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Apply Wilder's smoothing to the rest of the data
        for i in range(period, len(gains)):
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Ensure RSI is within bounds
        return max(0, min(100, rsi))
    except Exception as e:
        log_error_to_csv(f"RSI calculation error: {e}", "RSI_ERROR", "calculate_rsi", "ERROR")
        return 50

def calculate_sma(df, period=20):
    """Calculate Simple Moving Average efficiently"""
    try:
        if df is None or len(df) < period:
            return pd.Series([])
        
        # Use pandas rolling for efficiency
        return df['close'].rolling(window=period).mean()
    except Exception as e:
        print(f"SMA calculation error: {e}")
        return pd.Series([])

def calculate_macd(prices, fast=None, slow=None, signal=None):
    """Calculate MACD using configuration parameters"""
    fast = fast or config.MACD_FAST
    slow = slow or config.MACD_SLOW
    signal = signal or config.MACD_SIGNAL
    
    try:
        # Handle different input types - ensure we have a numpy array of floats
        if hasattr(prices, 'values'):  # pandas Series
            prices = prices.values
        elif isinstance(prices, list):
            prices = np.array(prices)
        elif isinstance(prices, (int, float)):  # Single value
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}
        
        # Convert to float and handle any string values
        try:
            prices = np.array([float(p) for p in prices])
        except (ValueError, TypeError) as e:
            log_error_to_csv(f"Price conversion error in MACD: {e}, prices type: {type(prices)}", 
                           "DATA_TYPE_ERROR", "calculate_macd", "ERROR")
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}
        
        if len(prices) < slow:
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}
        
        # Calculate exponential moving averages for more accurate MACD
        def ema(data, period):
            alpha = 2 / (period + 1)
            ema_values = [float(data[0])]  # Start with first value as float
            for price in data[1:]:
                ema_values.append(alpha * float(price) + (1 - alpha) * ema_values[-1])
            return np.array(ema_values)
        
        fast_ema = ema(prices, fast)
        slow_ema = ema(prices, slow)
        
        # MACD line = Fast EMA - Slow EMA
        macd_line = fast_ema - slow_ema
        
        # Signal line = EMA of MACD line
        signal_line = ema(macd_line, signal)
        
        # Histogram = MACD - Signal
        histogram = macd_line - signal_line
        
        # Current values
        current_macd = float(macd_line[-1])
        current_signal = float(signal_line[-1])
        current_histogram = float(histogram[-1])
        
        # Determine trend based on MACD crossover and histogram
        if current_macd > current_signal and current_histogram > 0:
            trend = "BULLISH"
        elif current_macd < current_signal and current_histogram < 0:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        return {
            "macd": round(current_macd, 6),
            "signal": round(current_signal, 6),
            "histogram": round(current_histogram, 6),
            "trend": trend
        }
    except Exception as e:
        log_error_to_csv(f"MACD calculation error: {e}", "MACD_ERROR", "calculate_macd", "ERROR")
        return {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}

def fetch_data(symbol="BTCUSDT", interval="1h", limit=100):
    """Fetch historical price data from Binance."""
    try:
        if _verbose():
            print(f"\n=== Fetching data for {symbol} ===")  # Debug log
        if client:
            if _verbose():
                print("Using Binance client...")  # Debug log
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            if _verbose():
                print(f"Received {len(klines)} candles from Binance")  # Debug log
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                                             'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                                             'taker_buy_quote_asset_volume', 'ignore'])
            
            # Convert numeric columns to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        else:
            error_msg = "Trading client not initialized. Cannot fetch market data."
            log_error_to_csv(error_msg, "CLIENT_ERROR", "fetch_data", "ERROR")
            return None
        
        # Calculate technical indicators
        df['sma5'] = df['close'].rolling(5).mean()
        df['sma20'] = df['close'].rolling(20).mean()

        # EMA family (uses config periods)
        try:
            ema_fast = config.EMA_PERIODS.get('fast', 12)
            ema_slow = config.EMA_PERIODS.get('slow', 26)
            ema_mid = config.EMA_PERIODS.get('mid', 50)
            ema_long = config.EMA_PERIODS.get('long', 200)
        except Exception:
            ema_fast, ema_slow, ema_mid, ema_long = 12, 26, 50, 200
        df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=ema_mid, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=ema_long, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=20).std()
        df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=20).std()
        
        # Calculate RSI with proper error handling
        prices = df['close'].values
        try:
            rsi_value = calculate_rsi(prices)
            if isinstance(rsi_value, (int, float)):
                df['rsi'] = rsi_value  # Single value for entire series
            else:
                df['rsi'] = 50  # Default fallback
        except Exception as rsi_error:
            log_error_to_csv(f"RSI calculation failed for {symbol}: {rsi_error}", 
                           "RSI_ERROR", "fetch_data", "WARNING")
            df['rsi'] = 50
        
        # Calculate MACD with proper error handling
        try:
            macd_data = calculate_macd(prices)
            df['macd'] = macd_data.get('macd', 0)
            df['macd_signal'] = macd_data.get('signal', 0)
            df['macd_histogram'] = macd_data.get('histogram', 0)
            df['macd_trend'] = macd_data.get('trend', 'NEUTRAL')
        except Exception as macd_error:
            log_error_to_csv(f"MACD calculation failed for {symbol}: {macd_error}", 
                           "MACD_ERROR", "fetch_data", "WARNING")
            df['macd'] = 0
            df['macd_signal'] = 0
            df['macd_histogram'] = 0
            df['macd_trend'] = 'NEUTRAL'
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # True Range helpers for ATR and ADX
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(config.ATR_PERIOD).mean()

        # Stochastic Oscillator %K and %D
        try:
            k_period = config.STOCH.get('k_period', 14)
            d_period = config.STOCH.get('d_period', 3)
        except Exception:
            k_period, d_period = 14, 3
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        df['stoch_k'] = np.where(
            (highest_high - lowest_low) > 0,
            (df['close'] - lowest_low) / (highest_high - lowest_low) * 100,
            50
        )
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()

        # VWAP (rolling approximation)
        try:
            vwap_window = config.VWAP.get('window', 20)
        except Exception:
            vwap_window = 20
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        pv = typical_price * df['volume']
        df['vwap'] = pv.rolling(window=vwap_window).sum() / df['volume'].rolling(window=vwap_window).sum()

        # ADX
        try:
            adx_period = config.ADX.get('period', 14)
        except Exception:
            adx_period = 14
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr_smooth = tr.rolling(window=adx_period).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=adx_period).sum() / atr_smooth)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=adx_period).sum() / atr_smooth)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
        df['adx'] = dx.rolling(window=adx_period).mean()
        
    # Volume trend
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_trend'] = df['volume'] / df['volume_sma']
        
        return df
        
    except Exception as e:
        error_msg = f"Error fetching data for {symbol}: {e}"
        log_error_to_csv(error_msg, "DATA_FETCH_ERROR", "fetch_data", "ERROR")
        bot_status['errors'].append(error_msg)
        return None

def detect_market_regime():
    """Professional market regime detection for intelligent timing"""
    try:
        print("\n=== Detecting Market Regime ===")
        
        # Get multi-timeframe data for regime analysis
        btc_1h = fetch_data("BTCUSDT", "1h", 48)  # 48 hours
        btc_5m = fetch_data("BTCUSDT", "5m", 288)  # 24 hours in 5-min candles
        
        if btc_1h is None or btc_5m is None or len(btc_1h) < 24 or len(btc_5m) < 144:
            return 'NORMAL'  # Default regime
        
        # Calculate market volatility measures
        hourly_vol = btc_1h['close'].pct_change().rolling(24).std() * np.sqrt(24 * 365)
        five_min_vol = btc_5m['close'].pct_change().rolling(144).std() * np.sqrt(288 * 365)
        
        current_hourly_vol = hourly_vol.iloc[-1] if not pd.isna(hourly_vol.iloc[-1]) else 0.5
        current_5m_vol = five_min_vol.iloc[-1] if not pd.isna(five_min_vol.iloc[-1]) else 0.5
        
        # Volume surge detection
        avg_volume_1h = btc_1h['volume'].rolling(24).mean().iloc[-1]
        current_volume_1h = btc_1h['volume'].iloc[-1]
        volume_surge = current_volume_1h / avg_volume_1h if avg_volume_1h > 0 else 1
        
        # Price movement analysis
        price_change_1h = abs(btc_1h['close'].pct_change().iloc[-1])
        price_change_24h = abs((btc_1h['close'].iloc[-1] - btc_1h['close'].iloc[-24]) / btc_1h['close'].iloc[-24])
        
        # Market regime classification
        if (current_hourly_vol > 1.5 or current_5m_vol > 2.0 or 
            volume_surge > 3.0 or price_change_1h > 0.05):
            regime = 'EXTREME'
        elif (current_hourly_vol > 0.8 or current_5m_vol > 1.2 or 
              volume_surge > 2.0 or price_change_1h > 0.03):
            regime = 'VOLATILE'
        elif (current_hourly_vol < 0.3 and current_5m_vol < 0.5 and 
              volume_surge < 1.2 and price_change_1h < 0.01):
            regime = 'QUIET'
        else:
            regime = 'NORMAL'
        
        # Store regime data for analytics
        bot_status['market_regime'] = regime
        bot_status['volatility_metrics'] = {
            'hourly_vol': current_hourly_vol,
            'five_min_vol': current_5m_vol,
            'volume_surge': volume_surge,
            'price_change_1h': price_change_1h,
            'price_change_24h': price_change_24h
        }
        
        print(f"Market Regime: {regime}")
        print(f"Hourly Volatility: {current_hourly_vol:.3f}")
        print(f"5min Volatility: {current_5m_vol:.3f}")
        print(f"Volume Surge: {volume_surge:.2f}x")
        print(f"1h Price Change: {price_change_1h:.3f}")
        
        return regime
        
    except Exception as e:
        log_error_to_csv(str(e), "REGIME_DETECTION", "detect_market_regime", "ERROR")
        return 'NORMAL'

def detect_breakout_opportunities():
    """Real-time breakout and momentum opportunity detection with rate limiting"""
    try:
        opportunities = []
        major_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]  # Restored original 5 symbols
        
        # Rate limiting between API calls
        breakout_delay = 0.3  # 300ms delay between fetch calls
        
        for symbol in major_pairs:
            try:
                # Rate limiting before API calls
                time.sleep(breakout_delay)
                
                # Get short-term data for breakout detection (reduced limits)
                df_5m = fetch_data(symbol, "5m", 100)  # Reduced from 144 to 100
                
                time.sleep(breakout_delay)  # Rate limit between calls
                
                df_1m = fetch_data(symbol, "1m", 40)   # Reduced from 60 to 40
                
                if df_5m is None or df_1m is None or len(df_5m) < 40 or len(df_1m) < 20:  # Reduced minimums
                    continue
                
                current_price = df_1m['close'].iloc[-1]
                
                # Bollinger Band breakout detection
                bb_upper = df_5m['bb_upper'].iloc[-1]
                bb_lower = df_5m['bb_lower'].iloc[-1]
                bb_middle = df_5m['bb_middle'].iloc[-1]
                
                # Volume spike detection
                avg_volume = df_5m['volume'].rolling(48).mean().iloc[-1]
                current_volume = df_1m['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # Momentum detection
                momentum_5m = (current_price - df_5m['close'].iloc[-6]) / df_5m['close'].iloc[-6]  # 30min momentum
                momentum_1m = (current_price - df_1m['close'].iloc[-10]) / df_1m['close'].iloc[-10]  # 10min momentum
                
                # RSI divergence detection
                rsi_current = df_5m['rsi'].iloc[-1]
                rsi_prev = df_5m['rsi'].iloc[-12]  # 1 hour ago
                
                opportunity_score = 0
                signals = []
                
                # Breakout signals
                if current_price > bb_upper and volume_ratio > 2.0:
                    opportunity_score += 30
                    signals.append("BB_BREAKOUT_UP")
                elif current_price < bb_lower and volume_ratio > 2.0:
                    opportunity_score += 30
                    signals.append("BB_BREAKOUT_DOWN")
                
                # Momentum signals
                if momentum_5m > 0.02 and momentum_1m > 0.01:
                    opportunity_score += 25
                    signals.append("STRONG_MOMENTUM_UP")
                elif momentum_5m < -0.02 and momentum_1m < -0.01:
                    opportunity_score += 25
                    signals.append("STRONG_MOMENTUM_DOWN")
                
                # Volume surge
                if volume_ratio > 3.0:
                    opportunity_score += 20
                    signals.append("VOLUME_SURGE")
                
                # RSI extremes with volume
                if rsi_current < 25 and volume_ratio > 1.5:
                    opportunity_score += 15
                    signals.append("RSI_OVERSOLD_VOLUME")
                elif rsi_current > 75 and volume_ratio > 1.5:
                    opportunity_score += 15
                    signals.append("RSI_OVERBOUGHT_VOLUME")
                
                if opportunity_score >= 40:  # High opportunity threshold
                    opportunities.append({
                        'symbol': symbol,
                        'score': opportunity_score,
                        'signals': signals,
                        'price': current_price,
                        'volume_ratio': volume_ratio,
                        'momentum_5m': momentum_5m,
                        'momentum_1m': momentum_1m,
                        'rsi': rsi_current,
                        'bb_position': 'ABOVE' if current_price > bb_upper else 'BELOW' if current_price < bb_lower else 'INSIDE'
                    })
                    
            except Exception as e:
                log_error_to_csv(str(e), "BREAKOUT_DETECTION", f"detect_breakout_opportunities_{symbol}", "WARNING")
                continue
        
        # Sort by opportunity score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        if opportunities:
            print(f"\n=== BREAKOUT OPPORTUNITIES DETECTED ===")
            for opp in opportunities[:3]:  # Top 3
                print(f"{opp['symbol']}: Score {opp['score']}, Signals: {', '.join(opp['signals'])}")
        
        return opportunities
        
    except Exception as e:
        log_error_to_csv(str(e), "BREAKOUT_DETECTION", "detect_breakout_opportunities", "ERROR")
        return []

def calculate_smart_interval():
    """Calculate intelligent scanning interval based on market conditions"""
    try:
        # Get current market regime
        current_regime = bot_status.get('market_regime', 'NORMAL')
        base_intervals = bot_status.get('adaptive_intervals', {
            'QUIET': 1800, 'NORMAL': 900, 'VOLATILE': 300, 'EXTREME': 60, 'HUNTING': 30
        })
        
        # Check for hunting mode triggers
        hunting_triggers = 0
        
        # Time-based factors (market opening/closing times)
        current_hour = get_cairo_time().hour
        
        # US market hours (convert to Cairo time: UTC+2)
        us_market_hours = list(range(16, 24)) + list(range(0, 1))  # 2:30 PM - 11 PM Cairo time
        asian_market_hours = list(range(2, 10))  # 2 AM - 10 AM Cairo time
        
        if current_hour in us_market_hours:
            hunting_triggers += 1  # US market active
        if current_hour in asian_market_hours:
            hunting_triggers += 1  # Asian market active
            
        # Check for high volatility events
        volatility_metrics = bot_status.get('volatility_metrics', {})
        if (volatility_metrics.get('volume_surge', 1) > 2.5 or 
            volatility_metrics.get('price_change_1h', 0) > 0.03):
            hunting_triggers += 2
            
        # Check for recent profitable trades (momentum)
        recent_trades = bot_status.get('trading_summary', {}).get('trades_history', [])
        if len(recent_trades) >= 2:
            recent_profitable = sum(1 for trade in recent_trades[-2:] if trade.get('profit_loss', 0) > 0)
            if recent_profitable >= 2:
                hunting_triggers += 1  # Hot streak
                
        # Determine final interval
        if hunting_triggers >= 3 or current_regime == 'EXTREME':
            bot_status['hunting_mode'] = True
            interval = base_intervals.get('HUNTING', 30)
            mode = 'HUNTING'
        else:
            bot_status['hunting_mode'] = False
            interval = base_intervals.get(current_regime, 900)
            mode = current_regime
            
        # Log interval decision
        print(f"\n=== Smart Interval Calculation ===")
        print(f"Market Regime: {current_regime}")
        print(f"Hunting Triggers: {hunting_triggers}")
        print(f"Selected Mode: {mode}")
        print(f"Interval: {interval} seconds ({interval/60:.1f} minutes)")
        
        return interval, mode
        
    except Exception as e:
        log_error_to_csv(str(e), "SMART_INTERVAL", "calculate_smart_interval", "ERROR")
        return 900, 'NORMAL'  # Default fallback

def should_scan_now():
    """Intelligent decision on whether to scan now based on market conditions"""
    try:
        current_time = get_cairo_time()
        
        # Always scan if no previous scan time
        if not bot_status.get('next_signal_time'):
            return True, "Initial scan"
            
        # Check if scheduled time has passed
        if current_time >= bot_status['next_signal_time']:
            return True, "Scheduled scan time reached"
            
        # Override scheduling for extreme conditions
        last_regime_check = bot_status.get('last_volatility_check')
        if (not last_regime_check or 
            (current_time - last_regime_check).total_seconds() > 300):  # Check regime every 5 minutes
            
            regime = detect_market_regime()
            bot_status['last_volatility_check'] = current_time
            
            if regime in ['EXTREME', 'VOLATILE']:
                return True, f"Market regime override: {regime}"
                
        # Check for breakout opportunities in extreme volatility
        if bot_status.get('market_regime') == 'EXTREME':
            opportunities = detect_breakout_opportunities()
            if opportunities:
                return True, f"Breakout opportunity detected: {opportunities[0]['symbol']}"
                
        return False, "Waiting for next scheduled scan"
        
    except Exception as e:
        log_error_to_csv(str(e), "SCAN_DECISION", "should_scan_now", "ERROR")
        return True, "Error in scan decision - defaulting to scan"

# Removed duplicate scan_trading_pairs definition (using the later optimized version)

def analyze_trading_pairs():
    """Analyze all available trading pairs and find the best opportunities"""
    pairs_analysis = []
    default_result = {"symbol": "BTCUSDT", "signal": "HOLD", "score": 0}
    
    try:
        if not client:
            return default_result
        
        try:
            exchange_info = get_exchange_info_cached()
        except Exception as e:
            log_error_to_csv(str(e), "PAIR_ANALYSIS", "analyze_trading_pairs", "ERROR")
            return default_result
        
        # Get all USDT pairs with good volume
        for symbol_info in exchange_info['symbols']:
            # Skip non-USDT or non-trading pairs
            if not (symbol_info['quoteAsset'] == 'USDT' and symbol_info['status'] == 'TRADING'):
                continue
            
            symbol = symbol_info['symbol']
            
            # Get 24hr stats
            try:
                # Get basic market stats
                ticker = client.get_ticker(symbol=symbol)
                volume_usdt = float(ticker['quoteVolume'])
                trades_24h = int(ticker['count'])
                
                # Filter out low volume/activity pairs
                if volume_usdt < 1000000 or trades_24h < 1000:  # Minimum $1M volume and 1000 trades
                    continue
                    
            except Exception as e:
                log_error_to_csv(str(e), "PAIR_ANALYSIS", f"analyze_trading_pairs_{symbol}_stats", "WARNING")
                continue

            try:
                # Get detailed market data
                df = fetch_data(symbol=symbol)
                if df is None or df.empty:
                    continue
                
                # Calculate metrics
                volatility = df['close'].pct_change().std() * np.sqrt(252)
                rsi = calculate_rsi(df['close'].values)
                macd_data = calculate_macd(df['close'].values)
                
                # Get sentiment for major coins
                sentiment = 'neutral'
                if symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
                    sentiment = analyze_market_sentiment()
                
                # Calculate trend metrics
                trend_strength = 0
                trend_score = 0
                if 'sma5' in df.columns and 'sma20' in df.columns:
                    trend_strength = abs(df['sma5'].iloc[-1] - df['sma20'].iloc[-1]) / df['sma20'].iloc[-1]
                    trend_score = 1 if df['sma5'].iloc[-1] > df['sma20'].iloc[-1] else -1
                
                momentum = df['close'].pct_change(5).iloc[-1]
                volume_trend = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                
                # Composite score calculation
                price_potential = 0
                if rsi < 30:  # Oversold
                    price_potential = 1
                elif rsi > 70:  # Overbought
                    price_potential = -1
                    
                momentum_score = momentum * 100  # Convert to percentage
                
                # Calculate final opportunity score
                base_score = (
                    price_potential * 0.3 +  # RSI weight
                    trend_score * 0.3 +      # Trend weight
                    momentum_score * 0.2 +    # Momentum weight
                    (volume_trend - 1) * 0.2  # Volume trend weight
                )
                
                # Apply volatility adjustment if configured
                if config.ADAPTIVE_STRATEGY['volatility_adjustment']:
                    score = base_score * (1 - (volatility/config.MODERATE_STRATEGY['volatility_max']))
                else:
                    score = base_score
                
                # Add sentiment boost for major coins
                if sentiment == 'bullish':
                    score *= 1.2
                elif sentiment == 'bearish':
                    score *= 0.8
                
                # Generate signal based on composite analysis
                signal = "HOLD"
                if score > 0.5:  # Strong bullish signal
                    signal = "BUY"
                elif score < -0.5:  # Strong bearish signal
                    signal = "SELL"
                
                # Store analysis results
                pairs_analysis.append({
                    "symbol": symbol,
                    "signal": signal,
                    "score": score,
                    "volume_usdt": volume_usdt,
                    "volatility": volatility,
                    "rsi": rsi,
                    "trend_strength": trend_strength,
                    "volume_trend": volume_trend,
                    "sentiment": sentiment
                })
            
            except Exception as e:
                log_error_to_csv(str(e), "PAIR_ANALYSIS", f"analyze_trading_pairs_{symbol}_analysis", "WARNING")
                continue
        
        # Sort by absolute score (highest opportunity regardless of buy/sell)
        if pairs_analysis:
            pairs_analysis.sort(key=lambda x: abs(x['score']), reverse=True)
            return pairs_analysis[0]
        
        return {"symbol": "BTCUSDT", "signal": "HOLD", "score": 0}
            
    except Exception as e:
        log_error_to_csv(str(e), "PAIR_ANALYSIS", "analyze_trading_pairs", "ERROR")
        return {"symbol": "BTCUSDT", "signal": "HOLD", "score": 0}

def strict_strategy(df, symbol, indicators):
    """
    Conservative trading strategy with strict entry/exit conditions
    - Requires strong confirmation from multiple indicators
    - Focuses on minimizing risk
    - High threshold for entry/exit points
    """
    if df is None or len(df) < 30:
        return "HOLD", "Insufficient data"
        
    # Extract indicators
    rsi = indicators['rsi']
    macd_trend = indicators['macd_trend']
    sentiment = indicators['sentiment']
    sma5 = indicators['sma5']
    sma20 = indicators['sma20']
    volatility = indicators['volatility']
    current_price = indicators['current_price']
    ema50 = indicators.get('ema50')
    ema200 = indicators.get('ema200')
    stoch_k = indicators.get('stoch_k')
    vwap = indicators.get('vwap')
    adx = indicators.get('adx')
    
    # Get strict strategy thresholds from config
    strict_config = config.STRICT_STRATEGY
    
    # Strict buy conditions with configurable thresholds
    buy_conditions = [
        rsi < config.RSI_OVERSOLD,
        macd_trend == "BULLISH",
        sma5 > sma20,
        sentiment == "bullish",
        volatility < strict_config['volatility_max']
    ]
    # Apply advanced gates
    if strict_config.get('ema_alignment') and (ema50 is not None and ema200 is not None):
        buy_conditions.append(current_price > ema50 > ema200)
    if strict_config.get('adx_min') and adx is not None:
        buy_conditions.append(adx >= strict_config['adx_min'])
    if strict_config.get('stoch_buy_max') and stoch_k is not None:
        buy_conditions.append(stoch_k <= strict_config['stoch_buy_max'])
    if strict_config.get('use_vwap') and vwap is not None:
        buy_conditions.append(current_price >= vwap)

    # Strict sell conditions
    sell_conditions = [
        rsi > config.RSI_OVERBOUGHT,
        macd_trend == "BEARISH",
        sma5 < sma20,
        sentiment == "bearish",
        volatility < strict_config['volatility_max']
    ]
    if strict_config.get('ema_alignment') and (ema50 is not None and ema200 is not None):
        sell_conditions.append(current_price < ema50 < ema200)
    if strict_config.get('adx_min') and adx is not None:
        sell_conditions.append(adx >= strict_config['adx_min'])
    if strict_config.get('stoch_sell_min') and stoch_k is not None:
        sell_conditions.append(stoch_k >= strict_config['stoch_sell_min'])
    if strict_config.get('use_vwap') and vwap is not None:
        sell_conditions.append(current_price <= vwap)
    
    if all(buy_conditions):
        return "BUY", "Strong buy signal with multiple confirmations"
    elif all(sell_conditions):
        return "SELL", "Strong sell signal with multiple confirmations"
    
    return "HOLD", "Waiting for stronger signals"

def moderate_strategy(df, symbol, indicators):
    """
    Balanced trading strategy with moderate entry/exit conditions
    - More frequent trades
    - Balanced risk/reward
    - Moderate thresholds from configuration
    """
    if df is None or len(df) < 30:
        return "HOLD", "Insufficient data"
        
    # Extract indicators
    rsi = indicators['rsi']
    macd_trend = indicators['macd_trend']
    sentiment = indicators['sentiment']
    sma5 = indicators['sma5']
    sma20 = indicators['sma20']
    ema50 = indicators.get('ema50')
    ema200 = indicators.get('ema200')
    stoch_k = indicators.get('stoch_k')
    vwap = indicators.get('vwap')
    adx = indicators.get('adx')
    
    # Get moderate strategy config
    moderate_config = config.MODERATE_STRATEGY
    min_signals = moderate_config['min_signals']
    
    # Buy signals with configurable thresholds
    buy_signals = 0
    if rsi < config.RSI_OVERSOLD + 10: buy_signals += 1  # Less strict RSI
    if macd_trend == "BULLISH": buy_signals += 2
    if sma5 > sma20 and abs(sma5 - sma20)/sma20 > moderate_config['trend_strength']: buy_signals += 1
    if sentiment == "bullish": buy_signals += 1
    if moderate_config.get('ema_alignment') and (ema50 is not None and ema200 is not None) and (indicators['current_price'] > ema50 > ema200):
        buy_signals += 1
    if moderate_config.get('adx_min') and adx is not None and adx >= moderate_config['adx_min']:
        buy_signals += 1
    if moderate_config.get('stoch_buy_max') and stoch_k is not None and stoch_k <= moderate_config['stoch_buy_max']:
        buy_signals += 1
    if moderate_config.get('use_vwap') and vwap is not None and indicators['current_price'] >= vwap:
        buy_signals += 1
    
    # Sell signals (less strict)
    sell_signals = 0
    if rsi > 60: sell_signals += 1  # Less strict RSI
    if macd_trend == "BEARISH": sell_signals += 2
    if sma5 < sma20: sell_signals += 1
    if sentiment == "bearish": sell_signals += 1
    if moderate_config.get('ema_alignment') and (ema50 is not None and ema200 is not None) and (indicators['current_price'] < ema50 < ema200):
        sell_signals += 1
    if moderate_config.get('adx_min') and adx is not None and adx >= moderate_config['adx_min']:
        sell_signals += 1
    if moderate_config.get('stoch_sell_min') and stoch_k is not None and stoch_k >= moderate_config['stoch_sell_min']:
        sell_signals += 1
    if moderate_config.get('use_vwap') and vwap is not None and indicators['current_price'] <= vwap:
        sell_signals += 1
    
    if buy_signals >= 3:
        return "BUY", f"Moderate buy signal ({buy_signals} confirmations)"
    elif sell_signals >= 3:
        return "SELL", f"Moderate sell signal ({sell_signals} confirmations)"
    
    return "HOLD", "Insufficient signals for trade"

def adaptive_strategy(df, symbol, indicators):
    """
    Smart strategy that adapts based on market conditions using configuration parameters
    - Uses volatility and trend strength
    - Adjusts thresholds dynamically based on config
    - Considers market regime with configurable settings
    """
    if df is None or len(df) < 30:
        return "HOLD", "Insufficient data"
        
    # Extract indicators
    rsi = indicators['rsi']
    macd_trend = indicators['macd_trend']
    sentiment = indicators['sentiment']
    volatility = indicators['volatility']
    current_price = indicators['current_price']
    sma5 = indicators['sma5']
    sma20 = indicators['sma20']
    ema50 = indicators.get('ema50')
    ema200 = indicators.get('ema200')
    stoch_k = indicators.get('stoch_k')
    vwap = indicators.get('vwap')
    adx = indicators.get('adx')
    
    # Get adaptive strategy settings
    adaptive_config = config.ADAPTIVE_STRATEGY
    
    # Calculate market regime using config thresholds
    is_high_volatility = volatility > config.MODERATE_STRATEGY['volatility_max']
    trend_strength = abs((sma5 - sma20) / sma20)
    is_strong_trend = trend_strength > config.STRICT_STRATEGY['trend_strength']
    
    # Adjust thresholds based on market conditions and current market regime
    regime = bot_status.get('market_regime', 'NORMAL')
    if is_high_volatility or regime in ['VOLATILE', 'EXTREME']:
        rsi_buy = 35  # More conservative in high volatility
        rsi_sell = 65
        dynamic_threshold = max(25, adaptive_config.get('score_threshold', 30))
    elif regime == 'QUIET':
        rsi_buy = 45  # Harder to trigger in quiet markets
        rsi_sell = 55
        dynamic_threshold = min(35, adaptive_config.get('score_threshold', 30))
    else:
        rsi_buy = 40  # Default
        rsi_sell = 60
        dynamic_threshold = adaptive_config.get('score_threshold', 30)
        
    # Score-based system (0-100) with weights and per-component breakdown
    weights = config.ADAPTIVE_STRATEGY.get('weights', {
        'rsi': 0.2, 'macd': 0.2, 'ema_trend': 0.15, 'stoch': 0.15, 'adx': 0.15, 'vwap': 0.15
    })
    components = {
        'rsi': 0.0,
        'macd': 0.0,
        'ema_trend': 0.0,
        'stoch': 0.0,
        'adx': 0.0,
        'vwap': 0.0
    }

    # RSI (scaled by distance from thresholds)
    rsi_weight = weights.get('rsi', 0.2)
    if rsi < rsi_buy:
        # Normalize by distance to buy threshold
        rsi_norm = (rsi_buy - rsi) / max(1.0, rsi_buy)
        components['rsi'] = 100 * rsi_weight * rsi_norm
    elif rsi > rsi_sell:
        # Normalize by distance to sell threshold
        rsi_norm = (rsi - rsi_sell) / max(1.0, (100.0 - rsi_sell))
        components['rsi'] = -100 * rsi_weight * rsi_norm
    # else stays 0 near neutral band

    # MACD (fixed contribution based on trend)
    macd_w = weights.get('macd', 0.2)
    if macd_trend == 'BULLISH':
        components['macd'] = 100 * macd_w * 0.6
    elif macd_trend == 'BEARISH':
        components['macd'] = -100 * macd_w * 0.6

    # EMA trend (prefer EMA50/200 alignment; fallback to SMA cross)
    ema_w = weights.get('ema_trend', 0.15)
    if ema50 is not None and ema200 is not None:
        if current_price > ema50 > ema200:
            components['ema_trend'] = 100 * ema_w * 0.6
        elif current_price < ema50 < ema200:
            components['ema_trend'] = -100 * ema_w * 0.6
    else:
        if sma5 > sma20:
            components['ema_trend'] = 100 * ema_w * 0.3
        else:
            components['ema_trend'] = -100 * ema_w * 0.3

    # Stochastic (scaled by distance from overbought/oversold)
    stoch_w = weights.get('stoch', 0.15)
    if stoch_k is not None:
        st_oversold = float(config.STOCH.get('oversold', 20))
        st_overbought = float(config.STOCH.get('overbought', 80))
        if stoch_k < st_oversold:
            st_norm = (st_oversold - stoch_k) / max(1.0, st_oversold)
            components['stoch'] = 100 * stoch_w * st_norm
        elif stoch_k > st_overbought:
            st_norm = (stoch_k - st_overbought) / max(1.0, (100.0 - st_overbought))
            components['stoch'] = -100 * stoch_w * st_norm

    # ADX (symmetric: reward strong trend, small penalty for weak trend)
    adx_w = weights.get('adx', 0.15)
    if adx is not None:
        adx_min = float(config.ADAPTIVE_STRATEGY.get('adx_min', 20))
        if adx >= adx_min:
            components['adx'] = 100 * adx_w * 0.5
        else:
            deficit_ratio = max(0.0, (adx_min - adx) / max(1.0, adx_min))
            components['adx'] = -100 * adx_w * 0.2 * deficit_ratio

    # VWAP relation
    vwap_w = weights.get('vwap', 0.15)
    if vwap is not None:
        if current_price >= vwap:
            components['vwap'] = 100 * vwap_w * 0.4
        else:
            components['vwap'] = -100 * vwap_w * 0.4

    # Sum base score and apply regime-based scaling to keep breakdown consistent
    score = sum(components.values())
    if is_high_volatility:
        components = {k: v * 0.8 for k, v in components.items()}
    if is_strong_trend:
        components = {k: v * 1.2 for k, v in components.items()}
    score = sum(components.values())

    # Use dynamic score threshold for decisions with concise breakdown
    score_threshold = dynamic_threshold
    breakdown = (
        f"RSI {components['rsi']:+.1f}, "
        f"MACD {components['macd']:+.1f}, "
        f"EMA {components['ema_trend']:+.1f}, "
        f"Stoch {components['stoch']:+.1f}, "
        f"ADX {components['adx']:+.1f}, "
        f"VWAP {components['vwap']:+.1f}"
    )

    if score >= score_threshold:
        return "BUY", f"Adaptive buy signal (Score: {score:.0f}/{score_threshold}; {breakdown})"
    elif score <= -score_threshold:
        return "SELL", f"Adaptive sell signal (Score: {score:.0f}/{score_threshold}; {breakdown})"
    
    return "HOLD", f"Neutral (Score: {score:.0f}/±{score_threshold}; {breakdown})"

def get_account_balances_summary():
    """Get a summary of all non-zero account balances"""
    try:
        if not client:
            return {"error": "Client not initialized"}
        
        account_info = client.get_account()
        balances = {}
        total_usdt_value = 0
        
        print("\n💰 Account Balance Summary:")
        print("=" * 40)
        
        for balance in account_info['balances']:
            free_balance = float(balance['free'])
            locked_balance = float(balance['locked'])
            total_balance = free_balance + locked_balance
            
            if total_balance > 0:
                asset = balance['asset']
                balances[asset] = {
                    'free': free_balance,
                    'locked': locked_balance,
                    'total': total_balance
                }
                
                # Try to get USDT value for major coins
                usdt_value = 0
                if asset == 'USDT':
                    usdt_value = total_balance
                elif asset in ['BTC', 'ETH', 'BNB']:
                    try:
                        ticker = client.get_ticker(symbol=f"{asset}USDT")
                        price = float(ticker['lastPrice'])
                        usdt_value = total_balance * price
                    except:
                        usdt_value = 0  # Skip if price fetch fails
                
                total_usdt_value += usdt_value
                
                print(f"{asset:>8}: {free_balance:>12.8f} free, {locked_balance:>12.8f} locked "
                      f"(~${usdt_value:>8.2f})")
        
        print("=" * 40)
        print(f"{'TOTAL':>8}: ~${total_usdt_value:>8.2f} USDT value")
        print()
        
        return {
            'balances': balances,
            'total_usdt_value': total_usdt_value,
            'timestamp': format_cairo_time()
        }
        
    except Exception as e:
        error_msg = f"Error getting balance summary: {e}"
        print(f"❌ {error_msg}")
        log_error_to_csv(error_msg, "BALANCE_SUMMARY_ERROR", "get_account_balances_summary", "ERROR")
        return {"error": error_msg}

def check_coin_balance(symbol):
    """Check if we have sufficient balance to place a SELL order for the given symbol"""
    try:
        # Cache to reduce repeated API calls within a short window
        if 'balance_cache' not in bot_status:
            bot_status['balance_cache'] = {}
        cache_entry = bot_status['balance_cache'].get(symbol)
        now = get_cairo_time()
        if cache_entry and (now - cache_entry['time']).total_seconds() < 300:
            return cache_entry['has'], cache_entry['amount'], cache_entry['msg']
        if not client:
            print(f"⚠️ Client not initialized - cannot check balance for {symbol}")
            return False, 0, "Client not initialized"
        
        # Extract base asset from symbol (e.g., "BTC" from "BTCUSDT")
        if symbol.endswith('USDT'):
            base_asset = symbol[:-4]  # Remove "USDT"
        elif symbol.endswith('BUSD'):
            base_asset = symbol[:-4]  # Remove "BUSD" 
        else:
            # For other quote currencies, try to find the quote asset
            try:
                exchange_info = get_exchange_info_cached()
                symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
                if symbol_info:
                    base_asset = symbol_info['baseAsset']
                else:
                    print(f"⚠️ Cannot determine base asset for {symbol}")
                    return False, 0, "Unknown symbol format"
            except Exception as e:
                print(f"⚠️ Error getting symbol info for {symbol}: {e}")
                return False, 0, f"Symbol info error: {e}"
        
        print(f"🔍 Checking {base_asset} balance for potential sell order...")
        
        # Get account balances
        account_info = client.get_account()
        asset_balance = 0
        
        for balance in account_info['balances']:
            if balance['asset'] == base_asset:
                asset_balance = float(balance['free'])
                break
        
        print(f"💰 Available {base_asset} balance: {asset_balance}")
        
        # Get minimum quantity requirements
        min_sellable_qty = 0.001  # Default minimum
        try:
            exchange_info = get_exchange_info_cached()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if symbol_info:
                lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                if lot_size_filter:
                    min_sellable_qty = float(lot_size_filter['minQty'])
        except Exception as e:
            print(f"⚠️ Warning: Could not get minimum quantity for {symbol}: {e}")
        
        # Check if we have enough balance to place a meaningful sell order
        has_sufficient_balance = asset_balance >= min_sellable_qty
        
        print(f"📊 Balance check result:")
        print(f"   Available: {asset_balance} {base_asset}")
        print(f"   Minimum required: {min_sellable_qty} {base_asset}")
        print(f"   Can sell: {'✅ Yes' if has_sufficient_balance else '❌ No'}")
        
        result = (
            (True, asset_balance, f"Sufficient balance: {asset_balance} {base_asset}")
            if has_sufficient_balance
            else (False, asset_balance, f"Insufficient balance: {asset_balance} < {min_sellable_qty} {base_asset}")
        )
        # Save to cache
        bot_status['balance_cache'][symbol] = {
            'has': result[0], 'amount': result[1], 'msg': result[2], 'time': now
        }
        return result
            
    except Exception as e:
        error_msg = f"Error checking balance for {symbol}: {e}"
        print(f"❌ {error_msg}")
        log_error_to_csv(error_msg, "BALANCE_CHECK_ERROR", "check_coin_balance", "ERROR")
        return False, 0, error_msg

def signal_generator(df, symbol="BTCUSDT"):
    print("\n=== Generating Trading Signal ===")  # Debug log
    if df is None or len(df) < 30:
        print(f"Insufficient data for {symbol}")  # Debug log
        signal = "HOLD"
        bot_status.update({
            'last_signal': signal,
            'last_update': format_cairo_time()
        })
        log_signal_to_csv(signal, 0, {"symbol": symbol}, "Insufficient data")
        return signal
    
    # Enhanced risk management checks
    daily_pnl = bot_status['trading_summary'].get('total_revenue', 0)
    consecutive_losses = bot_status.get('consecutive_losses', 0)
    
    # Stop trading if daily loss limit exceeded
    if daily_pnl < -config.MAX_DAILY_LOSS:
        log_signal_to_csv("HOLD", 0, {"symbol": symbol}, f"Daily loss limit exceeded: ${daily_pnl}")
        return "HOLD"
    
    # Reduce activity after consecutive losses
    if consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
        # Still compute a signal but enforce HOLD at the end; don't spam with repeated logs
        risk_locked = True
    else:
        risk_locked = False
    
    sentiment = analyze_market_sentiment()
    
    # Get the latest technical indicators with error handling
    try:
        # Handle RSI - could be a single value or Series
        if 'rsi' in df.columns:
            if hasattr(df['rsi'], 'iloc'):
                rsi = float(df['rsi'].iloc[-1]) if not pd.isna(df['rsi'].iloc[-1]) else 50
            else:
                rsi = float(df['rsi']) if not pd.isna(df['rsi']) else 50
        else:
            rsi = 50
            
        # Handle MACD data
        if 'macd' in df.columns:
            if hasattr(df['macd'], 'iloc'):
                macd = float(df['macd'].iloc[-1]) if not pd.isna(df['macd'].iloc[-1]) else 0
            else:
                macd = float(df['macd']) if not pd.isna(df['macd']) else 0
        else:
            macd = 0
            
        # Handle MACD trend
        if 'macd_trend' in df.columns:
            if hasattr(df['macd_trend'], 'iloc'):
                macd_trend = df['macd_trend'].iloc[-1] if not pd.isna(df['macd_trend'].iloc[-1]) else 'NEUTRAL'
            else:
                macd_trend = df['macd_trend'] if not pd.isna(df['macd_trend']) else 'NEUTRAL'
        else:
            macd_trend = 'NEUTRAL'
            
        # Handle SMAs
        if 'sma5' in df.columns and hasattr(df['sma5'], 'iloc'):
            sma5 = float(df['sma5'].iloc[-1]) if not pd.isna(df['sma5'].iloc[-1]) else 0
        else:
            sma5 = 0
            
        if 'sma20' in df.columns and hasattr(df['sma20'], 'iloc'):
            sma20 = float(df['sma20'].iloc[-1]) if not pd.isna(df['sma20'].iloc[-1]) else 0
        else:
            sma20 = 0
            
        # Handle current price
        if hasattr(df['close'], 'iloc'):
            current_price = float(df['close'].iloc[-1])
        else:
            current_price = float(df['close'])
            
        # Handle volatility
        if 'volatility' in df.columns and hasattr(df['volatility'], 'iloc'):
            volatility = float(df['volatility'].iloc[-1]) if not pd.isna(df['volatility'].iloc[-1]) else 0.5
        else:
            # Calculate basic volatility as fallback
            if hasattr(df['close'], 'pct_change'):
                volatility = float(df['close'].pct_change().std() * np.sqrt(252))
            else:
                volatility = 0.5

        # New indicators
        ema50 = float(df['ema50'].iloc[-1]) if 'ema50' in df.columns and not pd.isna(df['ema50'].iloc[-1]) else None
        ema200 = float(df['ema200'].iloc[-1]) if 'ema200' in df.columns and not pd.isna(df['ema200'].iloc[-1]) else None
        stoch_k = float(df['stoch_k'].iloc[-1]) if 'stoch_k' in df.columns and not pd.isna(df['stoch_k'].iloc[-1]) else None
        stoch_d = float(df['stoch_d'].iloc[-1]) if 'stoch_d' in df.columns and not pd.isna(df['stoch_d'].iloc[-1]) else None
        vwap = float(df['vwap'].iloc[-1]) if 'vwap' in df.columns and not pd.isna(df['vwap'].iloc[-1]) else None
        adx = float(df['adx'].iloc[-1]) if 'adx' in df.columns and not pd.isna(df['adx'].iloc[-1]) else None
                
    except Exception as e:
        log_error_to_csv(f"Error extracting indicators: {str(e)}", "INDICATOR_ERROR", "signal_generator", "ERROR")
        return "HOLD"
    
    # Handle NaN values
    if pd.isna(rsi) or pd.isna(macd) or pd.isna(sma5) or pd.isna(sma20):
        log_signal_to_csv("HOLD", current_price, {'symbol': symbol, 'rsi': rsi, 'macd': macd, 'sentiment': sentiment}, "NaN values detected")
        return "HOLD"
        
    # Prepare indicators dictionary for strategies
    indicators = {
        'symbol': symbol,  # Add symbol to indicators for proper logging
        'rsi': rsi,
        'macd': macd,
        'macd_trend': macd_trend,
        'sentiment': sentiment,
        'sma5': sma5,
        'sma20': sma20,
        'current_price': current_price,
        'volatility': volatility,
        'ema50': ema50,
        'ema200': ema200,
        'stoch_k': stoch_k,
        'stoch_d': stoch_d,
        'vwap': vwap,
        'adx': adx
    }
    
    # Use selected strategy with enhanced error handling
    try:
        strategy = bot_status.get('trading_strategy', 'STRICT')
        print(f"Using strategy: {strategy}")  # Debug log
        
        if strategy == 'STRICT':
            signal, reason = strict_strategy(df, symbol, indicators)
        elif strategy == 'MODERATE':
            signal, reason = moderate_strategy(df, symbol, indicators)
        elif strategy == 'ADAPTIVE':
            signal, reason = adaptive_strategy(df, symbol, indicators)
        else:
            print(f"Unknown strategy {strategy}, defaulting to STRICT")  # Debug log
            signal, reason = strict_strategy(df, symbol, indicators)  # Default to strict
        
        # Enhanced Signal Filtering (if modules available)
        if ENHANCED_MODULES_AVAILABLE and signal != "HOLD":
            print(f"\n🔍 Applying enhanced signal filtering...")
            
            # Prepare market data for filtering
            market_data = {
                'volume_24h': getattr(df, 'volume', pd.Series([1000000])).iloc[-1] if hasattr(df, 'volume') else 1000000,
                'price_change_24h_pct': indicators.get('price_change_24h', 0),
                'volume_ratio': indicators.get('volume_ratio', 1.0),
                'spread_pct': 0.05,  # Default spread
                'volume_consistency': 0.5  # Default consistency
            }
            
            try:
                signal_filter = get_signal_filter()
                filtered_result = signal_filter.filter_and_validate_signal(
                    symbol, signal, indicators, market_data, confidence_threshold=0.65
                )
                
                print(f"   Raw Signal: {signal}")
                print(f"   Filtered Signal: {filtered_result['signal']}")
                print(f"   Confidence: {filtered_result['confidence']:.2f}")
                print(f"   Quality Score: {filtered_result['quality_score']:.2f}")
                print(f"   Filters Passed: {filtered_result['filters_passed']}/7")
                print(f"   Reason: {filtered_result['reason']}")
                
                # Update signal and reason with filtered results
                signal = filtered_result['signal']
                reason = f"Enhanced Filter: {filtered_result['reason']}"
                
                # Store filtering metrics in bot status
                bot_status['last_signal_quality'] = {
                    'confidence': filtered_result['confidence'],
                    'quality_score': filtered_result['quality_score'],
                    'filters_passed': filtered_result['filters_passed']
                }
                
            except Exception as filter_error:
                print(f"   ⚠️ Signal filtering error: {filter_error}")
                # Continue with original signal if filtering fails
        
        # Advanced ML Intelligence Analysis (if enabled and signal is not HOLD)
        if ENHANCED_MODULES_AVAILABLE and signal != "HOLD" and config.ML_ENABLED:
            print(f"\n🧠 Applying ML Intelligence Analysis...")
            
            try:
                # Comprehensive Market Intelligence Analysis
                market_intel = market_intelligence.get_market_intelligence_summary(df, {
                    'action': signal,
                    'symbol': symbol,
                    'rsi': rsi,
                    'macd': macd,
                    'macd_trend': macd_trend,
                    'volume_ratio': indicators.get('volume_ratio', 1.0),
                    'current_price': current_price,
                    'volatility': volatility
                })
                
                print(f"   📊 Market Regime: {market_intel['market_regime']['primary_regime']} "
                      f"(Confidence: {market_intel['market_regime']['confidence']:.2f})")
                print(f"   🔍 Pattern Confidence: {market_intel['pattern_analysis']['pattern_confidence']:.2f}")
                print(f"   📈 Signal Success Probability: {market_intel['signal_probability']:.2f}")
                print(f"   🎯 Intelligence Score: {market_intel['intelligence_score']:.2f}")
                
                # Update adaptive thresholds based on market regime
                if config.ADAPTIVE_THRESHOLDS_ENABLED:
                    adaptive_thresholds = market_intel['adaptive_thresholds']
                    print(f"   ⚙️ Adaptive RSI Thresholds: {adaptive_thresholds['rsi_oversold']:.1f}/{adaptive_thresholds['rsi_overbought']:.1f}")
                    print(f"   ⚙️ Adaptive MACD Threshold: {adaptive_thresholds['macd_threshold']:.4f}")
                
                # ML Pattern Recognition & Market Regime Prediction
                if config.PATTERN_RECOGNITION_ENABLED:
                    signal_success_probability = ml_predictor.predict_signal_success(
                        {'action': signal, 'rsi': rsi, 'macd': macd}, indicators
                    )
                    print(f"   🤖 ML Signal Success Prediction: {signal_success_probability:.2f}")
                
                # Market Regime Prediction
                if config.REGIME_DETECTION_ENABLED:
                    regime_prediction = ml_predictor.predict_market_regime(df)
                    print(f"   🌍 ML Regime Prediction: {regime_prediction['regime']} "
                          f"(Confidence: {regime_prediction['confidence']:.2f})")
                
                # Intelligence-based signal adjustment
                intelligence_score = market_intel['intelligence_score']
                signal_probability = market_intel['signal_probability']
                
                # Apply ML-based signal validation
                if intelligence_score < config.INTELLIGENCE_CONFIDENCE_THRESHOLD:
                    if signal != "HOLD":
                        print(f"   ⚠️ Low intelligence confidence ({intelligence_score:.2f} < {config.INTELLIGENCE_CONFIDENCE_THRESHOLD}) - Reducing position conviction")
                        # Don't change signal to HOLD, but flag for reduced position sizing
                        indicators['ml_confidence_low'] = True
                
                # Apply signal probability threshold
                if signal_probability < 0.4:  # Low success probability
                    print(f"   ⚠️ Low signal success probability ({signal_probability:.2f}) - Consider HOLD")
                    if signal_probability < 0.25:  # Very low probability
                        original_signal = signal
                        signal = "HOLD"
                        reason = f"ML Intelligence: Very low success probability ({signal_probability:.2f}) - Original: {original_signal}"
                        print(f"   🔄 Signal changed from {original_signal} to HOLD due to low ML probability")
                
                # Store ML intelligence in bot status
                bot_status['last_ml_intelligence'] = {
                    'market_regime': market_intel['market_regime']['primary_regime'],
                    'regime_confidence': market_intel['market_regime']['confidence'],
                    'pattern_confidence': market_intel['pattern_analysis']['pattern_confidence'],
                    'signal_probability': signal_probability,
                    'intelligence_score': intelligence_score,
                    'adaptive_thresholds': market_intel['adaptive_thresholds'],
                    'market_stress': market_intel['market_stress']['stress_level']
                }
                
                # Update reason with ML insights
                if signal != "HOLD":
                    ml_insights = []
                    if market_intel['market_regime']['confidence'] > 0.8:
                        ml_insights.append(f"Regime: {market_intel['market_regime']['primary_regime']}")
                    if signal_probability > 0.7:
                        ml_insights.append(f"High success prob: {signal_probability:.2f}")
                    if intelligence_score > 0.8:
                        ml_insights.append(f"High intelligence: {intelligence_score:.2f}")
                    
                    if ml_insights:
                        reason += f" | ML: {', '.join(ml_insights)}"
                
                print(f"   ✅ ML Intelligence analysis completed")
                
            except Exception as ml_error:
                print(f"   ⚠️ ML Intelligence error: {ml_error}")
                # Continue with signal if ML analysis fails
                pass
                
        # Update bot status with latest signal and timestamp
        current_time = format_cairo_time()
        bot_status.update({
            'last_signal': signal,
            'last_update': current_time,
            'last_strategy': strategy
        })
            
        print(f"Strategy {strategy} generated signal: {signal} - {reason}")  # Debug log
        
        # IMPORTANT: Check balance before allowing SELL signals
        if signal == "SELL":
            print(f"\n🔍 Checking if we have {symbol} balance for SELL order...")
            has_balance, available_balance, balance_msg = check_coin_balance(symbol)
            
            if not has_balance:
                print(f"❌ Cannot place SELL order: {balance_msg}")
                signal = "HOLD"
                reason = f"No balance to sell - {balance_msg}"
                print(f"🔄 Signal changed from SELL to HOLD due to insufficient balance")
                
                # Log the balance-prevented sell signal
                log_signal_to_csv(
                    "HOLD",
                    current_price,
                    indicators,
                    f"Strategy {strategy} wanted SELL but {balance_msg}"
                )
                
                # Send Telegram notification about blocked sell signal
                if TELEGRAM_AVAILABLE and getattr(config, 'TELEGRAM_SEND_SIGNALS', False):
                    try:
                        notify_signal("HOLD", symbol, current_price, indicators, 
                                    f"SELL blocked - {balance_msg}")
                    except Exception as telegram_error:
                        print(f"Telegram balance notification failed: {telegram_error}")
                
                return signal
            else:
                print(f"✅ Balance check passed: {balance_msg}")
                # Continue with original SELL signal
        
        # IMPORTANT: Check USDT balance before allowing BUY signals
        if signal == "BUY":
            try:
                if client:
                    account_info = client.get_account()
                    usdt_balance = 0
                    for balance in account_info['balances']:
                        if balance['asset'] == 'USDT':
                            usdt_balance = float(balance['free'])
                            break
                    
                    min_usdt_required = 10.0  # Minimum $10 USDT required
                    if usdt_balance < min_usdt_required:
                        print(f"❌ Insufficient USDT for BUY order: ${usdt_balance:.2f} < ${min_usdt_required}")
                        signal = "HOLD"
                        reason = f"Insufficient USDT balance: ${usdt_balance:.2f}"
                        print(f"🔄 Signal changed from BUY to HOLD due to insufficient USDT")
                        
                        # Log the balance-prevented buy signal
                        log_signal_to_csv(
                            "HOLD",
                            current_price,
                            indicators,
                            f"Strategy {strategy} wanted BUY but insufficient USDT: ${usdt_balance:.2f}"
                        )
                        
                        # Send Telegram notification about blocked buy signal
                        if TELEGRAM_AVAILABLE and getattr(config, 'TELEGRAM_SEND_SIGNALS', False):
                            try:
                                notify_signal("HOLD", symbol, current_price, indicators, 
                                            f"BUY blocked - insufficient USDT: ${usdt_balance:.2f}")
                            except Exception as telegram_error:
                                print(f"Telegram balance notification failed: {telegram_error}")
                        
                        return signal
                    else:
                        print(f"✅ USDT balance check passed: ${usdt_balance:.2f} available")
                else:
                    print("⚠️ Client not available for USDT balance check")
            except Exception as e:
                print(f"⚠️ Error checking USDT balance: {e}")
                # Continue with BUY signal if balance check fails
        
        # Enforce risk lock AFTER computing signal
        if risk_locked and signal != "HOLD":
            reason = f"Risk lock (consecutive losses {consecutive_losses}/{config.MAX_CONSECUTIVE_LOSSES})"
            signal = "HOLD"
        # Log strategy decision to signals log (single place)
        log_signal_to_csv(signal, current_price, indicators, f"Strategy {strategy} - {reason}")

        # Send Telegram notification for trading signals (configurable)
        if TELEGRAM_AVAILABLE and signal in ["BUY", "SELL"] and getattr(config, 'TELEGRAM_SEND_SIGNALS', False):
            try:
                notify_signal(signal, symbol, current_price, indicators, reason)
            except Exception as telegram_error:
                print(f"Telegram signal notification failed: {telegram_error}")

    except Exception as e:
        error_msg = f"Error in strategy execution: {str(e)}"
        print(error_msg)  # Debug log
        log_error_to_csv(error_msg, "STRATEGY_ERROR", "signal_generator", "ERROR")
        signal, reason = "HOLD", f"Strategy error: {str(e)}"
    
    return signal

def update_trade_tracking(trade_result, profit_loss=0):
    """Track consecutive wins/losses for smart risk management"""
    try:
        if trade_result == 'success':
            # Only update streaks on realized PnL events
            # Treat profit_loss == 0 (e.g., BUY entries or unknown PnL) as neutral: no change
            if profit_loss is None:
                return
            if profit_loss > 0:
                bot_status['consecutive_losses'] = 0  # Reset on profitable close
                bot_status['consecutive_wins'] = bot_status.get('consecutive_wins', 0) + 1
            elif profit_loss < 0:
                bot_status['consecutive_losses'] = bot_status.get('consecutive_losses', 0) + 1
                bot_status['consecutive_wins'] = 0
            else:
                # Flat/unknown PnL: don't modify counters
                bot_status['consecutive_wins'] = bot_status.get('consecutive_wins', 0)
        else:
            bot_status['consecutive_losses'] = bot_status.get('consecutive_losses', 0) + 1
            bot_status['consecutive_wins'] = 0
            
        # Log if consecutive losses are getting high
        if bot_status['consecutive_losses'] >= 3:
            log_error_to_csv(
                f"Consecutive losses: {bot_status['consecutive_losses']}", 
                "RISK_WARNING", 
                "update_trade_tracking", 
                "WARNING"
            )
    except Exception as e:
        log_error_to_csv(str(e), "TRACKING_ERROR", "update_trade_tracking", "ERROR")

def execute_trade(signal, symbol="BTCUSDT", qty=None):
    print("\n=== Trade Execution Debug Log ===")
    print(f"Attempting trade: {signal} for {symbol}")
    print(f"Initial quantity: {qty}")
    
    if signal == "HOLD":
        print("Signal is HOLD - no action needed")
        return f"Signal: {signal} - No action taken"
        
    # Get symbol info for precision and filters
    symbol_info = None
    try:
        if client:
            print("Getting exchange info from Binance API...")
            exchange_info = get_exchange_info_cached()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if symbol_info:
                print(f"Symbol info found for {symbol}:")
                print(f"Base Asset: {symbol_info['baseAsset']}")
                print(f"Quote Asset: {symbol_info['quoteAsset']}")
                print(f"Minimum Lot Size: {next((f['minQty'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), 'unknown')}")
                
                # Get current ticker info
                ticker = client.get_ticker(symbol=symbol)
                print(f"Current {symbol} price: ${float(ticker['lastPrice']):.2f}")
                print(f"24h Volume: {float(ticker['volume']):.2f} {symbol_info['baseAsset']}")
                print(f"24h Price Change: {float(ticker['priceChangePercent']):.2f}%")
            else:
                print(f"Warning: No symbol info found for {symbol}")
        else:
            print("Warning: Client not initialized - running in demo mode")
    except Exception as e:
        log_error_to_csv(str(e), "SYMBOL_INFO_ERROR", "execute_trade", "ERROR")
        print(f"Error getting symbol info: {e}")
        return f"Failed to get symbol info: {e}"
    
    # Calculate position size based on available balance and risk management
    try:
        if client:
            print("\n=== Enhanced Position Sizing & Risk Management ===")
            balance = client.get_account()
            
            # More robust balance extraction
            usdt_balance = 0
            btc_balance = 0
            for b in balance['balances']:
                if b['asset'] == 'USDT':
                    usdt_balance = float(b['free'])
                elif b['asset'] == 'BTC':
                    btc_balance = float(b['free'])
            
            print(f"Available USDT balance: {usdt_balance}")
            print(f"Available BTC balance: {btc_balance}")
            
            # Get current market price
            print("\n=== Price Check ===")
            ticker = client.get_ticker(symbol=symbol)
            current_price = float(ticker['lastPrice'])
            print(f"Current {symbol} price: {current_price}")
            print(f"24h price change: {ticker['priceChangePercent']}%")
            
            # Enhanced Position Sizing (if modules available)
            if ENHANCED_MODULES_AVAILABLE:
                try:
                    print("\n🧠 Calculating optimal position size...")
                    
                    # Get current positions for risk assessment
                    current_positions = {}  # This should be populated with actual positions
                    
                    # Prepare market conditions
                    market_conditions = {
                        'regime': bot_status.get('market_regime', 'NORMAL'),
                        'volatility': bot_status.get('volatility_metrics', {}).get('hourly_vol', 0.5),
                        'volume_surge': bot_status.get('volatility_metrics', {}).get('volume_surge', 1.0),
                        'volume_24h': float(ticker['volume']) * current_price,
                        'price_change_24h_pct': abs(float(ticker['priceChangePercent'])),
                        'spread_pct': 0.05  # Default spread
                    }
                    
                    # Get volatility from indicators if available
                    volatility = 0.5
                    if 'volatility' in locals():
                        volatility = locals()['volatility']
                    
                    # Get confidence score from signal quality
                    confidence_score = bot_status.get('last_signal_quality', {}).get('confidence', 0.5)
                    
                    # Calculate optimal position size
                    position_manager = get_position_manager()
                    position_result = position_manager.calculate_optimal_position_size(
                        symbol=symbol,
                        signal=signal,
                        current_price=current_price,
                        account_balance=usdt_balance,
                        volatility=volatility,
                        confidence_score=confidence_score,
                        market_regime=market_conditions['regime']
                    )
                    
                    # Comprehensive Risk Assessment
                    print("\n🛡️ Conducting comprehensive risk assessment...")
                    risk_manager = get_risk_manager()
                    
                    proposed_trade = {
                        'symbol': symbol,
                        'signal': signal,
                        'quantity': position_result['quantity'],
                        'price': current_price
                    }
                    
                    risk_assessment = risk_manager.comprehensive_risk_check(
                        account_balance=usdt_balance,
                        current_positions=current_positions,
                        proposed_trade=proposed_trade,
                        market_conditions=market_conditions
                    )
                    
                    print(f"   Risk Assessment: {'✅ APPROVED' if risk_assessment['approved'] else '❌ BLOCKED'}")
                    print(f"   Risk Score: {risk_assessment['risk_score']:.2f}")
                    print(f"   Warnings: {len(risk_assessment['warnings'])}")
                    print(f"   Blocks: {len(risk_assessment['blocks'])}")
                    
                    # Check if trade is approved
                    if not risk_assessment['approved']:
                        print(f"🚫 Trade blocked by risk management:")
                        for block in risk_assessment['blocks']:
                            print(f"   - {block}")
                        return f"Trade blocked: {'; '.join(risk_assessment['blocks'])}"
                    
                    # Apply risk adjustments
                    if 'position_size_multiplier' in risk_assessment.get('adjustments', {}):
                        multiplier = risk_assessment['adjustments']['position_size_multiplier']
                        position_result['quantity'] *= multiplier
                        print(f"   📉 Position size adjusted by {multiplier:.0%} due to risk factors")
                    
                    # Use enhanced position size
                    qty = position_result['quantity']
                    
                    print(f"\n📊 Enhanced Position Sizing Results:")
                    print(f"   Optimal Quantity: {qty:.8f}")
                    print(f"   Risk Amount: ${position_result['risk_amount']:.2f}")
                    print(f"   Risk Percentage: {position_result['risk_percentage']:.2f}%")
                    print(f"   Factors Applied:")
                    for factor, value in position_result.get('factors', {}).items():
                        print(f"     - {factor}: {value:.3f}")
                        
                    # Store enhanced metrics in trade info
                    trade_info['enhanced_metrics'] = {
                        'position_sizing': position_result,
                        'risk_assessment': risk_assessment
                    }
                    
                except Exception as enhanced_error:
                    print(f"   ⚠️ Enhanced positioning error: {enhanced_error}")
                    # Fall back to basic calculation
                    risk_amount = usdt_balance * (config.RISK_PERCENTAGE / 100)
                    qty = risk_amount / current_price
                    print(f"   Falling back to basic position sizing: {qty:.8f}")
            else:
                # Original basic calculation
                risk_amount = usdt_balance * (config.RISK_PERCENTAGE / 100)
                print(f"Risk amount ({config.RISK_PERCENTAGE}% of balance): {risk_amount} USDT")
                
                # Calculate quantity based on risk amount and current price
                raw_qty = risk_amount / current_price
                print(f"Raw quantity (before adjustments): {raw_qty}")
                qty = raw_qty
            
            if symbol_info:
                print("\n=== Final Position Sizing ===")
                # Get lot size filter
                lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                min_qty = float(lot_size_filter['minQty']) if lot_size_filter else 0.001
                print(f"Minimum allowed quantity: {min_qty}")
                
                # Ensure minimum trade value (Binance requires ~$10 minimum)
                min_trade_value = 10.0  # $10 minimum trade value
                if qty * current_price < min_trade_value:
                    qty = min_trade_value / current_price
                    print(f"Adjusted quantity for minimum trade value ($10): {qty}")
                
                qty = max(min_qty, qty)
                print(f"Quantity after minimum check: {qty}")
                
                # Round to correct precision
                step_size = float(lot_size_filter['stepSize']) if lot_size_filter else 0.001
                precision = len(str(step_size).split('.')[-1])
                qty = round(qty - (qty % float(step_size)), precision)
                print(f"Final quantity after rounding (step size {step_size}): {qty}")
                print(f"Estimated trade value: {qty * current_price} USDT")
    except Exception as e:
        log_error_to_csv(str(e), "POSITION_SIZE_ERROR", "execute_trade", "ERROR")
        qty = 0.001  # Fallback to minimum quantity
    
    # Create trade info structure
    trade_info = {
        'timestamp': format_cairo_time(),
        'signal': signal,
        'symbol': symbol,
        'quantity': qty,
        'status': 'simulated',
        'price': 0,
        'value': 0,
        'fee': 0
    }
    
    if client is None:
        error_msg = "Trading client not initialized. Cannot execute trade."
        log_error_to_csv(error_msg, "CLIENT_ERROR", "execute_trade", "ERROR")
        return error_msg
    
    try:
        print("\n=== Trade Execution ===")
        if signal == "BUY":
            print("Processing BUY order...")
            account_info = client.get_account()

            # Debug: Print all balances to see what we're getting
            print("=== Account Balances Debug ===")
            for balance in account_info['balances']:
                if float(balance['free']) > 0 or balance['asset'] == 'USDT':
                    print(f"{balance['asset']}: free={balance['free']}, locked={balance['locked']}")

            # More robust USDT balance extraction
            usdt_balance = None
            for balance in account_info['balances']:
                if balance['asset'] == 'USDT':
                    usdt_balance = balance
                    break

            if usdt_balance is None:
                print("❌ USDT balance not found in account")
                trade_info['status'] = 'no_usdt_balance'
                bot_status['trading_summary']['failed_trades'] += 1
                log_error_to_csv("USDT balance not found in account", "BALANCE_ERROR", "execute_trade", "ERROR")
                return "USDT balance not found"

            usdt = float(usdt_balance['free'])
            print(f"USDT available for buy: {usdt}")
            print(f"Minimum required: 10 USDT")
            print(f"Risk amount would be: {usdt * (config.RISK_PERCENTAGE / 100):.2f} USDT")

            if usdt < 10:
                print("❌ Insufficient USDT balance (minimum 10 USDT required)")
                trade_info['status'] = 'insufficient_funds'
                bot_status['trading_summary']['failed_trades'] += 1
                log_error_to_csv(f"Insufficient USDT for buy: {usdt} < 10", "BALANCE_ERROR", "execute_trade", "WARNING")
                return f"Insufficient USDT: {usdt:.2f} < 10.00"

            order = client.order_market_buy(symbol=symbol, quantity=qty)
            trade_info['price'] = float(order['fills'][0]['price']) if order['fills'] else 0
            trade_info['value'] = float(order['cummulativeQuoteQty'])
            trade_info['fee'] = sum([float(fill['commission']) for fill in order['fills']])
            trade_info['status'] = 'success'

            # Update trading summary
            bot_status['trading_summary']['total_buy_volume'] += trade_info['value']
            bot_status['trading_summary']['successful_trades'] += 1

        elif signal == "SELL":
            print("Processing SELL order...")

            # Double-check balance before executing (additional safety)
            has_balance, available_balance, balance_msg = check_coin_balance(symbol)

            if not has_balance:
                print(f"❌ Final balance check failed: {balance_msg}")
                trade_info['status'] = 'insufficient_funds'
                trade_info['error'] = balance_msg
                bot_status['trading_summary']['failed_trades'] += 1
                log_error_to_csv(f"SELL order blocked by final balance check: {balance_msg}",
                                 "BALANCE_ERROR", "execute_trade", "WARNING")
                return f"SELL order blocked: {balance_msg}"

            # Extract base asset from symbol (e.g., "BTC" from "BTCUSDT")
            base_asset = symbol[:-4] if symbol.endswith('USDT') else symbol.split(symbol_info['quoteAsset'])[0]

            print(f"✅ Final balance check passed. Proceeding with SELL order...")
            print(f"Available {base_asset} balance: {available_balance}")
            print(f"Requested quantity: {qty}")

            # Adjust quantity if needed to not exceed available balance
            if qty > available_balance:
                print(f"⚠️ Adjusting quantity from {qty} to {available_balance} (max available)")
                qty = available_balance
                trade_info['quantity'] = qty

            # Final quantity check with minimum requirements
            if qty <= 0:
                print("❌ Quantity is zero or negative after adjustments")
                trade_info['status'] = 'insufficient_quantity'
                bot_status['trading_summary']['failed_trades'] += 1
                return "Cannot place SELL order: quantity too small"

            print(f"🚀 Placing market sell order: {qty} {base_asset}")
            order = client.order_market_sell(symbol=symbol, quantity=qty)
            trade_info['price'] = float(order['fills'][0]['price']) if order['fills'] else 0
            trade_info['value'] = float(order['cummulativeQuoteQty'])
            trade_info['fee'] = sum([float(fill['commission']) for fill in order['fills']])
            trade_info['status'] = 'success'

            print(f"✅ SELL order executed successfully!")
            print(f"   Order ID: {order.get('orderId', 'N/A')}")
            print(f"   Price: ${trade_info['price']:.4f}")
            print(f"   Value: ${trade_info['value']:.2f}")
            print(f"   Fee: ${trade_info['fee']:.4f}")

            # Update trading summary
            bot_status['trading_summary']['total_sell_volume'] += trade_info['value']
            bot_status['trading_summary']['successful_trades'] += 1

            # Calculate revenue (sell value minus rough cost basis)
            revenue = 0
            try:
                if bot_status['trading_summary']['total_buy_volume'] > 0 and qty and qty > 0:
                    # Rough average buy price estimate
                    avg_buy_price = bot_status['trading_summary']['total_buy_volume'] / max(1, (bot_status['trading_summary']['successful_trades'] // 2 or 1))
                    revenue = trade_info['value'] - (qty * avg_buy_price)
            except Exception:
                revenue = 0
            bot_status['trading_summary']['total_revenue'] += revenue
            # Track daily_loss (treat negative revenue as loss)
            bot_status['daily_loss'] = bot_status.get('daily_loss', 0.0) + (-revenue if revenue < 0 else 0.0)

        # Update trade history (keep last 10 trades)
        bot_status['trading_summary']['trades_history'].insert(0, trade_info)
        if len(bot_status['trading_summary']['trades_history']) > 10:
            bot_status['trading_summary']['trades_history'].pop()

        # Log real trade to CSV
        try:
            balance_before = balance_after = 0
            if client:
                account = client.get_account()
                # More robust balance extraction for logging
                usdt_balance = 0
                btc_balance = 0
                for balance in account['balances']:
                    if balance['asset'] == 'USDT':
                        usdt_balance = float(balance['free'])
                    elif balance['asset'] == 'BTC':
                        btc_balance = float(balance['free'])
                balance_after = usdt_balance + (btc_balance * trade_info['price'])

            additional_data = {
                'rsi': bot_status.get('rsi', 50),
                'macd_trend': bot_status.get('macd', {}).get('trend', 'NEUTRAL'),
                'sentiment': bot_status.get('sentiment', 'neutral'),
                'balance_before': balance_before,
                'balance_after': balance_after,
                'profit_loss': (revenue if (signal == "SELL" and 'revenue' in locals()) else 0),
                'order_id': order.get('orderId', '') if 'order' in locals() else ''
            }
            trade_info['order_id'] = additional_data['order_id']
            trade_info['profit_loss'] = additional_data['profit_loss']
            log_trade_to_csv(trade_info, additional_data)

            # Send Telegram notification for successful trades
            if TELEGRAM_AVAILABLE:
                try:
                    notify_trade(trade_info, is_executed=True)
                except Exception as telegram_error:
                    print(f"Telegram trade notification failed: {telegram_error}")

        except Exception as csv_error:
            log_error_to_csv(f"CSV logging error: {csv_error}", "CSV_ERROR", "execute_trade", "WARNING")

        # Update statistics
        total_trades = bot_status['trading_summary']['successful_trades'] + bot_status['trading_summary']['failed_trades']
        bot_status['total_trades'] = total_trades

        if total_trades > 0:
            bot_status['trading_summary']['win_rate'] = (bot_status['trading_summary']['successful_trades'] / total_trades) * 100
            bot_status['trading_summary']['average_trade_size'] = (
                bot_status['trading_summary']['total_buy_volume'] + bot_status['trading_summary']['total_sell_volume']
            ) / total_trades if total_trades > 0 else 0

        # Update smart trade tracking (only pass PnL for realized SELLs)
        realized_pnl = None
        if signal == "SELL":
            realized_pnl = revenue
        update_trade_tracking('success', realized_pnl)

        return f"{signal} order executed: {order['orderId']} at ${trade_info['price']:.2f}"

    except BinanceAPIException as e:
        trade_info['status'] = 'api_error'
        bot_status['trading_summary']['failed_trades'] += 1
        bot_status['trading_summary']['trades_history'].insert(0, trade_info)
        bot_status['errors'].append(str(e))

        # Update smart trade tracking for failed trades
        update_trade_tracking('failed', -1)

        # Log failed trade to CSV
        additional_data = {
            'rsi': bot_status.get('rsi', 50),
            'macd_trend': bot_status.get('macd', {}).get('trend', 'NEUTRAL'),
            'sentiment': bot_status.get('sentiment', 'neutral'),
            'balance_before': 0,
            'balance_after': 0,
            'profit_loss': 0
        }
        log_trade_to_csv(trade_info, additional_data)
        log_error_to_csv(str(e), "API_ERROR", "execute_trade", "ERROR")

        # Send Telegram notification for failed trades
        if TELEGRAM_AVAILABLE:
            try:
                notify_trade(trade_info, is_executed=False)
            except Exception as telegram_error:
                print(f"Telegram failed trade notification failed: {telegram_error}")

        return f"Order failed: {str(e)}"

def scan_trading_pairs(base_assets=None, quote_asset="USDT", min_volume_usdt=1000000):
    """Smart multi-coin scanner for best trading opportunities with rate limiting"""
    opportunities = []
    
    # Default assets if none provided
    if base_assets is None:
        base_assets = ["BTC", "ETH", "BNB", "XRP", "SOL", "MATIC", "DOT", "ADA", "AVAX", "LINK"]  # Restored original 10 symbols
    
    # Add rate limiting to prevent API ban
    scan_delay = 0.5  # 500ms delay between API calls
    
    for base in base_assets:
        try:
            symbol = f"{base}{quote_asset}"
            
            # Rate limiting - wait before each API call
            time.sleep(scan_delay)
            
            # Get 24h ticker statistics
            ticker = client.get_ticker(symbol=symbol)
            volume_usdt = float(ticker['quoteVolume'])
            price_change_pct = float(ticker['priceChangePercent'])
            
            # Skip if volume too low
            if volume_usdt < min_volume_usdt:
                continue
            
            # Rate limiting before data fetch
            time.sleep(scan_delay)
            
            # Fetch market data with smaller limit to reduce API weight
            df = fetch_data(symbol=symbol, limit=30)  # Reduced from 50 to 30
            if df is None or len(df) < 15:  # Reduced minimum from 20 to 15
                continue
            
            # Calculate technical indicators with proper error handling
            current_price = float(df['close'].iloc[-1])
            
            # Get RSI - it should already be calculated in fetch_data
            if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]):
                current_rsi = float(df['rsi'].iloc[-1])
            else:
                # Fallback calculation
                prices = df['close'].values
                current_rsi = calculate_rsi(prices, period=14)
            
            # Get MACD trend - it should already be calculated in fetch_data  
            if 'macd_trend' in df.columns and not pd.isna(df['macd_trend'].iloc[-1]):
                macd_trend = df['macd_trend'].iloc[-1]
            else:
                # Fallback calculation
                prices = df['close'].values
                macd_result = calculate_macd(prices)
                macd_trend = macd_result.get('trend', 'NEUTRAL')
            
            # Get SMA values with error handling
            try:
                sma_fast = calculate_sma(df, period=10)
                sma_slow = calculate_sma(df, period=20)
                
                if len(sma_fast) == 0 or len(sma_slow) == 0:
                    continue  # Skip if we can't calculate SMAs
                    
                sma_fast_value = float(sma_fast.iloc[-1])
                sma_slow_value = float(sma_slow.iloc[-1])
            except Exception as sma_error:
                log_error_to_csv(f"SMA calculation error for {symbol}: {sma_error}", 
                               "SMA_ERROR", "scan_trading_pairs", "WARNING")
                continue
            
            # Score the opportunity (0-100)
            opportunity_score = 0
            signals = []
            
            # Check if we have balance for this coin (for potential sell signals)
            has_balance, available_balance, balance_msg = check_coin_balance(symbol)
            can_sell = has_balance and available_balance > 0
            
            # RSI scoring with balance-aware adjustments
            if current_rsi < 30:  # Oversold - good for buying
                opportunity_score += 30
                signals.append("RSI_OVERSOLD")
            elif current_rsi > 70:  # Overbought - good for selling if we have balance
                if can_sell:
                    opportunity_score += 25  # Higher score if we can actually sell
                    signals.append("RSI_OVERBOUGHT_SELLABLE")
                else:
                    opportunity_score += 5  # Lower score if we can't sell
                    signals.append("RSI_OVERBOUGHT_NO_BALANCE")
            elif 45 <= current_rsi <= 55:  # Neutral zone
                opportunity_score += 10
                signals.append("RSI_NEUTRAL")
            
            # MACD scoring with balance awareness
            if macd_trend == "BULLISH":
                opportunity_score += 20
                signals.append("MACD_BULLISH")
            elif macd_trend == "BEARISH":
                if can_sell:
                    opportunity_score += 15  # Bearish trend good for selling if we have balance
                    signals.append("MACD_BEARISH_SELLABLE")
                else:
                    signals.append("MACD_BEARISH_NO_BALANCE")
            
            # Price momentum scoring
            if abs(price_change_pct) > 5:  # High volatility
                opportunity_score += 15
                signals.append("HIGH_VOLATILITY")
            
            # Volume scoring
            if volume_usdt > min_volume_usdt * 5:  # Very high volume
                opportunity_score += 15
                signals.append("HIGH_VOLUME")
            
            # SMA trend scoring with balance considerations
            if current_price > sma_fast_value > sma_slow_value:
                opportunity_score += 10
                signals.append("UPTREND")
            elif current_price < sma_fast_value < sma_slow_value:
                if can_sell:
                    opportunity_score += 15  # Downtrend good for selling if we have balance
                    signals.append("DOWNTREND_SELLABLE")
                else:
                    opportunity_score += 5  # Lower score if we can't sell
                    signals.append("DOWNTREND_NO_BALANCE")
            
            # Add balance information to the opportunity
            balance_info = {
                'has_balance': can_sell,
                'available_balance': available_balance if has_balance else 0,
                'balance_msg': balance_msg
            }
            
            opportunities.append({
                'symbol': symbol,
                'score': opportunity_score,
                'price': current_price,
                'volume_usdt': volume_usdt,
                'price_change_pct': price_change_pct,
                'rsi': current_rsi,
                'macd_trend': macd_trend,
                'signals': signals,
                'balance_info': balance_info,  # Add balance information
                'data': df  # Include data for immediate analysis if selected
            })
            
        except Exception as e:
            log_error_to_csv(f"Error scanning {base}{quote_asset}: {e}", 
                           "SCAN_ERROR", "scan_trading_pairs", "WARNING")
            continue
    
    # Sort by opportunity score (highest first)
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    
    # Log top opportunities with balance information
    if opportunities:
        print(f"\n=== Top Trading Opportunities ===")
        for i, opp in enumerate(opportunities[:5]):  # Show top 5
            balance_status = "✅" if opp['balance_info']['has_balance'] else "❌"
            balance_amount = f"{opp['balance_info']['available_balance']:.4f}" if opp['balance_info']['has_balance'] else "0"
            
            print(f"{i+1}. {opp['symbol']}: Score {opp['score']}, RSI {opp['rsi']:.1f}, "
                  f"Change {opp['price_change_pct']:.2f}%, Balance: {balance_status}({balance_amount}), "
                  f"Signals: {', '.join(opp['signals'])}")
    
    return opportunities

def trading_loop():
    """Professional AI Trading Wolf - Intelligent Timing and Opportunity Hunting"""
    bot_status['running'] = True
    bot_status['signal_scanning_active'] = True  # Activate signal scanning
    consecutive_errors = 0
    max_consecutive_errors = 5
    error_sleep_time = 60  # Start with 1 minute on errors
    
    print("\n🐺 === AI TRADING WOLF ACTIVATED ===")
    print("🎯 Professional timing system engaged")
    print("📊 Market regime detection online")
    print("⚡ Breakout opportunity scanning active")
    print("📡 Signal scanning activated")
    print("\n🛡️ === RATE LIMITING ACTIVE ===")
    print("⏱️ Global signal cooldown: 45 seconds between ANY signals")
    print("🔒 Symbol signal cooldown: 90 seconds per symbol")
    print("🚫 Signal type cooldown: 180 seconds for same signal type")
    print("📊 Scan cycle limit: 1 signal per scanning cycle")
    print("🕒 BTC fallback cooldown: 60 seconds")
    print("=" * 50)
    
    # Initialize trading summary if not exists
    if 'trading_summary' not in bot_status:
        bot_status['trading_summary'] = {
            'successful_trades': 0,
            'failed_trades': 0,
            'total_trades': 0,
            'total_buy_volume': 0.0,
            'total_sell_volume': 0.0,
            'total_revenue': 0.0,
            'win_rate': 0.0,
            'average_trade_size': 0.0,
            'trades_history': []
        }
    
    # Ensure API client is initialized (should already be done at startup)
    if not bot_status.get('api_connected', False):
        print("⚠️ API client not connected at trading loop start - attempting reconnection...")
        initialize_client()
        if not bot_status.get('api_connected', False):
            log_error_to_csv("API client not initialized before trading loop start", "CLIENT_ERROR", "trading_loop", "ERROR")
            time.sleep(10)  # Wait longer before giving up
            return  # Exit trading loop if can't connect

    # Initialize multi-coin tracking and regime detection
    bot_status['monitored_pairs'] = {}
    bot_status['market_regime'] = 'NORMAL'
    bot_status['hunting_mode'] = False
    bot_status['last_daily_summary'] = None  # Track when we last sent daily summary
    
    # Initial market regime detection and IMMEDIATE first scan
    initial_regime = detect_market_regime()
    # Mark last regime check now to avoid immediate duplicate checks
    bot_status['last_volatility_check'] = get_cairo_time()
    initial_interval, initial_mode = calculate_smart_interval()
    
    print(f"🎯 Initial scan mode: {initial_mode} ({initial_interval}s)")
    print(f"🚀 Performing immediate startup scan...")
    
    # Perform immediate first scan
    try:
        print(f"\n🐺 === WOLF SCANNING ACTIVATED (STARTUP) ===")
        print(f"🕒 Time: {format_cairo_time()}")
        print(f"🎯 Scan Reason: STARTUP_SCAN")
        print(f"📊 Market Regime: {bot_status.get('market_regime', 'NORMAL')}")
        
        # Scan all trading pairs immediately (restored to original scan)
        scan_results = scan_trading_pairs()  # Uses default 10 symbols
        bot_status['last_scan_time'] = get_cairo_time()  # Record scan time
        print(f"✅ Startup scan completed - found {len(scan_results) if scan_results else 0} opportunities")
        
        # Generate signals for top opportunities at startup
        if scan_results:
            print("\n🎯 Analyzing top opportunities from startup scan...")
            max_startup_targets = 3  # Analyze top 3 opportunities
            signals_generated = 0
            
            for i, opportunity in enumerate(scan_results[:max_startup_targets]):
                current_symbol = opportunity['symbol']
                current_score = opportunity.get('score', 0)
                
                print(f"\n📊 Analyzing startup target {i+1}: {current_symbol} (Score: {current_score:.1f})")
                
                # Get fresh data for signal generation
                df = fetch_data(symbol=current_symbol, interval="5m", limit=100)
                if df is not None:
                    signal = signal_generator(df, current_symbol)
                    current_price = float(df['close'].iloc[-1])
                    
                    print(f"💡 Generated signal: {signal} @ ${current_price:.4f}")
                    
                    # Log signal to CSV with startup context
                    indicators = {
                        'symbol': current_symbol,
                        'rsi': float(df['rsi'].iloc[-1]),
                        'macd': float(df['macd'].iloc[-1]),
                        'macd_trend': df['macd_trend'].iloc[-1],
                        'opportunity_score': current_score
                    }
                    log_signal_to_csv(signal, current_price, indicators, "STARTUP_SCAN")
                    
                    # Update pair tracking
                    if current_symbol not in bot_status['monitored_pairs']:
                        bot_status['monitored_pairs'][current_symbol] = {
                            'last_signal': signal,
                            'last_price': current_price,
                            'rsi': float(df['rsi'].iloc[-1]),
                            'macd': {
                                'macd': float(df['macd'].iloc[-1]),
                                'signal': float(df['macd_signal'].iloc[-1]),
                                'trend': df['macd_trend'].iloc[-1]
                            },
                            'last_update': format_cairo_time(),
                            'opportunity_score': current_score
                        }
                    
                    # Update main status with best target
                    if i == 0:
                        bot_status.update({
                            'current_symbol': current_symbol,
                            'last_signal': signal,
                            'last_price': current_price,
                            'last_update': format_cairo_time()
                        })
                    
                    signals_generated += 1
            
            print(f"\n✨ Startup signal generation complete - Analyzed {signals_generated} opportunities")
        
    except Exception as e:
        print(f"⚠️ Startup scan failed: {e}")
    
    # Set next scan time after immediate scan
    bot_status['next_signal_time'] = get_cairo_time() + timedelta(seconds=initial_interval)
    bot_status['signal_interval'] = initial_interval
    print(f"📅 Next scan: {format_cairo_time(bot_status['next_signal_time'])}")
    
    last_major_scan = get_cairo_time()
    quick_scan_count = 0
    
    while bot_status['running']:
        try:
            current_time = get_cairo_time()

            # Safety: decay consecutive losses after cooldown period (e.g., 2 hours without trades)
            try:
                last_trade_time = None
                if bot_status.get('trading_summary', {}).get('trades_history'):
                    last_trade = bot_status['trading_summary']['trades_history'][0]
                    last_trade_time = last_trade.get('timestamp')
                if last_trade_time:
                    # Parse time if string
                    if isinstance(last_trade_time, str):
                        try:
                            last_trade_dt = datetime.fromisoformat(last_trade_time.replace('Z', '+00:00'))
                        except Exception:
                            last_trade_dt = current_time
                    else:
                        last_trade_dt = last_trade_time
                    if (current_time - last_trade_dt).total_seconds() > 2 * 3600:
                        # Gradually reduce penalties
                        bot_status['consecutive_losses'] = max(0, bot_status.get('consecutive_losses', 0) - 1)
                # Hard cap to avoid indefinite lockout
                bot_status['consecutive_losses'] = min(bot_status.get('consecutive_losses', 0), config.MAX_CONSECUTIVE_LOSSES)
            except Exception:
                pass
            
            # Health check - only reinitialize if connection is actually lost
            if not bot_status['api_connected']:
                print("🔄 API connection lost - attempting to reconnect...")
                initialize_client()
                if not bot_status['api_connected']:
                    print("❌ Failed to reconnect to API - retrying in next cycle")
                    time.sleep(30)  # Wait before retrying
                    continue
            
            # Check for profit-taking opportunities in existing positions
            try:
                print("\n💰 Checking existing positions for profit-taking opportunities...")
                
                # Get current balances
                assets_with_balance = []
                account_info = client.get_account() if client else None
                
                if account_info:
                    for balance in account_info['balances']:
                        asset = balance['asset']
                        free_balance = float(balance['free'])
                        
                        if free_balance > 0 and asset != 'USDT':
                            symbol = f"{asset}USDT"
                            ticker = client.get_ticker(symbol=symbol)
                            current_price = float(ticker['lastPrice'])
                            usdt_value = free_balance * current_price
                            
                            df = fetch_data(symbol=symbol, interval="5m", limit=100)
                            if df is not None:
                                rsi = float(df['rsi'].iloc[-1])
                                
                                if rsi >= 70:  # Overbought condition
                                    print(f"🎯 Found overbought position: {symbol}")
                                    print(f"   💰 Balance: {free_balance:.8f} {asset}")
                                    print(f"   💵 Value: ${usdt_value:.2f}")
                                    print(f"   📈 RSI: {rsi:.1f}")
                                    
                                    # Execute SELL directly on overbought condition
                                    print(f"⚡ Executing profit-taking SELL for {symbol} (RSI: {rsi:.1f})")
                                    result = execute_trade("SELL", symbol)
                                    print(f"📊 Result: {result}")
            except Exception as e:
                print(f"⚠️ Error checking profit-taking opportunities: {e}")
            
            # Intelligent scan decision
            should_scan, scan_reason = should_scan_now()
            
            if not should_scan:
                # Sleep in short bursts to allow for interruptions
                time.sleep(min(30, bot_status.get('signal_interval', 300) // 10))
                continue
                
            print(f"\n🐺 === WOLF SCANNING ACTIVATED ===")
            print(f"🕒 Time: {format_cairo_time()}")
            print(f"🎯 Scan Reason: {scan_reason}")
            print(f"📊 Market Regime: {bot_status.get('market_regime', 'NORMAL')}")
            print(f"⚡ Hunting Mode: {'ON' if bot_status.get('hunting_mode') else 'OFF'}")
            
            # Update market regime every major scan
            if (current_time - last_major_scan).total_seconds() > 1800:  # Every 30 minutes
                detect_market_regime()
                last_major_scan = current_time
                quick_scan_count = 0
                
            # Quick breakout scan if in hunting mode
            breakout_opportunities = []
            if bot_status.get('hunting_mode') or bot_status.get('market_regime') in ['VOLATILE', 'EXTREME']:
                breakout_opportunities = detect_breakout_opportunities()
                quick_scan_count += 1
                
                if breakout_opportunities:
                    print(f"🚀 BREAKOUT OPPORTUNITIES DETECTED:")
                    for opp in breakout_opportunities[:2]:
                        print(f"   💎 {opp['symbol']}: Score {opp['score']}, Signals: {', '.join(opp['signals'])}")
            
            # Full market scan (intelligent frequency)
            should_full_scan = (
                not breakout_opportunities or  # No breakouts found
                quick_scan_count >= 5 or      # Max quick scans reached
                (current_time - last_major_scan).total_seconds() > 3600  # Force every hour
            )
            
            if should_full_scan:
                print("🔍 Performing FULL MARKET SCAN")
                opportunities = scan_trading_pairs(
                    base_assets=["BTC", "ETH", "BNB", "XRP", "SOL", "MATIC", "DOT", "ADA", "AVAX", "LINK"],  # Restored original 10 symbols
                    quote_asset="USDT",
                    min_volume_usdt=500000  # Lower threshold for more opportunities
                )
                bot_status['last_scan_time'] = get_cairo_time()  # Record full scan time
                quick_scan_count = 0
            else:
                print("⚡ Using BREAKOUT SCAN results")
                opportunities = breakout_opportunities
                bot_status['last_scan_time'] = get_cairo_time()  # Record breakout scan time
            
            # Process opportunities
            if not opportunities:
                print("😴 No significant opportunities found - Wolf resting")
                
                # Fallback to default pair (only if not already processed)
                current_symbol = "BTCUSDT"
                
                # Check if BTCUSDT was already processed in recent scan (within last 60 seconds)
                last_btc_scan = bot_status.get('last_btc_scan_time')
                current_time = get_cairo_time()
                
                if (last_btc_scan is None or 
                    (current_time - last_btc_scan).total_seconds() > 60):
                    
                    df = fetch_data(symbol=current_symbol, interval="5m", limit=100)
                    if df is not None:
                        signal = signal_generator(df, current_symbol)
                        current_price = float(df['close'].iloc[-1])
                        
                        bot_status.update({
                            'current_symbol': current_symbol,
                            'last_signal': signal,
                            'last_price': current_price,
                            'last_update': format_cairo_time(),
                            'rsi': float(df['rsi'].iloc[-1]),
                            'macd': {
                                'macd': float(df['macd'].iloc[-1]),
                                'signal': float(df['macd_signal'].iloc[-1]),
                                'trend': df['macd_trend'].iloc[-1]
                            },
                            'last_btc_scan_time': current_time  # Track when we last scanned BTC
                        })
                        print(f"📊 Default analysis: {signal} for {current_symbol}")
                else:
                    print(f"⚠️ Skipping default {current_symbol} scan - analyzed recently")
            else:
                print(f"🎯 Found {len(opportunities)} hunting targets")
                
                # Process top opportunities with intelligent prioritization
                max_targets = 1  # Limit to 1 target per cycle to prevent signal flooding
                processed_symbols = set()  # Track processed symbols to avoid duplicates
                signals_generated_this_cycle = 0  # Track signals in this cycle
                
                for i, opportunity in enumerate(opportunities[:max_targets]):
                    current_symbol = opportunity['symbol']
                    current_score = opportunity.get('score', 0)
                    
                    # Skip if we've already processed this symbol in this cycle
                    if current_symbol in processed_symbols:
                        print(f"⚠️ Skipping {current_symbol} - already processed in this cycle")
                        continue
                    processed_symbols.add(current_symbol)
                    
                    # Limit signals per cycle
                    if signals_generated_this_cycle >= 1:
                        print(f"🛑 Signal limit reached for this cycle - skipping remaining opportunities")
                        break
                    
                    print(f"\n🎯 === TARGET {i+1}: {current_symbol} ===")
                    print(f"💪 Score: {current_score:.1f}")
                    
                    # Get fresh data for analysis
                    interval = "1m" if bot_status.get('hunting_mode') else "5m"
                    df = fetch_data(symbol=current_symbol, interval=interval, limit=100)
                    
                    if df is None:
                        continue
                        
                    # Enhanced signal generation with market regime consideration
                    signal = signal_generator(df, current_symbol)
                    signals_generated_this_cycle += 1  # Track signals generated in this cycle
                    current_price = float(df['close'].iloc[-1])
                    
                    print(f"🚦 Signal: {signal} (#{signals_generated_this_cycle} this cycle)")
                    print(f"💰 Price: ${current_price:.4f}")
                    
                    if 'rsi' in opportunity:
                        print(f"📈 RSI: {opportunity['rsi']:.1f}")
                    if 'signals' in opportunity:
                        print(f"⚡ Triggers: {', '.join(opportunity['signals'])}")
                    
                    # Update pair tracking
                    if current_symbol not in bot_status['monitored_pairs']:
                        bot_status['monitored_pairs'][current_symbol] = {
                            'last_signal': 'HOLD',
                            'last_price': 0,
                            'rsi': 50,
                            'macd': {'macd': 0, 'signal': 0, 'trend': 'NEUTRAL'},
                            'sentiment': 'neutral',
                            'total_trades': 0,
                            'successful_trades': 0,
                            'last_trade_time': None
                        }
                    
                    bot_status['monitored_pairs'][current_symbol].update({
                        'last_signal': signal,
                        'last_price': current_price,
                        'rsi': float(df['rsi'].iloc[-1]),
                        'macd': {'trend': df['macd_trend'].iloc[-1]},
                        'last_update': format_cairo_time(),
                        'opportunity_score': current_score
                    })
                    
                    # Update main status with best target
                    if i == 0:
                        bot_status.update({
                            'current_symbol': current_symbol,
                            'last_signal': signal,
                            'last_price': current_price,
                            'last_update': format_cairo_time(),
                            'rsi': float(df['rsi'].iloc[-1]),
                            'macd': {'trend': df['macd_trend'].iloc[-1]},
                            'opportunity_score': current_score
                        })
                    
                    # Execute trade with enhanced conditions
                    if signal in ["BUY", "SELL"]:
                        # Initialize risk tracking if not present
                        if 'consecutive_losses' not in bot_status:
                            bot_status['consecutive_losses'] = 0
                        if 'daily_loss' not in bot_status:
                            bot_status['daily_loss'] = 0.0
                        
                        # Risk management checks with debug logging
                        consecutive_losses = bot_status.get('consecutive_losses', 0)
                        daily_loss = bot_status.get('daily_loss', 0.0)
                        
                        print(f"🔍 Risk Management Check:")
                        print(f"   Consecutive losses: {consecutive_losses}/{config.MAX_CONSECUTIVE_LOSSES}")
                        print(f"   Daily loss: ${daily_loss:.2f}/${config.MAX_DAILY_LOSS}")
                        print(f"   API Connected: {bot_status.get('api_connected', False)}")
                        print(f"   Can Trade (Account): {bot_status.get('can_trade', False)}")
                        
                        can_trade = (
                            consecutive_losses < config.MAX_CONSECUTIVE_LOSSES and
                            daily_loss < config.MAX_DAILY_LOSS and
                            bot_status.get('api_connected', False) and
                            bot_status.get('can_trade', False)
                        )
                        
                        # Additional hunting mode conditions
                        if bot_status.get('hunting_mode'):
                            can_trade = can_trade and current_score >= 50  # Higher threshold in hunting mode
                            print(f"   Hunting mode score: {current_score}/50")
                        
                        if can_trade:
                            print(f"🚀 EXECUTING {signal} for {current_symbol}")
                            result = execute_trade(signal, current_symbol)
                            print(f"📊 Result: {result}")
                            
                            # Update tracking
                            bot_status['monitored_pairs'][current_symbol]['total_trades'] += 1
                            if "executed" in str(result).lower():
                                bot_status['monitored_pairs'][current_symbol]['successful_trades'] += 1
                            
                            # In hunting mode, only take the best trade
                            if bot_status.get('hunting_mode'):
                                break
                        else:
                            print(f"🛑 Trade blocked by risk management")
                            print(f"   Consecutive losses: {consecutive_losses}/{config.MAX_CONSECUTIVE_LOSSES}")
                            print(f"   Daily loss: ${daily_loss:.2f}/${config.MAX_DAILY_LOSS}")
                            print(f"   API Connected: {bot_status.get('api_connected', False)}")
                            print(f"   Account Can Trade: {bot_status.get('can_trade', False)}")
                            if bot_status.get('hunting_mode'):
                                print(f"   Hunting mode score: {current_score}/50")
            
            consecutive_errors = 0  # Reset error counter on successful cycle
            
            # Calculate next scan time with intelligent timing
            next_interval, next_mode = calculate_smart_interval()
            bot_status['next_signal_time'] = get_cairo_time() + timedelta(seconds=next_interval)
            bot_status['signal_interval'] = next_interval
            
            print(f"\n🎯 Next scan: {next_mode} mode in {next_interval}s ({next_interval/60:.1f}min)")
            print(f"📅 Expected at: {format_cairo_time(bot_status['next_signal_time'])}")
            
            # Send periodic Telegram market updates (every hour)
            if TELEGRAM_AVAILABLE and (current_time - last_major_scan).total_seconds() > 3600:
                try:
                    volatility_metrics = bot_status.get('volatility_metrics', {})
                    next_scan_str = format_cairo_time(bot_status['next_signal_time'])
                    notify_market_update(
                        bot_status.get('market_regime', 'NORMAL'),
                        bot_status.get('hunting_mode', False),
                        next_scan_str,
                        volatility_metrics
                    )
                except Exception as telegram_error:
                    print(f"Telegram market update failed: {telegram_error}")
            
            # Send daily summary each day at 08:00 Cairo time (once per day) and log daily performance for the previous day
            if config.TELEGRAM.get('notifications', {}).get('daily_summary', True):
                try:
                    last_summary_date = bot_status.get('last_daily_summary')
                    current_date = current_time.strftime('%Y-%m-%d')
                    # Trigger within the first 15 minutes after 08:00 to allow for scheduling jitter
                    if (current_time.hour == 8 and current_time.minute < 15 and
                        (last_summary_date != current_date)):
                        # First, write yesterday's performance row to CSV
                        yesterday = (current_time - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
                        log_daily_performance(yesterday)
                        # Then, send Telegram daily summary if Telegram is available
                        if TELEGRAM_AVAILABLE:
                            notify_daily_summary(bot_status.get('trading_summary', {}))
                        bot_status['last_daily_summary'] = current_date
                        print(f"📊 Daily performance logged and summary processed for {current_date} (08:00 Cairo)")
                except Exception as telegram_error:
                    print(f"Daily summary notification failed: {telegram_error}")
            
            # Process any queued Telegram messages
            if TELEGRAM_AVAILABLE:
                try:
                    process_queued_notifications()
                except Exception as telegram_error:
                    print(f"Telegram queue processing failed: {telegram_error}")
            
            # === AUTOMATIC POSITION REBALANCING ===
            try:
                # Check if it's time for automatic rebalancing
                if config.REBALANCING.get('enabled', True):
                    rebalance_interval = config.REBALANCING.get('check_interval', 3600)  # Default 1 hour
                    last_rebalance = bot_status.get('last_rebalance_check', 0)
                    
                    # Convert to timestamp if it's a datetime object
                    if isinstance(last_rebalance, datetime):
                        last_rebalance = last_rebalance.timestamp()
                    
                    current_timestamp = current_time.timestamp()
                    
                    if current_timestamp - last_rebalance >= rebalance_interval:
                        print(f"\n🔄 Automatic rebalancing check (every {rebalance_interval//60} minutes)")
                        
                        # Execute rebalancing
                        rebalance_results = execute_position_rebalancing()
                        
                        # Update last rebalance time
                        bot_status['last_rebalance_check'] = current_timestamp
                        bot_status['last_rebalance_results'] = rebalance_results
                        
                        # Log summary
                        actions_count = len(rebalance_results.get('partial_sells', [])) + len(rebalance_results.get('dust_liquidations', []))
                        if actions_count > 0:
                            print(f"✅ Rebalancing completed: {actions_count} actions, ${rebalance_results.get('total_freed_usdt', 0):.2f} USDT freed")
                        else:
                            print("ℹ️ No rebalancing actions needed")
                            
            except Exception as rebalance_error:
                print(f"⚠️ Automatic rebalancing error: {rebalance_error}")
                log_error_to_csv(str(rebalance_error), "AUTO_REBALANCE_ERROR", "trading_loop", "WARNING")
            
            # Smart sleep with early wake capabilities
            sleep_chunks = max(1, next_interval // 30)  # Wake up periodically
            chunk_size = next_interval / sleep_chunks
            
            for _ in range(int(sleep_chunks)):
                if not bot_status['running']:
                    break
                time.sleep(chunk_size)
        
        except KeyboardInterrupt:
            print("\n🛑 === KEYBOARD INTERRUPT ===")
            bot_status['running'] = False
            break
            
        except Exception as e:
            consecutive_errors += 1
            error_msg = f"Trading wolf error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}"
            print(f"⚠️ {error_msg}")
            
            # Log error to CSV
            log_error_to_csv(str(e), "TRADING_LOOP_ERROR", "trading_loop", "ERROR")
            
            # Update bot status
            bot_status['errors'].append(error_msg)
            bot_status['last_error'] = error_msg
            bot_status['last_update'] = format_cairo_time()
            
            if consecutive_errors >= max_consecutive_errors:
                print(f"💀 Maximum errors reached ({max_consecutive_errors}). Wolf hibernating.")
                bot_status['running'] = False
                bot_status['status'] = 'stopped_due_to_errors'
                break
            
            # Smart error recovery with exponential backoff
            sleep_time = min(error_sleep_time * (2 ** (consecutive_errors - 1)), 300)  # Max 5 minutes
            print(f"😴 Wolf resting for {sleep_time} seconds before retry...")
            time.sleep(sleep_time)
    
    print("\n🐺 === AI TRADING WOLF DEACTIVATED ===")
    bot_status['running'] = False
    bot_status['status'] = 'stopped'

def smart_portfolio_manager():
    """Advanced portfolio management with dynamic risk allocation"""
    try:
        if not client:
            return {"error": "API not connected"}
        
        account = client.get_account()
        balances = {b['asset']: float(b['free']) for b in account['balances'] if float(b['free']) > 0}
        
        # Calculate total portfolio value in USDT
        total_usdt_value = balances.get('USDT', 0)
        for asset, amount in balances.items():
            if asset != 'USDT' and amount > 0:
                try:
                    ticker = client.get_ticker(symbol=f"{asset}USDT")
                    price = float(ticker['price'])
                    total_usdt_value += amount * price
                except:
                    continue
        
        # Smart position sizing based on portfolio value and risk
        max_position_size = total_usdt_value * (config.RISK_PERCENTAGE / 100)
        
        # Adjust for volatility and consecutive losses
        volatility_adjustment = 1.0
        loss_adjustment = 1.0
        
        consecutive_losses = bot_status.get('consecutive_losses', 0)
        if consecutive_losses > 0:
            loss_adjustment = max(0.1, 1.0 - (consecutive_losses * 0.2))  # Reduce size by 20% per loss
        
        adjusted_position_size = max_position_size * volatility_adjustment * loss_adjustment
        
        portfolio_info = {
            'total_value_usdt': total_usdt_value,
            'max_position_size': max_position_size,
            'adjusted_position_size': adjusted_position_size,
            'risk_percentage': config.RISK_PERCENTAGE,
            'consecutive_losses': consecutive_losses,
            'loss_adjustment': loss_adjustment,
            'balances': balances,
            'portfolio_allocation': {}
        }
        
        # Calculate portfolio allocation percentages
        for asset, amount in balances.items():
            if asset == 'USDT':
                portfolio_info['portfolio_allocation'][asset] = (amount / total_usdt_value) * 100
            else:
                try:
                    ticker = client.get_ticker(symbol=f"{asset}USDT")
                    price = float(ticker['price'])
                    asset_value = amount * price
                    portfolio_info['portfolio_allocation'][asset] = (asset_value / total_usdt_value) * 100
                except:
                    portfolio_info['portfolio_allocation'][asset] = 0
        
        return portfolio_info
        
    except Exception as e:
        return {"error": f"Portfolio management error: {e}"}

# Flask Routes and Dashboard Functions
def stop_trading_bot():
    """Stop the trading bot"""
    bot_status['running'] = False
    bot_status['signal_scanning_active'] = False  # Deactivate signal scanning
    bot_status['next_signal_time'] = None  # Clear next signal time when stopped
    
    # Send Telegram notification for bot stop
    if TELEGRAM_AVAILABLE:
        try:
            notify_bot_status("STOPPED", "Manually stopped by user")
        except Exception as telegram_error:
            print(f"Telegram bot stop notification failed: {telegram_error}")

def start_trading_bot():
    """Start the trading bot in a separate thread"""
    try:
        if bot_status.get('running', False):
            print("⚠️ Trading bot is already running")
            return
            
        # Check if there's already a trading thread running
        import threading
        for thread in threading.enumerate():
            if thread.name == 'trading_loop_thread' and thread.is_alive():
                print("⚠️ Trading thread already exists and is running")
                bot_status['running'] = True  # Ensure status is consistent
                return
            
        # Only initialize client if not already connected
        if not bot_status.get('api_connected', False):
            print("🔧 Initializing API client...")
            if not initialize_client():
                print("❌ Failed to initialize API client; bot not started")
                log_error_to_csv("Failed to initialize API client on start", "CLIENT_ERROR", "start_trading_bot", "ERROR")
                return
        
        # Start trading loop in background thread with a unique name
        trading_thread = threading.Thread(target=trading_loop, daemon=True, name='trading_loop_thread')
        trading_thread.start()
        bot_status['running'] = True
        bot_status['status'] = 'running'
        print("✅ Trading bot started successfully")
        
        # Send Telegram notification for bot start
        if TELEGRAM_AVAILABLE:
            try:
                current_strategy = bot_status.get('trading_strategy', 'STRICT')
                notify_bot_status("STARTED", f"Strategy: {current_strategy}")
            except Exception as telegram_error:
                print(f"Telegram bot start notification failed: {telegram_error}")
                
    except Exception as e:
        print(f"❌ Failed to start trading bot: {e}")
        log_error_to_csv(str(e), "START_ERROR", "start_trading_bot", "ERROR")

@app.route('/download_logs')
def download_logs():
    """Create a zip file containing all CSV log files and send it to the user"""
    try:
        # Create an in-memory zip file
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Get all CSV files from logs directory
            logs_dir = Path('logs')
            if not logs_dir.exists():
                return jsonify({'error': 'No log files found'}), 404
                
            for csv_file in logs_dir.glob('*.csv'):
                if csv_file.exists():
                    # Add file to zip with relative path
                    zf.write(csv_file, csv_file.name)
        
        # Prepare the zip file for sending
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='trading_bot_logs.zip'
        )
    except Exception as e:
        print(f"Error creating log zip file: {e}")
        return jsonify({'error': 'Failed to create zip file'}), 500

@app.route('/')
def home():
    # Get current strategy for display
    current_strategy = bot_status.get('trading_strategy', 'STRICT')
    strategy_descriptions = {
        'STRICT': '🎯 Conservative strategy with strict rules to minimize risk',
        'MODERATE': '⚖️ Balanced strategy for more frequent trading opportunities',
        'ADAPTIVE': '🧠 Smart strategy that adapts to market conditions'
    }
    strategy_desc = strategy_descriptions.get(current_strategy, '')
    
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐺 CRYPTIX AI Trading Wolf</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 15px;
            color: #333;
        }
        
        .container {
            max-width: 420px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 24px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
            overflow: hidden;
            backdrop-filter: blur(20px);
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px 20px;
            text-align: center;
            position: relative;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .header-content {
            position: relative;
            z-index: 1;
        }
        
        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .subtitle {
            font-size: 0.9rem;
            opacity: 0.9;
            font-weight: 500;
        }
        
        .main-content {
            padding: 25px 20px;
        }
        
        /* Status Cards */
        .status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 25px;
        }
        
        .status-card {
            padding: 16px;
            border-radius: 16px;
            text-align: center;
            font-weight: 600;
            font-size: 0.85rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }
        
        .status-card:hover {
            transform: translateY(-2px);
        }
        
        .status-running {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status-stopped {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .status-connected {
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        
        .status-disconnected {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        .status-label {
            display: block;
            font-size: 0.75rem;
            opacity: 0.8;
            margin-bottom: 4px;
        }
        
        .status-value {
            font-size: 0.9rem;
            font-weight: 700;
        }
        
        /* Wolf Intelligence Section */
        .wolf-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 25px;
            border: 1px solid #dee2e6;
        }
        
        .wolf-title {
            text-align: center;
            font-size: 1.1rem;
            font-weight: 700;
            color: #495057;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        .wolf-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .wolf-card {
            padding: 12px;
            border-radius: 12px;
            text-align: center;
            font-size: 0.75rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .wolf-card .label {
            opacity: 0.8;
            margin-bottom: 4px;
            font-weight: 500;
        }
        
        .wolf-card .value {
            font-weight: 700;
            font-size: 0.85rem;
        }
        
        /* Trading Info Section */
        .trading-section {
            background: white;
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border: 1px solid #e9ecef;
        }
        
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #495057;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .info-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #f8f9fa;
        }
        
        .info-item:last-child {
            border-bottom: none;
        }
        
        .info-label {
            font-weight: 600;
            color: #6c757d;
            font-size: 0.85rem;
        }
        
        .info-value {
            font-weight: 700;
            color: #495057;
            font-size: 0.9rem;
        }
        
        .signal-buy { color: #28a745; }
        .signal-sell { color: #dc3545; }
        .signal-hold { color: #6c757d; }
        
        .countdown-timer {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            padding: 6px 12px;
            border-radius: 12px;
            font-family: 'SF Mono', Monaco, monospace;
            font-weight: 700;
            font-size: 0.85rem;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        }
        
        /* Strategy Section */
        .strategy-section {
            background: white;
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border: 1px solid #e9ecef;
        }
        
        .strategy-desc {
            text-align: center;
            color: #6c757d;
            font-size: 0.85rem;
            margin-bottom: 20px;
            line-height: 1.5;
        }
        
        .strategy-buttons {
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
        }
        
        .strategy-btn {
            padding: 14px;
            border-radius: 14px;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.85rem;
            text-align: center;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .strategy-btn.active {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            transform: scale(1.02);
        }
        
        .strategy-btn:not(.active) {
            background: #f8f9fa;
            color: #6c757d;
            border: 1px solid #dee2e6;
        }
        
        .strategy-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Controls */
        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 20px;
        }
        
        .btn {
            padding: 14px;
            border-radius: 14px;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.85rem;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .btn-start {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
        }
        
        .btn-stop {
            background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
            color: white;
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #6c757d 0%, #5a6268 100%);
            color: white;
        }
        
        .btn-warning {
            background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
            color: #212529;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            font-size: 0.75rem;
            color: #6c757d;
            border-top: 1px solid #f8f9fa;
        }
        
        .footer a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }
        
        /* Mobile Optimizations */
        @media (max-width: 480px) {
            body { padding: 10px; }
            .container { max-width: 100%; }
            .header { padding: 20px 15px; }
            .main-content { padding: 20px 15px; }
            .header h1 { font-size: 1.6rem; }
            .status-grid { 
                grid-template-columns: 1fr 1fr;
                gap: 8px; 
            }
            .wolf-grid { gap: 8px; }
            .controls { grid-template-columns: 1fr; }
        }
        
        /* Animations */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .countdown-timer {
            animation: pulse 2s infinite;
        }
        
        /* Responsive adjustments */
        @media (min-width: 481px) and (max-width: 768px) {
            .container { max-width: 480px; }
            .strategy-buttons { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>
                    🐺 CRYPTIX<br>Trading Wolf
                </h1>
                <div class="subtitle">Professional Trading Intelligence</div>
            </div>
        </div>
        
        <div class="main-content">
            <!-- Status Cards -->
            <div class="status-grid">
                <div class="status-card {{ 'status-running' if status.running else 'status-stopped' }}">
                    <div class="status-label">Bot Status</div>
                    <div class="status-value">{{ 'Running' if status.running else 'Stopped' }}</div>
                </div>
                <div class="status-card {{ 'status-connected' if status.api_connected else 'status-disconnected' }}">
                    <div class="status-label">API Status</div>
                    <div class="status-value">{{ 'Connected' if status.api_connected else 'Disconnected' }}</div>
                </div>
            </div>
            
            <!-- AI Wolf Intelligence -->
            <div class="wolf-section">
                <div class="wolf-title">
                    🧠 AI Wolf Intelligence
                </div>
                <div class="wolf-grid">
                    <div class="wolf-card" style="background: {{ '#d4edda' if status.get('market_regime') == 'EXTREME' else '#d1ecf1' if status.get('market_regime') == 'VOLATILE' else '#fff3cd' if status.get('market_regime') == 'QUIET' else '#e9ecef' }}; 
                                               color: {{ '#155724' if status.get('market_regime') == 'EXTREME' else '#0c5460' if status.get('market_regime') == 'VOLATILE' else '#856404' if status.get('market_regime') == 'QUIET' else '#495057' }};">
                        <div class="label">Market Regime</div>
                        <div class="value">{{ status.get('market_regime', 'NORMAL') }}</div>
                    </div>
                    <div class="wolf-card" style="background: {{ '#f8d7da' if status.get('hunting_mode') else '#e9ecef' }}; 
                                               color: {{ '#721c24' if status.get('hunting_mode') else '#495057' }};">
                        <div class="label">Wolf Mode</div>
                        <div class="value">{{ 'HUNTING 🎯' if status.get('hunting_mode') else 'PASSIVE' }}</div>
                    </div>
                    <div class="wolf-card" style="background: #e9ecef; color: #495057;">
                        <div class="label">Scan Interval</div>
                        <div class="value">{{ (status.get('signal_interval', 900) // 60) }}min</div>
                    </div>
                    <div class="wolf-card" style="background: #e9ecef; color: #495057;">
                        <div class="label">Next Scan</div>
                        <div class="value countdown-timer">{{ time_remaining }}</div>
                    </div>
                </div>
            </div>
            
            <!-- Trading Information -->
            <div class="trading-section">
                <div class="section-title">📊 Trading Status</div>
                <div class="info-item">
                    <span class="info-label">Last Signal</span>
                    <span class="info-value signal-{{ status.last_signal.lower() }}">{{ status.last_signal }}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Last Scan</span>
                    <span class="info-value">{{ status.last_scan_time.strftime('%H:%M:%S') if status.last_scan_time else 'Never' }}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Current Symbol</span>
                    <span class="info-value">{{ status.current_symbol }}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Current Price</span>
                    <span class="info-value">${{ "{:,.2f}".format(status.last_price) if status.last_price else 'N/A' }}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Total Revenue</span>
                    <span class="info-value" style="color: {{ '#28a745' if status.trading_summary.total_revenue > 0 else '#dc3545' if status.trading_summary.total_revenue < 0 else '#6c757d' }}">
                        ${{ "{:,.2f}".format(status.trading_summary.total_revenue) }}
                    </span>
                </div>
                <div class="info-item">
                    <span class="info-label">Win Rate</span>
                    <span class="info-value">{{ "{:.1f}".format(status.trading_summary.win_rate) }}%</span>
                </div>
            </div>
            
            <!-- Strategy Section -->
            <div class="strategy-section">
                <div class="section-title">🎯 Trading Strategy</div>
                <div class="strategy-desc">{{ strategy_desc }}</div>
                <div class="strategy-buttons">
                    <a href="/strategy/strict" class="strategy-btn {{ 'active' if status.trading_strategy == 'STRICT' else '' }}">
                        🎯 <span>Strict - Conservative Trading</span>
                    </a>
                    <a href="/strategy/moderate" class="strategy-btn {{ 'active' if status.trading_strategy == 'MODERATE' else '' }}">
                        ⚖️ <span>Moderate - Balanced Approach</span>
                    </a>
                    <a href="/strategy/adaptive" class="strategy-btn {{ 'active' if status.trading_strategy == 'ADAPTIVE' else '' }}">
                        🧠 <span>Adaptive - Smart & Dynamic</span>
                    </a>
                </div>
            </div>
            
            <!-- Controls -->
            <div class="controls">
                <a href="/start" class="btn btn-start">🚀 Start Bot</a>
                <a href="/stop" class="btn btn-stop">🛑 Stop Bot</a>
            </div>
            
            <div style="margin-bottom: 15px;">
                <a href="/logs" class="btn btn-secondary" style="width: 100%; display: block;">📋 View Logs</a>
            </div>
        </div>
        
        <div class="footer">
            <div style="margin-bottom: 10px;">
                <strong>Cairo Time: {{ current_time }}</strong>
            </div>
            Auto-refresh every 30s • <a href="javascript:location.reload()">Manual Refresh</a>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {
            location.reload();
        }, 30000);
        
        // Add touch feedback for mobile
        document.querySelectorAll('.btn, .strategy-btn').forEach(button => {
            button.addEventListener('touchstart', function() {
                this.style.transform = 'scale(0.98)';
            });
            button.addEventListener('touchend', function() {
                this.style.transform = '';
            });
        });
    </script>
</body>
</html>
    """, status=bot_status, current_time=format_cairo_time(), time_remaining=get_time_remaining_for_next_signal(), strategy_desc=strategy_desc)


@app.route('/start')
def start():
    """Manual start route"""
    if not bot_status.get('running', False):
        try:
            start_trading_bot()
            return redirect('/')
        except Exception as e:
            bot_status['errors'].append(f"Failed to start bot: {str(e)}")
            return redirect('/')
    else:
        print("⚠️ Bot is already running")
        return redirect('/')

@app.route('/stop')
def stop():
    """Manual stop route"""
    try:
        stop_trading_bot()  # Call the proper stop function
        print("Bot manually stopped via web interface")
        return redirect('/')
    except Exception as e:
        print(f"Error stopping bot: {e}")
        return f"Error stopping bot: {e}"

@app.route('/force_scan')
def force_scan():
    """Force an immediate signal scan"""
    try:
        if not bot_status.get('running', False):
            return jsonify({'error': 'Bot is not running'}), 400
            
        if not bot_status.get('api_connected', False):
            return jsonify({'error': 'API not connected'}), 400
        
        print("🚀 Manual scan triggered from web interface")
        
        # Force immediate scan by setting next_signal_time to now
        bot_status['next_signal_time'] = get_cairo_time()
        
        return jsonify({
            'success': True, 
            'message': 'Scan triggered successfully',
            'next_scan_time': format_cairo_time(bot_status['next_signal_time'])
        })
        
    except Exception as e:
        print(f"Error triggering manual scan: {e}")
        return jsonify({'error': f'Failed to trigger scan: {str(e)}'}), 500

@app.route('/strategy/<name>')
def set_strategy(name):
    """Switch trading strategy"""
    try:
        if name.upper() in ['STRICT', 'MODERATE', 'ADAPTIVE']:
            previous_strategy = bot_status.get('trading_strategy', 'STRICT')
            new_strategy = name.upper()
            
            # Update bot status
            bot_status['trading_strategy'] = new_strategy
            
            # Log the strategy change
            log_error_to_csv(
                f"Strategy changed from {previous_strategy} to {new_strategy}",
                "STRATEGY_CHANGE",
                "set_strategy",
                "INFO"
            )
            
            # Print debug info
            print(f"Strategy changed: {previous_strategy} -> {new_strategy}")
            print(f"Current bot status: {bot_status}")
            
            return redirect('/')
        else:
            log_error_to_csv(
                f"Invalid strategy name: {name}",
                "STRATEGY_ERROR",
                "set_strategy",
                "ERROR"
            )
            return "Invalid strategy name", 400
    except Exception as e:
        error_msg = f"Error changing strategy: {str(e)}"
        log_error_to_csv(error_msg, "STRATEGY_ERROR", "set_strategy", "ERROR")
        print(error_msg)
        return error_msg, 500

@app.route('/api/status')
def api_status():
    """JSON API endpoint for bot status"""
    return jsonify(bot_status)

@app.route('/api/balances')
def api_balances():
    """JSON API endpoint for account balances"""
    try:
        balance_summary = get_account_balances_summary()
        return jsonify(balance_summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logs')
def view_logs():
    """View CSV logs interface"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📋 CRYPTIX Logs</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 15px;
            color: #333;
        }
        
        .container {
            max-width: 420px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 24px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
            overflow: hidden;
            backdrop-filter: blur(20px);
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px 20px;
            text-align: center;
            position: relative;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .header-content {
            position: relative;
            z-index: 1;
        }
        
        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .subtitle {
            font-size: 0.9rem;
            opacity: 0.9;
            font-weight: 500;
        }
        
        .main-content {
            padding: 25px 20px;
        }
        
        .back-link {
            display: inline-block;
            margin-bottom: 25px;
            padding: 12px 20px;
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            text-decoration: none;
            border-radius: 14px;
            font-size: 0.9rem;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
            transition: all 0.3s ease;
            width: 100%;
            text-align: center;
            box-sizing: border-box;
        }
        
        .back-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(40, 167, 69, 0.4);
        }
        
        /* Log Files Section */
        .log-section {
            background: white;
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border: 1px solid #e9ecef;
        }
        
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #495057;
            margin-bottom: 15px;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        .log-links {
            display: grid;
            grid-template-columns: 1fr;
            gap: 12px;
        }
        
        .log-links a {
            padding: 16px;
            border-radius: 14px;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.85rem;
            text-align: center;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .log-links a:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .log-links a.download {
            background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
            color: #212529;
        }
        
        .log-links a.download:hover {
            box-shadow: 0 4px 12px rgba(255, 193, 7, 0.3);
        }
        
        /* Stats Section */
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 12px;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #f8f9fa;
        }
        
        .stat-item:last-child {
            border-bottom: none;
        }
        
        .stat-label {
            font-weight: 600;
            color: #6c757d;
            font-size: 0.85rem;
        }
        
        .stat-value {
            font-weight: 700;
            color: #495057;
            font-size: 0.9rem;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            font-size: 0.75rem;
            color: #6c757d;
            border-top: 1px solid #f8f9fa;
        }
        
        .footer a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }
        
        /* Mobile Optimizations */
        @media (max-width: 480px) {
            body { padding: 10px; }
            .container { max-width: 100%; }
            .header { padding: 20px 15px; }
            .main-content { padding: 20px 15px; }
            .header h1 { font-size: 1.6rem; }
        }
        
        /* Touch feedback */
        .log-links a {
            -webkit-tap-highlight-color: transparent;
        }
        
        /* Responsive adjustments */
        @media (min-width: 481px) and (max-width: 768px) {
            .container { max-width: 480px; }
            .log-links { grid-template-columns: 1fr 1fr; }
            .log-links a.download { grid-column: 1 / -1; }
        }
        
        @media (min-width: 769px) {
            .container { max-width: 600px; }
            .log-links { grid-template-columns: 1fr 1fr; }
            .log-links a.download { grid-column: 1 / -1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>
                    📋 CRYPTIX<br>Trading Logs
                </h1>
                <div class="subtitle">Activity & Performance Monitoring</div>
            </div>
        </div>
        
        <div class="main-content">
            <a href="/" class="back-link">← Back to Dashboard</a>
            
            <!-- Log Files Section -->
            <div class="log-section">
                <div class="section-title">
                    📊 Available Log Files
                </div>
                <div class="log-links">
                    <a href="/logs/trades">
                        📊 <span>Trade History</span>
                    </a>
                    <a href="/logs/signals">
                        📈 <span>Signal History</span>
                    </a>
                    <a href="/logs/performance">
                        📉 <span>Daily Performance</span>
                    </a>
                    <a href="/logs/errors">
                        ❌ <span>Error Log</span>
                    </a>
                    <a href="/download_logs" class="download">
                        💾 <span>Download All CSV Files</span>
                    </a>
                </div>
            </div>
            
            <!-- Quick Stats Section -->
            <div class="log-section">
                <div class="section-title">
                    📈 Quick Statistics
                </div>
                <div class="stats-grid">
                    <div class="stat-item">
                        <span class="stat-label">Total Trades Logged</span>
                        <span class="stat-value">{{ total_trades }}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">CSV Files Location</span>
                        <span class="stat-value">/logs/</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Last Updated</span>
                        <span class="stat-value">{{ current_time }}</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <div style="margin-bottom: 10px;">
                <strong>Cairo Time: {{ current_time }}</strong>
            </div>
            <a href="javascript:location.reload()">Refresh Data</a>
        </div>
    </div>
    
    <script>
        // Add touch feedback for mobile
        document.querySelectorAll('.log-links a, .back-link').forEach(button => {
            button.addEventListener('touchstart', function() {
                this.style.transform = 'scale(0.98)';
            });
            button.addEventListener('touchend', function() {
                this.style.transform = '';
            });
        });
    </script>
</body>
</html>
    """, total_trades=len(get_csv_trade_history()), current_time=format_cairo_time())

@app.route('/logs/trades')
def view_trade_logs():
    """View trade history CSV"""
    trades = get_csv_trade_history(30)  # Last 30 days
    
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trade History</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            background: #f5f5f5;
            padding: 10px;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            padding: 15px; 
            border-radius: 10px;
            overflow-x: hidden;
        }
        .table-wrapper {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            margin: 0 -15px;
            padding: 0 15px;
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 15px; 
            font-size: 0.85rem;
            min-width: 800px;
        }
        th, td { 
            padding: 10px 12px; 
            border: 1px solid #ddd; 
            text-align: left;
            white-space: nowrap;
        }
        th { 
            background: #f8f9fa; 
            font-weight: bold; 
            position: sticky; 
            top: 0;
            z-index: 1;
        }
        tr:nth-child(even) { background: #f9f9f9; }
        .back-link { 
            display: inline-block; 
            margin-bottom: 20px; 
            padding: 12px 20px; 
            background: #28a745; 
            color: white; 
            text-decoration: none; 
            border-radius: 5px;
            font-size: 0.9rem;
        }
        .back-link:hover {
            background: #218838;
        }
        h1 {
            font-size: 1.8rem;
            margin: 15px 0;
        }
        .status-success { background: #d4edda; }
        .status-simulated { background: #d1ecf1; }
        .status-error { background: #f8d7da; }
        .signal-buy { color: #28a745; font-weight: bold; }
        .signal-sell { color: #dc3545; font-weight: bold; }
        .signal-hold { color: #ffc107; font-weight: bold; }
        
        @media (max-width: 768px) {
            body {
                padding: 5px;
            }
            .container {
                padding: 10px;
            }
            h1 {
                font-size: 1.5rem;
                margin: 10px 0;
            }
            table {
                font-size: 0.8rem;
            }
            th, td {
                padding: 8px 10px;
            }
            .back-link {
                width: 100%;
                text-align: center;
                box-sizing: border-box;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/logs" class="back-link">← Back to Logs</a>
        <h1>📊 Trade History (Last 30 Days - Newest First)</h1>
        
        {% if trades %}
        <table>
            <thead>
                <tr>
                    <th>Time (Cairo)</th>
                    <th>Signal</th>
                    <th>Symbol</th>
                    <th>Quantity</th>
                    <th>Price</th>
                    <th>Value</th>
                    <th>Fee</th>
                    <th>Status</th>
                    <th>RSI</th>
                    <th>MACD</th>
                    <th>Sentiment</th>
                    <th>P&L</th>
                </tr>
            </thead>
            <tbody>
                {% for trade in trades %}
                <tr class="status-{{ trade.status }}">
                    <td>{{ trade.cairo_time }}</td>
                    <td class="signal-{{ trade.signal.lower() }}">{{ trade.signal }}</td>
                    <td>{{ trade.symbol }}</td>
                    <td>{{ "%.6f"|format(trade.quantity) }}</td>
                    <td>${{ "%.2f"|format(trade.price) }}</td>
                    <td>${{ "%.2f"|format(trade.value) }}</td>
                    <td>${{ "%.4f"|format(trade.fee) }}</td>
                    <td>{{ trade.status }}</td>
                    <td>{{ "%.1f"|format(trade.rsi) }}</td>
                    <td>{{ trade.macd_trend }}</td>
                    <td>{{ trade.sentiment }}</td>
                    <td>${{ "%.2f"|format(trade.profit_loss) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p>No trades found in the last 30 days.</p>
        {% endif %}
    </div>
</body>
</html>
    """, trades=trades)

@app.route('/logs/signals')
def view_signal_logs():
    """View signal history CSV"""
    try:
        csv_files = setup_csv_logging()
        
        if not csv_files['signals'].exists():
            signals = []
        else:
            df = pd.read_csv(csv_files['signals'])
            # Sort by timestamp column to show newest first, then get last 100
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp', ascending=False).head(100)
            else:
                # Fallback: get last 100 signals and reverse order
                df = df.tail(100).iloc[::-1]
            signals = df.to_dict('records')
        
        return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal History</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.8rem; }
        th, td { padding: 6px 8px; border: 1px solid #ddd; text-align: left; }
        th { background: #f8f9fa; font-weight: bold; position: sticky; top: 0; }
        tr:nth-child(even) { background: #f9f9f9; }
        .back-link { display: inline-block; margin-bottom: 20px; padding: 10px 20px; background: #28a745; color: white; text-decoration: none; border-radius: 5px; }
        .signal-buy { color: #28a745; font-weight: bold; }
        .signal-sell { color: #dc3545; font-weight: bold; }
        .signal-hold { color: #ffc107; font-weight: bold; }
        .sentiment-bullish { color: #28a745; }
        .sentiment-bearish { color: #dc3545; }
        .sentiment-neutral { color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <a href="/logs" class="back-link">← Back to Logs</a>
        <h1>📈 Signal History (Latest 100 Signals - Newest First)</h1>
        
        {% if signals %}
        <table>
            <thead>
                <tr>
                    <th>Time (Cairo)</th>
                    <th>Signal</th>
                    <th>Symbol</th>
                    <th>Price</th>
                    <th>RSI</th>
                    <th>MACD</th>
                    <th>MACD Trend</th>
                    <th>Sentiment</th>
                    <th>SMA5</th>
                    <th>SMA20</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody>
                {% for signal in signals %}
                <tr>
                    <td>{{ signal.cairo_time }}</td>
                    <td class="signal-{{ signal.signal.lower() }}">{{ signal.signal }}</td>
                    <td>{{ signal.symbol }}</td>
                    <td>${{ "%.2f"|format(signal.price) }}</td>
                    <td>{{ "%.1f"|format(signal.rsi) }}</td>
                    <td>{{ "%.6f"|format(signal.macd) }}</td>
                    <td>{{ signal.macd_trend }}</td>
                    <td class="sentiment-{{ signal.sentiment }}">{{ signal.sentiment }}</td>
                    <td>${{ "%.2f"|format(signal.sma5) }}</td>
                    <td>${{ "%.2f"|format(signal.sma20) }}</td>
                    <td style="font-size: 0.7rem;">{{ signal.reason[:100] }}...</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p>No signals found.</p>
        {% endif %}
    </div>
</body>
</html>
        """, signals=signals)
        
    except Exception as e:
        return f"Error loading signal logs: {e}"

@app.route('/logs/performance')
def view_performance_logs():
    """View daily performance CSV in simple format"""
    try:
        csv_files = setup_csv_logging()
        performance_data = []
        
        if csv_files['performance'].exists():
            df = pd.read_csv(csv_files['performance'])
            # Convert to list of dictionaries, newest first (reverse order)
            performance_data = df.iloc[::-1].to_dict('records')
        
        return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Performance</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            background: #f5f5f5;
            padding: 10px;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            padding: 15px; 
            border-radius: 10px;
            overflow-x: hidden;
        }
        .table-wrapper {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            margin: 0 -15px;
            padding: 0 15px;
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 15px; 
            font-size: 0.85rem;
            min-width: 800px;
        }
        th, td { 
            padding: 10px 12px; 
            border: 1px solid #ddd; 
            text-align: left;
            white-space: nowrap;
        }
        th { 
            background: #f8f9fa; 
            font-weight: bold; 
            position: sticky; 
            top: 0;
            z-index: 1;
        }
        tr:nth-child(even) { background: #f9f9f9; }
        .back-link { 
            display: inline-block; 
            margin-bottom: 20px; 
            padding: 12px 20px; 
            background: #28a745; 
            color: white; 
            text-decoration: none; 
            border-radius: 5px;
            font-size: 0.9rem;
        }
        .back-link:hover {
            background: #218838;
        }
        h1 {
            font-size: 1.8rem;
            margin: 15px 0;
        }
        .positive { color: #28a745; font-weight: bold; }
        .negative { color: #dc3545; font-weight: bold; }
        .neutral { color: #6c757d; font-weight: bold; }
        
        @media (max-width: 768px) {
            body {
                padding: 5px;
            }
            .container {
                padding: 10px;
            }
            h1 {
                font-size: 1.5rem;
                margin: 10px 0;
            }
            table {
                font-size: 0.8rem;
            }
            th, td {
                padding: 8px 10px;
            }
            .back-link {
                width: 100%;
                text-align: center;
                box-sizing: border-box;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/logs" class="back-link">← Back to Logs</a>
        <h1>📊 Daily Performance (CSV Format)</h1>
        
        {% if performance_data %}
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Total Trades</th>
                        <th>Successful Trades</th>
                        <th>Failed Trades</th>
                        <th>Win Rate (%)</th>
                        <th>Total Revenue</th>
                        <th>Daily P&L</th>
                        <th>Total Volume</th>
                        <th>Max Drawdown</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in performance_data %}
                    <tr>
                        <td>{{ row.date }}</td>
                        <td>{{ row.total_trades }}</td>
                        <td>{{ row.successful_trades }}</td>
                        <td>{{ row.failed_trades }}</td>
                        <td class="{{ 'positive' if row.win_rate > 60 else 'negative' if row.win_rate < 40 else 'neutral' }}">
                            {{ "%.1f"|format(row.win_rate) }}%
                        </td>
                        <td class="{{ 'positive' if row.total_revenue > 0 else 'negative' if row.total_revenue < 0 else 'neutral' }}">
                            ${{ "%.2f"|format(row.total_revenue) }}
                        </td>
                        <td class="{{ 'positive' if row.daily_pnl > 0 else 'negative' if row.daily_pnl < 0 else 'neutral' }}">
                            ${{ "%.2f"|format(row.daily_pnl) }}
                        </td>
                        <td>${{ "%.2f"|format(row.total_volume) }}</td>
                        <td>{{ "%.2f"|format(row.max_drawdown) if row.max_drawdown else '0.00' }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% else %}
        <p>No daily performance data found. Performance data will appear here once the bot starts trading and logging daily summaries.</p>
        {% endif %}
    </div>
</body>
</html>
        """, performance_data=performance_data)
        
    except Exception as e:
        return f"Error loading performance logs: {e}"

@app.route('/logs/errors')
def view_error_logs():
    """View error log CSV"""
    try:
        csv_files = setup_csv_logging()
        
        if not csv_files['errors'].exists():
            errors = []
        else:
            df = pd.read_csv(csv_files['errors'])
            # Sort by timestamp to show newest first, then get last 50
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                df = df.sort_values('timestamp', ascending=False).head(50)
            else:
                # Fallback: get last 50 errors and reverse order to show newest first
                df = df.tail(50).iloc[::-1]
            errors = df.to_dict('records')
        
        return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error Log</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.8rem; }
        th, td { padding: 6px 8px; border: 1px solid #ddd; text-align: left; }
        th { background: #f8f9fa; font-weight: bold; position: sticky; top: 0; }
        tr:nth-child(even) { background: #f9f9f9; }
        .back-link { display: inline-block; margin-bottom: 20px; padding: 10px 20px; background: #28a745; color: white; text-decoration: none; border-radius: 5px; }
        .error { background: #f8d7da; }
        .warning { background: #fff3cd; }
        .critical { background: #f5c6cb; }
    </style>
</head>
<body>
    <div class="container">
        <a href="/logs" class="back-link">← Back to Logs</a>
        <h1>❌ Error Log (Last 50 Errors - Newest First)</h1>
        
        {% if errors %}
        <table>
            <thead>
                <tr>
                    <th>Time (Cairo)</th>
                    <th>Severity</th>
                    <th>Error Type</th>
                    <th>Function</th>
                    <th>Error Message</th>
                    <th>Bot Status</th>
                </tr>
            </thead>
            <tbody>
                {% for error in errors %}
                <tr class="{{ error.severity.lower() }}">
                    <td>{{ error.cairo_time }}</td>
                    <td>{{ error.severity }}</td>
                    <td>{{ error.error_type }}</td>
                    <td>{{ error.function_name }}</td>
                    <td style="max-width: 300px; word-wrap: break-word;">{{ error.error_message }}</td>
                    <td>{{ error.bot_status }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p>No errors found.</p>
        {% endif %}
    </div>
</body>
</html>
        """, errors=errors)
        
    except Exception as e:
        return f"Error loading error logs: {e}"

@app.route('/ping')
def ping():
    """Simple ping endpoint for uptime monitoring"""
    return {"status": "alive", "timestamp": format_cairo_time()}, 200

@app.route('/health')
def health():
    """Comprehensive health check with system and bot status"""
    try:
        health_data = {
            'status': 'healthy',
            'timestamp': format_cairo_time(),
            'bot_running': bot_status.get('running', False),
            'api_connected': bot_status.get('api_connected', False),
            'last_update': bot_status.get('last_update', 'Never'),
            'error_count': len(bot_status.get('errors', [])),
            'consecutive_errors': bot_status.get('consecutive_errors', 0),
            'uptime_seconds': (get_cairo_time() - bot_status.get('start_time', get_cairo_time())).total_seconds(),
            'account_type': bot_status.get('account_type', 'Unknown'),
            'can_trade': bot_status.get('can_trade', False)
        }
        
        # Environment variable check (for debugging Render deployment)
        env_check = {
            'api_key_present': bool(os.getenv("API_KEY")),
            'api_secret_present': bool(os.getenv("API_SECRET")),
            'api_key_length': len(os.getenv("API_KEY", "")),
            'api_secret_length': len(os.getenv("API_SECRET", "")),
            'environment': os.getenv("RENDER", "local"),  # Render sets this automatically
        }
        
        if env_check['api_key_present']:
            api_key_val = os.getenv("API_KEY", "")
            env_check['api_key_preview'] = f"{api_key_val[:8]}...{api_key_val[-4:]}" if len(api_key_val) >= 12 else "invalid"
        
        health_data['environment'] = env_check
        
        # Try to get memory info if psutil is available
        try:
            import psutil  # Optional dependency for system metrics
            process = psutil.Process()
            health_data['memory_usage_mb'] = round(process.memory_info().rss / 1024 / 1024, 2)
            health_data['cpu_percent'] = round(process.cpu_percent(), 2)
        except ImportError:
            health_data['memory_usage_mb'] = 'unknown'
            health_data['cpu_percent'] = 'unknown'
        
        # Determine overall health status
        if not bot_status.get('api_connected', False):
            health_data['status'] = 'degraded'
        elif bot_status.get('consecutive_errors', 0) >= 3:
            health_data['status'] = 'warning'
        elif not bot_status.get('running', False):
            health_data['status'] = 'stopped'
            
        return jsonify(health_data)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': format_cairo_time()
        }), 500

# ==================== POSITION REBALANCING FUNCTIONS ====================

def get_position_rsi(symbol, period=14):
    """Get current RSI for a specific trading pair"""
    try:
        if not client:
            return None
            
        # Get klines data for RSI calculation
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=period + 10)
        if not klines or len(klines) < period:
            print(f"⚠️ Insufficient data for RSI calculation for {symbol}")
            return None
            
        # Extract closing prices
        prices = [float(kline[4]) for kline in klines]
        rsi = calculate_rsi(prices, period)
        
        print(f"📊 {symbol} RSI({period}): {rsi:.1f}")
        return rsi
        
    except Exception as e:
        print(f"❌ Error calculating RSI for {symbol}: {e}")
        return None

def detect_dust_positions(min_usdt_value=5.0):
    """Detect small positions that should be liquidated to USDT"""
    try:
        if not client:
            return []
            
        account_info = client.get_account()
        dust_positions = []
        
        print(f"\n🔍 Scanning for dust positions (< ${min_usdt_value:.2f})...")
        
        for balance in account_info['balances']:
            asset = balance['asset']
            free_balance = float(balance['free'])
            total_balance = free_balance + float(balance['locked'])
            
            # Skip USDT and positions with zero balance
            if asset == 'USDT' or total_balance <= 0:
                continue
                
            # Try to get USDT value
            usdt_value = 0
            try:
                if asset in ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOT']:
                    ticker = client.get_ticker(symbol=f"{asset}USDT")
                    price = float(ticker['lastPrice'])
                    usdt_value = total_balance * price
                    
                    if 0 < usdt_value < min_usdt_value:
                        dust_positions.append({
                            'asset': asset,
                            'symbol': f"{asset}USDT",
                            'quantity': free_balance,
                            'total_quantity': total_balance,
                            'usdt_value': usdt_value,
                            'price': price
                        })
                        print(f"   💨 {asset}: {total_balance:.8f} (~${usdt_value:.2f})")
            except Exception as e:
                print(f"   ⚠️ Could not check {asset}: {e}")
                continue
                
        return dust_positions
        
    except Exception as e:
        print(f"❌ Error detecting dust positions: {e}")
        return []

def sell_partial_position(symbol, percentage=50.0, reason="Partial profit taking"):
    """Sell a percentage of an existing position"""
    try:
        if not client:
            return {"success": False, "error": "Client not initialized"}
            
        base_asset = symbol.replace('USDT', '')
        
        # Get current balance
        account_info = client.get_account()
        current_balance = 0
        for balance in account_info['balances']:
            if balance['asset'] == base_asset:
                current_balance = float(balance['free'])
                break
                
        if current_balance <= 0:
            return {"success": False, "error": f"No {base_asset} balance to sell"}
            
        # Calculate sell quantity
        sell_quantity = current_balance * (percentage / 100.0)
        
        # Get symbol info for precision
        symbol_info = client.get_symbol_info(symbol)
        if not symbol_info:
            return {"success": False, "error": f"Could not get symbol info for {symbol}"}
            
        # Find lot size filter
        lot_size_filter = None
        for f in symbol_info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                lot_size_filter = f
                break
                
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            sell_quantity = round(sell_quantity / step_size) * step_size
            
        # Check minimum quantity
        min_qty = float(lot_size_filter['minQty']) if lot_size_filter else 0.001
        if sell_quantity < min_qty:
            return {"success": False, "error": f"Quantity {sell_quantity} below minimum {min_qty}"}
            
        print(f"\n🎯 Partial Sell Order - {reason}")
        print(f"   Symbol: {symbol}")
        print(f"   Current Balance: {current_balance:.8f} {base_asset}")
        print(f"   Sell Percentage: {percentage}%")
        print(f"   Sell Quantity: {sell_quantity:.8f} {base_asset}")
        
        # Execute sell order
        order = client.order_market_sell(symbol=symbol, quantity=sell_quantity)
        
        # Calculate order details
        avg_price = float(order['fills'][0]['price']) if order['fills'] else 0
        total_value = float(order['cummulativeQuoteQty'])
        total_fee = sum([float(fill['commission']) for fill in order['fills']])
        
        result = {
            "success": True,
            "order_id": order.get('orderId'),
            "symbol": symbol,
            "quantity": sell_quantity,
            "price": avg_price,
            "value": total_value,
            "fee": total_fee,
            "percentage": percentage,
            "reason": reason,
            "timestamp": format_cairo_time()
        }
        
        print(f"✅ Partial sell executed successfully!")
        print(f"   Order ID: {result['order_id']}")
        print(f"   Price: ${avg_price:.4f}")
        print(f"   Value: ${total_value:.2f}")
        print(f"   Fee: ${total_fee:.4f}")
        
        # Log the trade
        log_trade_to_csv(
            action="SELL_PARTIAL",
            symbol=symbol,
            quantity=sell_quantity,
            price=avg_price,
            value=total_value,
            fee=total_fee,
            additional_data={
                'percentage': percentage,
                'reason': reason,
                'order_type': 'partial_sell'
            }
        )
        
        # Send Telegram notification if available
        if TELEGRAM_AVAILABLE:
            notify_trade(
                action="SELL_PARTIAL",
                symbol=symbol,
                quantity=sell_quantity,
                price=avg_price,
                additional_info=f"{percentage}% partial sell - {reason}"
            )
            
        return result
        
    except Exception as e:
        error_msg = f"Error in partial sell for {symbol}: {e}"
        print(f"❌ {error_msg}")
        log_error_to_csv(error_msg, "PARTIAL_SELL_ERROR", "sell_partial_position", "ERROR")
        return {"success": False, "error": str(e)}

def liquidate_dust_position(dust_position):
    """Liquidate a single dust position to USDT"""
    try:
        if not client:
            return {"success": False, "error": "Client not initialized"}
            
        asset = dust_position['asset']
        symbol = dust_position['symbol']
        quantity = dust_position['quantity']
        
        print(f"\n💨 Liquidating dust position: {asset}")
        print(f"   Quantity: {quantity:.8f} {asset}")
        print(f"   Est. Value: ${dust_position['usdt_value']:.2f}")
        
        # Execute market sell
        order = client.order_market_sell(symbol=symbol, quantity=quantity)
        
        # Calculate order details
        avg_price = float(order['fills'][0]['price']) if order['fills'] else 0
        total_value = float(order['cummulativeQuoteQty'])
        total_fee = sum([float(fill['commission']) for fill in order['fills']])
        
        result = {
            "success": True,
            "order_id": order.get('orderId'),
            "asset": asset,
            "symbol": symbol,
            "quantity": quantity,
            "price": avg_price,
            "value": total_value,
            "fee": total_fee,
            "timestamp": format_cairo_time()
        }
        
        print(f"✅ Dust liquidation successful!")
        print(f"   Received: ${total_value:.2f} USDT")
        print(f"   Fee: ${total_fee:.4f}")
        
        # Log the trade
        log_trade_to_csv(
            action="LIQUIDATE_DUST",
            symbol=symbol,
            quantity=quantity,
            price=avg_price,
            value=total_value,
            fee=total_fee,
            additional_data={
                'dust_value': dust_position['usdt_value'],
                'order_type': 'dust_liquidation'
            }
        )
        
        return result
        
    except Exception as e:
        error_msg = f"Error liquidating dust {dust_position['asset']}: {e}"
        print(f"❌ {error_msg}")
        log_error_to_csv(error_msg, "DUST_LIQUIDATION_ERROR", "liquidate_dust_position", "ERROR")
        return {"success": False, "error": str(e)}

def execute_position_rebalancing():
    """Execute comprehensive position rebalancing based on market conditions"""
    try:
        print("\n" + "="*60)
        print("🔄 POSITION REBALANCING ANALYSIS")
        print("="*60)
        
        rebalancing_results = {
            "timestamp": format_cairo_time(),
            "partial_sells": [],
            "dust_liquidations": [],
            "errors": [],
            "total_freed_usdt": 0
        }
        
        # 1. Check ALL assets with balances for overbought conditions
        if not client:
            rebalancing_results["errors"].append("Client not initialized")
            return rebalancing_results
            
        # Use existing balance summary function to get all assets with balances
        balance_summary = get_account_balances_summary()
        if "error" in balance_summary:
            rebalancing_results["errors"].append(f"Balance summary error: {balance_summary['error']}")
            return rebalancing_results
            
        # Filter to monitored assets with meaningful balances
        monitored_assets = config.REBALANCING.get('assets_to_monitor', [])
        assets_with_balance = []
        
        for asset, balance_info in balance_summary['balances'].items():
            if asset in monitored_assets and balance_info['free'] > 0:
                # Get current price for value check
                try:
                    if asset != 'USDT':  # Skip USDT itself
                        ticker = client.get_ticker(symbol=f"{asset}USDT")
                        price = float(ticker['lastPrice'])
                        usdt_value = balance_info['total'] * price
                        
                        # Only include assets with meaningful value (>$1)
                        if usdt_value >= 1.0:
                            assets_with_balance.append({
                                'asset': asset,
                                'balance': balance_info['free'],
                                'symbol': f"{asset}USDT",
                                'usdt_value': usdt_value
                            })
                except Exception as e:
                    print(f"⚠️ Could not get price for {asset}: {e}")
                    continue
        
        print(f"\n📊 Found {len(assets_with_balance)} assets with balances to check:")
        for asset_info in assets_with_balance:
            print(f"   • {asset_info['asset']}: {asset_info['balance']:.8f} (~${asset_info['usdt_value']:.2f})")
            
        # Check each asset for overbought conditions
        for asset_info in assets_with_balance:
            asset = asset_info['asset']
            symbol = asset_info['symbol']
            balance = asset_info['balance']
            
            print(f"\n🔍 Checking {asset} for overbought conditions...")
            
            # Get RSI for this asset
            asset_rsi = get_position_rsi(symbol, 14)
            
            if asset_rsi is None:
                print(f"⚠️ Could not get RSI for {asset}")
                continue
                
            # Check against overbought threshold
            rsi_threshold = config.REBALANCING.get('rsi_overbought_threshold', 70)
            
            # Check for asset-specific thresholds
            asset_conditions = config.REBALANCING.get('partial_sell_conditions', {})
            if asset in asset_conditions:
                rsi_threshold = asset_conditions[asset].get('rsi_threshold', rsi_threshold)
                sell_percentage = asset_conditions[asset].get('sell_percentage', 40.0)
            else:
                sell_percentage = config.REBALANCING.get('partial_sell_percentage', 40.0)
                
            if asset_rsi >= rsi_threshold:
                print(f"� {asset} RSI: {asset_rsi:.1f} ≥ {rsi_threshold} - OVERBOUGHT DETECTED!")
                
                # Check if we should preserve core holdings
                preserve_holdings = config.REBALANCING.get('preserve_core_holdings', {})
                min_preserve = preserve_holdings.get(asset, 0)
                
                if balance <= min_preserve:
                    print(f"🛡️ {asset} balance ({balance:.8f}) at or below preservation limit ({min_preserve})")
                    continue
                    
                # Calculate sellable amount (preserve minimum if specified)
                sellable_balance = balance - min_preserve
                if sellable_balance <= 0:
                    print(f"ℹ️ No sellable {asset} after preserving minimum holdings")
                    continue
                    
                # Adjust sell percentage based on preservation requirements
                max_sell_qty = sellable_balance * (sell_percentage / 100.0)
                
                print(f"💡 Selling {sell_percentage}% of available {asset}")
                print(f"   Available: {sellable_balance:.8f} {asset}")
                print(f"   Will sell: {max_sell_qty:.8f} {asset}")
                print(f"   Will preserve: {min_preserve:.8f} {asset}")
                
                # Execute partial sell
                result = sell_partial_position(
                    symbol,
                    (max_sell_qty / balance) * 100,  # Convert back to percentage of total balance
                    f"RSI {asset_rsi:.1f} overbought signal (threshold: {rsi_threshold})"
                )
                
                if result["success"]:
                    rebalancing_results["partial_sells"].append(result)
                    rebalancing_results["total_freed_usdt"] += result["value"]
                    print(f"✅ {asset} partial sell completed: ${result['value']:.2f}")
                else:
                    rebalancing_results["errors"].append(f"{asset} partial sell failed: {result['error']}")
                    
            else:
                print(f"✅ {asset} RSI: {asset_rsi:.1f} < {rsi_threshold} - Within normal range")
            
        # 2. Detect and liquidate dust positions
        dust_positions = detect_dust_positions(min_usdt_value=5.0)
        
        if dust_positions:
            print(f"\n💨 Found {len(dust_positions)} dust positions to liquidate")
            
            for dust_pos in dust_positions:
                result = liquidate_dust_position(dust_pos)
                
                if result["success"]:
                    rebalancing_results["dust_liquidations"].append(result)
                    rebalancing_results["total_freed_usdt"] += result["value"]
                    print(f"✅ Liquidated {dust_pos['asset']}: ${result['value']:.2f}")
                else:
                    rebalancing_results["errors"].append(f"Failed to liquidate {dust_pos['asset']}: {result['error']}")
        else:
            print("ℹ️ No dust positions found")
            
        # 3. Summary
        print("\n" + "="*60)
        print("📊 REBALANCING SUMMARY")
        print("="*60)
        print(f"Partial Sells: {len(rebalancing_results['partial_sells'])}")
        print(f"Dust Liquidations: {len(rebalancing_results['dust_liquidations'])}")
        print(f"Total USDT Freed: ${rebalancing_results['total_freed_usdt']:.2f}")
        print(f"Errors: {len(rebalancing_results['errors'])}")
        
        if rebalancing_results['errors']:
            print("\n⚠️ Errors encountered:")
            for error in rebalancing_results['errors']:
                print(f"   - {error}")
                
        # Send Telegram summary if available
        if TELEGRAM_AVAILABLE and (rebalancing_results['partial_sells'] or rebalancing_results['dust_liquidations']):
            summary_msg = f"🔄 Position Rebalancing Complete\n"
            summary_msg += f"💰 Total USDT Freed: ${rebalancing_results['total_freed_usdt']:.2f}\n"
            summary_msg += f"📈 Partial Sells: {len(rebalancing_results['partial_sells'])}\n"
            summary_msg += f"💨 Dust Liquidated: {len(rebalancing_results['dust_liquidations'])}"
            
            notify_market_update(summary_msg)
            
        return rebalancing_results
        
    except Exception as e:
        error_msg = f"Error in position rebalancing: {e}"
        print(f"❌ {error_msg}")
        log_error_to_csv(error_msg, "REBALANCING_ERROR", "execute_position_rebalancing", "ERROR")
        return {
            "timestamp": format_cairo_time(),
            "partial_sells": [],
            "dust_liquidations": [],
            "errors": [error_msg],
            "total_freed_usdt": 0
        }

# Add Flask route for manual rebalancing trigger
@app.route('/api/rebalance', methods=['POST'])
def api_rebalance():
    """API endpoint to trigger position rebalancing"""
    try:
        if not client:
            return jsonify({"error": "Client not initialized"}), 500
            
        print("\n🔄 Manual rebalancing triggered via API")
        results = execute_position_rebalancing()
        
        return jsonify({
            "success": True,
            "message": "Rebalancing completed",
            "results": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": format_cairo_time()
        }), 500

if __name__ == '__main__':
    print("\n🚀 Starting CRYPTIX AI Trading Bot...")
    print("=" * 50)
    
    # Initialize bot systems
    try:
        # Initialize API client once at startup
        if not bot_status.get('api_connected', False):
            print("🔧 Initializing API client...")
            if not initialize_client():
                print("❌ Failed to initialize API client at startup")
                raise RuntimeError("API client init failed")

        # Configure Flask for production by default; override with FLASK_DEBUG=1
        flask_host = os.getenv('FLASK_HOST', '0.0.0.0')
        flask_port = 10000
        flask_debug = str(os.getenv('FLASK_DEBUG', '0')).strip().lower() in ['1', 'true', 'yes', 'on']
        print(f"🌐 Starting Flask server on {flask_host}:{flask_port} (debug={'ON' if flask_debug else 'OFF'})")
        app.run(host=flask_host, port=flask_port, debug=flask_debug)
    except Exception as e:
        print(f"Failed to start application: {e}")
        log_error_to_csv(str(e), "STARTUP_ERROR", "main", "CRITICAL")
        
