"""
Memory-Optimized Configuration for CRYPTIX-ML Trading Bot
This configuration reduces memory usage for Render deployment
"""

import os
from memory_optimizer import get_memory_safe_config, optimize_ml_data_loading

# Get memory-optimized configuration
MEMORY_CONFIG = get_memory_safe_config()
ML_DATA_CONFIG = optimize_ml_data_loading()

# Override memory-intensive settings
def apply_memory_optimizations():
    """Apply memory optimizations to reduce usage for Render deployment"""
    
    # Limit data fetching
    os.environ['ML_LOOKBACK_DAYS'] = str(MEMORY_CONFIG['ML_LOOKBACK_DAYS'])
    os.environ['ML_MIN_TRAINING_SAMPLES'] = str(MEMORY_CONFIG['ML_MIN_TRAINING_SAMPLES'])
    
    # Reduce API calls
    os.environ['FETCH_LIMIT'] = str(MEMORY_CONFIG['FETCH_LIMIT'])
    
    # Limit concurrent operations
    os.environ['MAX_CONCURRENT_REQUESTS'] = '3'
    
    # Optimize pandas memory usage
    import pandas as pd
    pd.set_option('mode.copy_on_write', True)
    pd.set_option('string_storage', 'python')
    
    # Optimize numpy
    import numpy as np
    np.seterr(all='ignore')
    
    print("🧠 Memory optimizations applied for Render deployment")

# Memory-optimized symbol lists
RENDER_SYMBOLS = ML_DATA_CONFIG['symbols'][:5]  # Only 5 symbols
RENDER_TIMEFRAMES = ML_DATA_CONFIG['timeframes'][:2]  # Only 2 timeframes

# Reduced scan settings
RENDER_SCAN_CONFIG = {
    'base_assets': ["BTC", "ETH", "BNB", "SOL", "XRP"],  # Reduced from 10 to 5
    'min_volume_usdt': 2000000,  # Higher minimum to reduce candidates
    'scan_delay': 1.0,  # Increased delay to reduce memory pressure
    'max_opportunities': 3,  # Reduced from 5 to 3
}

# Memory-safe bot status structure
MEMORY_SAFE_BOT_STATUS = {
    'monitored_pairs': {},  # Will be limited to 5 pairs max
    'trading_summary': {
        'trades_history': []  # Will be limited to 10 trades max
    },
    'errors': [],  # Will be limited to 20 errors max
    'market_data_cache': {},  # Will be limited to 5 cache entries max
}

# Apply optimizations when imported
apply_memory_optimizations()
