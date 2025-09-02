"""
Memory-Optimized Data Fetcher for Render Deployment
Reduces memory usage by limiting data size and using efficient processing
"""

import pandas as pd
import numpy as np
from binance.client import Client
import gc
from memory_optimizer import MemoryOptimizer, memory_monitor, log_memory_usage
import config

class RenderDataFetcher:
    """Memory-optimized data fetcher for Render deployment"""
    
    def __init__(self, client: Client):
        self.client = client
        self.optimizer = MemoryOptimizer(max_memory_mb=400)  # Conservative limit
        self.cache = {}
        self.max_cache_size = 5
    
    def _clear_old_cache(self):
        """Clear old cache entries to prevent memory buildup"""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entries
            oldest_keys = list(self.cache.keys())[:len(self.cache) - self.max_cache_size + 1]
            for key in oldest_keys:
                del self.cache[key]
            gc.collect()
    
    @memory_monitor(max_memory_mb=400)
    def fetch_optimized_data(self, symbol="BTCUSDT", interval="1h", limit=50):
        """
        Fetch market data optimized for memory usage
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            limit: Maximum number of candles (reduced from 100)
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{interval}_{limit}"
            if cache_key in self.cache:
                log_memory_usage("Cache hit")
                return self.cache[cache_key].copy()
            
            log_memory_usage("Before data fetch")
            
            # Fetch data with reduced limit
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            
            if not klines:
                return None
            
            # Create DataFrame with minimal columns only
            essential_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_asset_volume', 'number_of_trades', 
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])[essential_columns]
            
            # Optimize data types immediately
            df = self.optimizer.optimize_dataframe(df)
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate only essential indicators
            df = self._calculate_essential_indicators(df)
            
            # Cache the result (clear old cache first)
            self._clear_old_cache()
            self.cache[cache_key] = df.copy()
            
            log_memory_usage("After data processing")
            
            return df
            
        except Exception as e:
            print(f"Error fetching optimized data: {e}")
            return None
    
    def _calculate_essential_indicators(self, df):
        """Calculate only essential technical indicators to save memory"""
        try:
            # Simple moving averages (only 2 instead of many)
            df['sma5'] = df['close'].rolling(5).mean()
            df['sma20'] = df['close'].rolling(20).mean()
            
            # RSI (simplified calculation)
            df['rsi'] = self._calculate_simple_rsi(df['close'].values)
            
            # MACD (simplified)
            macd_data = self._calculate_simple_macd(df['close'].values)
            df['macd'] = macd_data['macd']
            df['macd_signal'] = macd_data['signal']
            df['macd_trend'] = macd_data['trend']
            
            # Basic volatility
            df['volatility'] = df['close'].pct_change().rolling(10).std()
            
            # Optimize the DataFrame again
            df = self.optimizer.optimize_dataframe(df)
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return df
    
    def _calculate_simple_rsi(self, prices, period=14):
        """Memory-efficient RSI calculation"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return max(0, min(100, rsi))
            
        except Exception:
            return 50.0
    
    def _calculate_simple_macd(self, prices, fast=12, slow=26, signal=9):
        """Memory-efficient MACD calculation"""
        try:
            if len(prices) < slow:
                return {"macd": 0, "signal": 0, "trend": "NEUTRAL"}
            
            # Simple exponential moving averages
            fast_ema = self._simple_ema(prices, fast)
            slow_ema = self._simple_ema(prices, slow)
            
            macd_line = fast_ema - slow_ema
            signal_line = self._simple_ema([macd_line], signal)[0]
            
            trend = "BULLISH" if macd_line > signal_line else "BEARISH"
            
            return {
                "macd": round(macd_line, 6),
                "signal": round(signal_line, 6),
                "trend": trend
            }
            
        except Exception:
            return {"macd": 0, "signal": 0, "trend": "NEUTRAL"}
    
    def _simple_ema(self, data, period):
        """Simple EMA calculation"""
        try:
            if isinstance(data, (list, tuple)):
                data = np.array(data)
            
            alpha = 2 / (period + 1)
            ema = data[0]
            
            for price in data[1:]:
                ema = alpha * price + (1 - alpha) * ema
            
            return ema
            
        except Exception:
            return 0.0

def get_render_data_fetcher(client):
    """Factory function to get memory-optimized data fetcher"""
    return RenderDataFetcher(client)
