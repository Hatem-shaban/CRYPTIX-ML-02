"""
Render Compatibility Patches
Ensures the bot works even if some dependencies are missing
"""

import logging
logger = logging.getLogger(__name__)

def patch_missing_dependencies():
    """Patch for missing dependencies on Render"""
    
    # Handle missing TA-Lib
    try:
        import talib
        logger.info("✅ TA-Lib available")
    except ImportError:
        logger.warning("⚠️ TA-Lib not available - using custom implementations")
        
        # Create mock talib module with basic functions
        import sys
        import types
        
        # Create a mock talib module
        mock_talib = types.ModuleType('talib')
        
        def mock_rsi(close, timeperiod=14):
            """Mock RSI using custom implementation"""
            try:
                from memory_patches import optimize_technical_indicators
                efficient_rsi, _ = optimize_technical_indicators()
                return [efficient_rsi(close, timeperiod)] * len(close)
            except:
                import numpy as np
                return np.array([50.0] * len(close))
        
        def mock_macd(close, fastperiod=12, slowperiod=26, signalperiod=9):
            """Mock MACD using custom implementation"""
            try:
                from memory_patches import optimize_technical_indicators
                _, efficient_macd = optimize_technical_indicators()
                result = efficient_macd(close, fastperiod, slowperiod, signalperiod)
                length = len(close)
                import numpy as np
                return (
                    np.array([result['macd']] * length),
                    np.array([result['signal']] * length),
                    np.array([result['histogram']] * length)
                )
            except:
                import numpy as np
                length = len(close)
                return (
                    np.array([0.0] * length),
                    np.array([0.0] * length),
                    np.array([0.0] * length)
                )
        
        def mock_sma(close, timeperiod=20):
            """Mock SMA using pandas rolling"""
            try:
                import pandas as pd
                if hasattr(close, 'rolling'):
                    return close.rolling(timeperiod).mean()
                else:
                    df = pd.Series(close)
                    return df.rolling(timeperiod).mean().values
            except:
                import numpy as np
                return np.array([close[0]] * len(close))
        
        # Add mock functions to the module
        mock_talib.RSI = mock_rsi
        mock_talib.MACD = mock_macd
        mock_talib.SMA = mock_sma
        
        # Add to sys.modules
        sys.modules['talib'] = mock_talib
        
        logger.info("✅ TA-Lib mock module created")
    
    # Handle other potential missing dependencies
    try:
        import lz4
        logger.info("✅ LZ4 available")
    except ImportError:
        logger.warning("⚠️ LZ4 not available - compression disabled")
        
        # Create mock lz4 module
        import sys
        import types
        
        mock_lz4 = types.ModuleType('lz4')
        
        class MockLZ4Frame:
            @staticmethod
            def compress(data):
                return data  # Return uncompressed
            
            @staticmethod
            def decompress(data):
                return data  # Return as-is
        
        mock_lz4.frame = MockLZ4Frame()
        sys.modules['lz4'] = mock_lz4
        sys.modules['lz4.frame'] = MockLZ4Frame()
    
    try:
        import pympler
        logger.info("✅ Pympler available")
    except ImportError:
        logger.warning("⚠️ Pympler not available - memory profiling disabled")
        
        # Create mock pympler
        import sys
        import types
        
        mock_pympler = types.ModuleType('pympler')
        
        class MockTracker:
            def print_diff(self):
                pass
            
            def track_object(self, obj):
                pass
        
        class MockSummary:
            @staticmethod
            def print_():
                pass
        
        mock_pympler.tracker = types.ModuleType('tracker')
        mock_pympler.tracker.SummaryTracker = MockTracker
        mock_pympler.summary = MockSummary()
        
        sys.modules['pympler'] = mock_pympler
        sys.modules['pympler.tracker'] = mock_pympler.tracker
        sys.modules['pympler.summary'] = mock_pympler.summary

def apply_render_compatibility():
    """Apply all Render compatibility patches"""
    try:
        patch_missing_dependencies()
        logger.info("✅ Render compatibility patches applied")
        return True
    except Exception as e:
        logger.error(f"Failed to apply compatibility patches: {e}")
        return False

if __name__ == "__main__":
    apply_render_compatibility()
