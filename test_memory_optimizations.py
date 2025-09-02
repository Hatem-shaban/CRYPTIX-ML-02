"""
Test Script for Render Memory Optimizations
Run this to verify all optimizations are working correctly
"""

import sys
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_memory_optimizations():
    """Test all memory optimization components"""
    
    print("🧪 Testing CRYPTIX-ML Memory Optimizations for Render...")
    print("=" * 60)
    
    # Test 1: Compatibility patches
    try:
        from render_compatibility import apply_render_compatibility
        if apply_render_compatibility():
            print("✅ Test 1: Render compatibility patches - PASSED")
        else:
            print("❌ Test 1: Render compatibility patches - FAILED")
    except Exception as e:
        print(f"❌ Test 1: Render compatibility patches - ERROR: {e}")
    
    # Test 2: Memory optimizer
    try:
        from memory_optimizer import get_memory_optimizer, log_memory_usage
        optimizer = get_memory_optimizer()
        if optimizer:
            log_memory_usage("Test")
            print("✅ Test 2: Memory optimizer - PASSED")
        else:
            print("❌ Test 2: Memory optimizer - FAILED")
    except Exception as e:
        print(f"❌ Test 2: Memory optimizer - ERROR: {e}")
    
    # Test 3: Memory patches
    try:
        from memory_patches import apply_memory_efficient_patches
        if apply_memory_efficient_patches():
            print("✅ Test 3: Memory patches - PASSED")
        else:
            print("❌ Test 3: Memory patches - FAILED")
    except Exception as e:
        print(f"❌ Test 3: Memory patches - ERROR: {e}")
    
    # Test 4: Render memory optimizer
    try:
        from render_memory_optimizer import optimize_render_deployment
        memory_manager = optimize_render_deployment()
        if memory_manager:
            stats = memory_manager.get_memory_usage()
            print(f"✅ Test 4: Render memory optimizer - PASSED (Memory: {stats['rss_mb']:.1f}MB)")
        else:
            print("❌ Test 4: Render memory optimizer - FAILED")
    except Exception as e:
        print(f"❌ Test 4: Render memory optimizer - ERROR: {e}")
    
    # Test 5: Auto memory manager
    try:
        from auto_memory_manager import start_auto_memory_management, stop_auto_memory_management
        manager = start_auto_memory_management(max_memory_mb=480, cleanup_interval=60)
        if manager:
            print("✅ Test 5: Auto memory manager - PASSED")
            stop_auto_memory_management()
        else:
            print("❌ Test 5: Auto memory manager - FAILED")
    except Exception as e:
        print(f"❌ Test 5: Auto memory manager - ERROR: {e}")
    
    # Test 6: Technical indicators (with compatibility)
    try:
        import numpy as np
        from memory_patches import optimize_technical_indicators
        
        efficient_rsi, efficient_macd = optimize_technical_indicators()
        
        # Test data
        test_prices = np.random.random(50) * 100 + 50
        
        # Test RSI
        rsi_result = efficient_rsi(test_prices, 14)
        if 0 <= rsi_result <= 100:
            print("✅ Test 6a: RSI calculation - PASSED")
        else:
            print("❌ Test 6a: RSI calculation - FAILED")
        
        # Test MACD
        macd_result = efficient_macd(test_prices, 12, 26, 9)
        if isinstance(macd_result, dict) and 'macd' in macd_result:
            print("✅ Test 6b: MACD calculation - PASSED")
        else:
            print("❌ Test 6b: MACD calculation - FAILED")
            
    except Exception as e:
        print(f"❌ Test 6: Technical indicators - ERROR: {e}")
    
    # Test 7: Data type optimization
    try:
        import pandas as pd
        
        # Create test DataFrame
        test_df = pd.DataFrame({
            'close': np.random.random(100) * 100,
            'volume': np.random.random(100) * 1000000,
            'high': np.random.random(100) * 110,
            'low': np.random.random(100) * 90
        })
        
        from render_memory_optimizer import RenderMemoryManager
        manager = RenderMemoryManager()
        optimized_df = manager.optimize_dataframe_memory(test_df)
        
        # Check if float64 columns were converted to float32
        float32_cols = optimized_df.select_dtypes(include=['float32']).columns
        if len(float32_cols) > 0:
            print("✅ Test 7: Data type optimization - PASSED")
        else:
            print("❌ Test 7: Data type optimization - FAILED")
            
    except Exception as e:
        print(f"❌ Test 7: Data type optimization - ERROR: {e}")
    
    print("=" * 60)
    print("🎯 Memory optimization testing completed!")
    
    # Final memory check
    try:
        from render_memory_optimizer import RenderMemoryManager
        manager = RenderMemoryManager()
        final_stats = manager.get_memory_usage()
        print(f"📊 Final Memory Usage: {final_stats['rss_mb']:.1f}MB")
        
        if final_stats['rss_mb'] < 480:
            print("✅ Memory usage within Render limits!")
        else:
            print("⚠️ Memory usage may exceed Render limits")
            
    except Exception as e:
        print(f"⚠️ Could not check final memory usage: {e}")

if __name__ == "__main__":
    test_memory_optimizations()
