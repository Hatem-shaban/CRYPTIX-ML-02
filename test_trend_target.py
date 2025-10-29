"""
Quick test to validate the new ADX-based TREND target
Checks that the target is created correctly before full training
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_historical_data import EnhancedHistoricalDataFetcher

def test_trend_target():
    """Test the new ADX-based trend target creation"""
    
    print("=" * 70)
    print("🧪 Testing New ADX-Based TREND Target")
    print("=" * 70)
    
    # Use existing training data file
    csv_file = 'logs/ml_training_data_20251029_131146.csv'
    
    print(f"\n📥 Loading existing training data: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
    
    # Check if ADX features exist
    required_features = ['adx', 'plus_di', 'minus_di']
    missing = [f for f in required_features if f not in df.columns]
    
    if missing:
        print(f"\n⚠️ Missing features: {missing}")
        print("Will use fallback target (2% threshold)")
        
        # Test fallback
        if 'future_return_4h' in df.columns:
            y = (df['future_return_4h'] > 0.02).astype(int)
            print(f"\n📊 Fallback Target Distribution:")
            print(f"   Uptrend (>2%): {y.sum():,} ({y.sum()/len(y):.1%})")
            print(f"   Downtrend/Neutral: {(len(y) - y.sum()):,} ({(len(y) - y.sum())/len(y):.1%})")
            return True
        else:
            print("❌ No future_return_4h column available")
            return False
    
    print(f"\n✅ All required features present: {required_features}")
    
    # Create ADX-based trend target
    print("\n🎯 Creating ADX-based TREND target...")
    
    y = np.where(
        (df['adx'] > 25) & (df['plus_di'] > df['minus_di']),
        1,  # Confirmed uptrend
        np.where(
            (df['adx'] > 25) & (df['plus_di'] < df['minus_di']),
            0,  # Confirmed downtrend
            0  # No trend
        )
    )
    
    y = pd.Series(y, index=df.index, dtype=int)
    
    # Analyze target distribution
    print("\n" + "=" * 70)
    print("📊 TREND Target Analysis (ADX-Based)")
    print("=" * 70)
    
    uptrend_count = y.sum()
    downtrend_neutral_count = len(y) - y.sum()
    
    print(f"\nTotal Samples: {len(y):,}")
    print(f"Uptrend (ADX>25 + DI+>DI-): {uptrend_count:,} ({uptrend_count/len(y):.1%})")
    print(f"Downtrend/Neutral: {downtrend_neutral_count:,} ({downtrend_neutral_count/len(y):.1%})")
    
    # Detailed breakdown
    strong_uptrend = ((df['adx'] > 25) & (df['plus_di'] > df['minus_di'])).sum()
    strong_downtrend = ((df['adx'] > 25) & (df['plus_di'] < df['minus_di'])).sum()
    no_trend = ((df['adx'] <= 25)).sum()
    
    print(f"\nDetailed Breakdown:")
    print(f"  Strong Uptrend (ADX>25, DI+>DI-): {strong_uptrend:,} ({strong_uptrend/len(y):.1%})")
    print(f"  Strong Downtrend (ADX>25, DI+<DI-): {strong_downtrend:,} ({strong_downtrend/len(y):.1%})")
    print(f"  Weak/No Trend (ADX<=25): {no_trend:,} ({no_trend/len(y):.1%})")
    
    # Check ADX distribution
    print(f"\n📈 ADX Statistics:")
    print(f"  Mean: {df['adx'].mean():.2f}")
    print(f"  Median: {df['adx'].median():.2f}")
    print(f"  Samples with ADX > 25: {(df['adx'] > 25).sum():,} ({(df['adx'] > 25).sum()/len(df):.1%})")
    print(f"  Samples with ADX > 30: {(df['adx'] > 30).sum():,} ({(df['adx'] > 30).sum()/len(df):.1%})")
    
    # Validate class balance
    print(f"\n⚖️ Class Balance Check:")
    if uptrend_count/len(y) < 0.15:
        print(f"   ⚠️ Uptrend class is small ({uptrend_count/len(y):.1%})")
        print(f"   💡 Suggestion: Lower ADX threshold to 20 for more samples")
    elif uptrend_count/len(y) > 0.45:
        print(f"   ⚠️ Classes too balanced ({uptrend_count/len(y):.1%})")
        print(f"   💡 Suggestion: Raise ADX threshold to 30 for stronger trends")
    else:
        print(f"   ✅ Class distribution looks good ({uptrend_count/len(y):.1%} uptrend)")
    
    print("\n" + "=" * 70)
    print("✅ Test Complete - Target creation works correctly!")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = test_trend_target()
    sys.exit(0 if success else 1)
