"""
Local Testing Script for Supabase Integration
Run this locally to test the setup before deploying to Render
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_environment():
    """Test environment variables"""
    print("🔍 Testing Environment Variables...")
    
    required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY']
    missing = []
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: Set")
        else:
            print(f"❌ {var}: Missing")
            missing.append(var)
    
    if missing:
        print(f"\n💡 Create a .env file with:")
        print("SUPABASE_URL=https://your-project.supabase.co")
        print("SUPABASE_SERVICE_KEY=your-service-key")
        return False
    
    return True

def test_supabase_connection():
    """Test Supabase connection"""
    print("\n🔍 Testing Supabase Connection...")
    
    try:
        from supabase_position_tracker import SupabasePositionTracker
        tracker = SupabasePositionTracker()
        
        health = tracker.health_check()
        if health['status'] == 'healthy':
            print("✅ Supabase connection successful")
            print(f"📊 Current data: {health['trades_count']} trades, {health['positions_count']} positions")
            return True
        else:
            print(f"❌ Supabase health check failed: {health.get('error')}")
            return False
    
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        return False

def test_position_tracker():
    """Test position tracker functionality"""
    print("\n🔍 Testing Position Tracker...")
    
    try:
        from supabase_position_tracker import get_position_tracker
        tracker = get_position_tracker()
        
        # Test adding a dummy trade
        print("📝 Adding test trade...")
        success = tracker.add_trade(
            symbol="TESTUSDT",
            action="BUY", 
            quantity=1.0,
            price=100.0,
            source="test"
        )
        
        if success:
            print("✅ Test trade added successfully")
            
            # Check if position was created
            position = tracker.get_position("TESTUSDT")
            if position:
                print(f"✅ Position created: {position['quantity']} TESTUSDT @ ${position['avg_buy_price']}")
            
            # Clean up test data
            tracker.supabase.table('trades').delete().eq('symbol', 'TESTUSDT').execute()
            tracker.supabase.table('positions').delete().eq('symbol', 'TESTUSDT').execute()
            print("🧹 Test data cleaned up")
            
            return True
        else:
            print("❌ Failed to add test trade")
            return False
    
    except Exception as e:
        print(f"❌ Position tracker test failed: {e}")
        return False

def test_migration_dry_run():
    """Test migration functionality without actually migrating"""
    print("\n🔍 Testing Migration Functions...")
    
    try:
        # Test positions.json migration check
        positions_file = Path("logs/positions.json")
        if positions_file.exists():
            print("✅ Found positions.json for potential migration")
        else:
            print("ℹ️ No positions.json found (this is okay)")
        
        # Test Binance Excel migration check  
        excel_file = Path("logs/Binance-Spot Order History-202510191019.xlsx")
        if excel_file.exists():
            print("✅ Found Binance Excel file for potential migration")
            
            # Try reading the Excel file
            import pandas as pd
            df = pd.read_excel(excel_file)
            print(f"📊 Excel file contains {len(df)} records")
        else:
            print("ℹ️ No Binance Excel file found")
        
        return True
    
    except Exception as e:
        print(f"❌ Migration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 CRYPTIX-ML Supabase Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Environment Variables", test_environment),
        ("Supabase Connection", test_supabase_connection),
        ("Position Tracker", test_position_tracker),
        ("Migration Functions", test_migration_dry_run)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to deploy to Render.")
        print("\n📝 Next steps:")
        print("1. Push code to GitHub")
        print("2. Add environment variables to Render")
        print("3. Deploy and monitor logs")
    else:
        print("\n⚠️ Some tests failed. Fix issues before deploying.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())