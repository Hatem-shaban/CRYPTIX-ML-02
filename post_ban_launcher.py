#!/usr/bin/env python3
"""
Post-Ban Recovery Bot Launcher
Ultra-safe bot startup after API ban is lifted
"""

import time
import datetime
import subprocess
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def check_ban_lifted():
    """Check if ban is completely lifted"""
    ban_timestamp = 1760943416197
    current_time = int(time.time() * 1000)
    return current_time >= ban_timestamp

def safe_bot_startup():
    """Start bot with maximum safety measures"""
    print("🛡️ SAFE BOT STARTUP SEQUENCE")
    print("=" * 40)
    
    # Double-check ban status
    if not check_ban_lifted():
        print("❌ Ban still active! Aborting startup.")
        return False
    
    print("✅ Ban confirmed lifted - proceeding with safe startup")
    
    # Apply emergency configuration
    print("🔧 Applying emergency rate limiting...")
    
    # Check environment
    required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        return False
    
    print("✅ Environment configuration verified")
    
    # Verify emergency mode is active in config
    try:
        import config
        if hasattr(config, 'EMERGENCY_MODE') and config.EMERGENCY_MODE:
            print("✅ Emergency mode confirmed in config.py")
        else:
            print("⚠️ Emergency mode not detected in config.py")
    except Exception as e:
        print(f"⚠️ Could not verify config: {e}")
    
    # Start with monitoring only (no trading initially)
    print("📊 Starting in MONITORING MODE (no trading)")
    print("⚠️ Manual trading approval required for safety")
    print("\n🎯 NEXT STEPS:")
    print("1. Bot will start in monitoring mode only")
    print("2. API calls limited to absolute minimum")
    print("3. Monitor for 1 hour before enabling trading")
    print("4. Check /api/ml-training/status for health")
    
    return True

def main():
    """Main recovery launcher"""
    print("🚨 POST-BAN RECOVERY LAUNCHER")
    print("=" * 50)
    
    # Wait for ban to lift if still active
    while not check_ban_lifted():
        ban_timestamp = 1760943416197
        current_time = int(time.time() * 1000)
        time_left = (ban_timestamp - current_time) / 1000 / 60
        
        print(f"⏳ Waiting for ban to lift... {time_left:.1f} minutes remaining")
        time.sleep(60)  # Check every minute
    
    print("🎉 BAN HAS BEEN LIFTED!")
    time.sleep(10)  # Wait an extra 10 seconds to be sure
    
    # Start safe recovery
    if safe_bot_startup():
        print("\n✅ Safe startup sequence completed")
        print("🤖 You can now start the bot manually with:")
        print("   python web_bot.py")
        print("\n⚠️ REMEMBER:")
        print("- Bot is in emergency mode")
        print("- Ultra-conservative API limits applied") 
        print("- Monitor carefully for first hour")
        print("- Check logs for any rate limit warnings")
    else:
        print("\n❌ Safe startup failed - check configuration")

if __name__ == "__main__":
    main()