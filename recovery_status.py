#!/usr/bin/env python3
"""
Emergency Recovery Status Checker
Monitors API ban status and provides safe restart guidance
"""

import time
from datetime import datetime

def check_recovery_status():
    """Check if the API ban has been lifted and provide status update"""
    
    # The ban timestamp from the original error
    ban_until_timestamp = 1760943416197
    current_timestamp = datetime.now().timestamp() * 1000
    
    print("\nCRYPTIX Recovery Status Check")
    print("=" * 50)
    
    if current_timestamp < ban_until_timestamp:
        # Ban is still active
        ban_lift_time = datetime.fromtimestamp(ban_until_timestamp / 1000)
        time_remaining = (ban_until_timestamp - current_timestamp) / 1000
        minutes_remaining = time_remaining / 60
        hours_remaining = minutes_remaining / 60
        
        print("STATUS: API BAN STILL ACTIVE")
        print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ban Lifts At: {ban_lift_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if hours_remaining >= 1:
            print(f"Time Remaining: {hours_remaining:.1f} hours ({minutes_remaining:.0f} minutes)")
        else:
            print(f"Time Remaining: {minutes_remaining:.1f} minutes")
        
        print("\nWhat to do while waiting:")
        print("   • Do NOT attempt to restart the bot")
        print("   • Do NOT make any API calls to Binance")
        print("   • Run this script periodically to check status")
        print("   • Use post_ban_launcher.py when ban lifts")
        
        return False
    else:
        # Ban has been lifted
        print("STATUS: API BAN HAS BEEN LIFTED!")
        print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ban was lifted at: {datetime.fromtimestamp(ban_until_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\nSafe to restart - Choose your method:")
        print("   1. Run: python post_ban_launcher.py (RECOMMENDED)")
        print("      └── Uses ultra-conservative settings")
        print("   2. Run: python web_bot.py")
        print("      └── Will auto-detect and use emergency limits")
        print("   3. Manual config adjustment + normal restart")
        
        print("\nIMPORTANT REMINDERS:")
        print("   • Bot will start with 30 calls/min limit (vs normal 1200)")
        print("   • Monitor first hour carefully for any rate warnings") 
        print("   • Consider using WebSocket streams (emergency_websocket.py)")
        print("   • Gradually increase limits over time if stable")
        
        return True

def continuous_monitor():
    """Continuously monitor until ban lifts"""
    print("Starting continuous monitoring... (Press Ctrl+C to stop)")
    
    try:
        while True:
            is_clear = check_recovery_status()
            if is_clear:
                print("\nRecovery complete! You can now safely restart the bot.")
                break
            
            print(f"\nChecking again in 2 minutes...")
            time.sleep(120)  # Check every 2 minutes
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        continuous_monitor()
    else:
        check_recovery_status()
        print("\n💡 Tip: Use 'python recovery_status.py --monitor' for continuous checking")