"""
Data Migration Script for CRYPTIX-ML Supabase Integration
Handles first-time setup and historical data import
"""

import os
import sys
import json
import logging
from datetime import datetime
from supabase_position_tracker import SupabasePositionTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/migration.log')
    ]
)
logger = logging.getLogger(__name__)

class DataMigration:
    """Handles all data migration tasks for Supabase integration"""
    
    def __init__(self):
        self.tracker = None
        self.setup_tracker()
    
    def setup_tracker(self):
        """Initialize Supabase tracker with error handling"""
        try:
            self.tracker = SupabasePositionTracker()
            logger.info("✅ Supabase tracker initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase tracker: {e}")
            logger.error("💡 Check your SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables")
            raise
    
    def run_health_check(self):
        """Verify Supabase connection and setup"""
        logger.info("🔍 Running Supabase health check...")
        
        health = self.tracker.health_check()
        
        if health['status'] == 'healthy':
            logger.info("✅ Supabase connection healthy")
            logger.info(f"📊 Current data: {health['trades_count']} trades, {health['positions_count']} positions")
            return True
        else:
            logger.error(f"❌ Supabase health check failed: {health.get('error', 'Unknown error')}")
            return False
    
    def migrate_binance_history(self, excel_path: str = None):
        """Import Binance historical data"""
        if excel_path is None:
            excel_path = "logs/Binance-Spot Order History-202510191019.xlsx"
        
        if not os.path.exists(excel_path):
            logger.warning(f"⚠️ Binance Excel file not found: {excel_path}")
            return False
        
        logger.info(f"🔄 Starting Binance history migration from: {excel_path}")
        
        try:
            success = self.tracker.migrate_from_binance_excel(excel_path)
            
            if success:
                logger.info("✅ Binance history migration completed successfully")
                
                # Show summary after migration
                summary = self.tracker.get_portfolio_summary()
                logger.info(f"📊 Portfolio summary: {summary['total_positions']} positions, Total cost: ${summary['total_cost']:.2f}")
                
                return True
            else:
                logger.error("❌ Binance history migration failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Migration error: {e}")
            return False
    
    def migrate_positions_json(self, json_path: str = None):
        """Import existing positions.json data"""
        if json_path is None:
            json_path = "logs/positions.json"
        
        if not os.path.exists(json_path):
            logger.warning(f"⚠️ Positions JSON file not found: {json_path}")
            return False
        
        logger.info(f"🔄 Starting positions.json migration from: {json_path}")
        
        try:
            success = self.tracker.migrate_from_json(json_path)
            
            if success:
                logger.info("✅ Positions JSON migration completed successfully")
                return True
            else:
                logger.error("❌ Positions JSON migration failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Migration error: {e}")
            return False
    
    def verify_migration(self):
        """Verify that migration was successful"""
        logger.info("🔍 Verifying migration results...")
        
        try:
            # Check trade history
            trades = self.tracker.get_trade_history(limit=10)
            logger.info(f"📈 Found {len(trades)} recent trades")
            
            if trades:
                logger.info("Recent trades:")
                for trade in trades[:3]:  # Show first 3
                    logger.info(f"  - {trade['action']} {trade['quantity']} {trade['symbol']} @ {trade['price']} ({trade['source']})")
            
            # Check positions
            positions = self.tracker.get_portfolio_summary()
            logger.info(f"💰 Current positions: {positions['total_positions']} symbols")
            
            if positions['positions']:
                logger.info("Current positions:")
                for symbol, pos in list(positions['positions'].items())[:3]:  # Show first 3
                    logger.info(f"  - {symbol}: {pos['quantity']} @ ${pos['avg_buy_price']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    def run_full_migration(self):
        """Run complete migration process"""
        logger.info("🚀 Starting full migration process...")
        
        # Step 1: Health check
        if not self.run_health_check():
            return False
        
        # Step 2: Migrate Binance history (priority)
        binance_success = self.migrate_binance_history()
        
        # Step 3: Migrate positions.json (if exists)
        json_success = self.migrate_positions_json()
        
        # Step 4: Verify results
        if binance_success or json_success:
            self.verify_migration()
            logger.info("✅ Migration process completed successfully!")
            return True
        else:
            logger.error("❌ No data was migrated")
            return False

def main():
    """Main migration entry point"""
    print("🔄 CRYPTIX-ML Supabase Migration Tool")
    print("=" * 50)
    
    # Check environment variables
    required_env = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY']
    missing_env = [var for var in required_env if not os.getenv(var)]
    
    if missing_env:
        print(f"❌ Missing environment variables: {', '.join(missing_env)}")
        print("\n💡 Please set these in your environment or .env file:")
        print("   SUPABASE_URL=https://your-project.supabase.co")
        print("   SUPABASE_SERVICE_KEY=your-service-key")
        sys.exit(1)
    
    try:
        migration = DataMigration()
        success = migration.run_full_migration()
        
        if success:
            print("\n🎉 Migration completed successfully!")
            print("Your CRYPTIX-ML bot is now ready to use Supabase for position tracking.")
        else:
            print("\n❌ Migration failed. Check the logs for details.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Migration failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()