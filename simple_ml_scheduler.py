"""
Simple Automated ML Training for CRYPTIX Trading Bot (Render-optimized)
Lightweight scheduler that runs ML training during off-peak hours
"""

import os
import sys
import time
import schedule
import threading
import logging
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration and modules
import config
from enhanced_ml_training import EnhancedMLTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMLScheduler:
    """Simple ML Training Scheduler for Render deployment"""
    
    def __init__(self):
        self.trainer = None
        self.is_running = False
        self.scheduler_thread = None
        self.training_in_progress = False
        
        # Configuration from config.py
        self.training_start_hour = getattr(config, 'ML_TRAINING_START_HOUR', 2)
        self.training_enabled = getattr(config, 'ML_TRAINING_ENABLED', True)
        self.retrain_days = getattr(config, 'ML_MODEL_RETRAIN_DAYS', 7)
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        logger.info(f"🤖 Simple ML Scheduler initialized - Training at {self.training_start_hour}:00 AM")
    
    def is_off_peak_hours(self) -> bool:
        """Check if current time is during off-peak trading hours"""
        current_hour = datetime.now().hour
        # Off-peak hours: 1 AM - 5 AM
        return current_hour in [1, 2, 3, 4, 5]
    
    def should_retrain_models(self) -> bool:
        """Check if models need retraining with better logic"""
        try:
            if not self.training_enabled:
                return False
            
            # Check training history
            history_file = 'ml_training_history.json'
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                if history:
                    last_training = history[-1]
                    last_time = datetime.fromisoformat(last_training['timestamp'])
                    days_since_training = (datetime.now() - last_time).days
                    
                    # Also check if the last training used stale data
                    data_period = last_training.get('data_period', '')
                    if 'to' in data_period:
                        try:
                            # Extract end date from data period
                            end_date_str = data_period.split('to')[-1].strip()
                            end_date = datetime.strptime(end_date_str.split()[0], '%Y-%m-%d')
                            days_since_data = (datetime.now() - end_date).days
                            
                            # Force retrain if data is more than 3 days old, regardless of last training
                            if days_since_data > 3:
                                logger.info(f"🔄 Data is {days_since_data} days old, forcing retrain")
                                return True
                        except Exception as e:
                            logger.warning(f"Could not parse data period: {e}")
                    
                    return days_since_training >= self.retrain_days
            
            return True  # No history, should train
            
        except Exception as e:
            logger.error(f"Error checking retrain status: {e}")
            return True
    
    def execute_ml_training(self):
        """Execute ML training with smart data freshness detection"""
        if self.training_in_progress:
            return
        
        self.training_in_progress = True
        
        try:
            logger.info("🚀 Starting ML training...")
            
            if self.trainer is None:
                self.trainer = EnhancedMLTrainer()
            
            # Check if we need to force fresh data
            force_fresh = self._should_force_fresh_data()
            
            # Execute training with appropriate settings
            results = self.trainer.train_all_models(
                days_back=60,           # Smaller dataset for Render
                force_refresh=force_fresh,
                incremental=not force_fresh  # Use incremental only if not forcing fresh
            )
            
            if results['success']:
                logger.info(f"✅ ML training completed - {len(results['models_trained'])} models trained")
                logger.info(f"📊 Data period: {results.get('data_period', 'Unknown')}")
            else:
                logger.error(f"❌ ML training failed")
                
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
        finally:
            self.training_in_progress = False
    
    def _should_force_fresh_data(self) -> bool:
        """Determine if we should force fresh data fetch"""
        try:
            # Check existing data files
            import os
            data_files = [f for f in os.listdir('logs') if f.startswith('ml_training_data_') and f.endswith('.csv')]
            
            if not data_files:
                return True  # No data files, need fresh
            
            # Check the newest file
            latest_file = max(data_files)
            file_path = os.path.join('logs', latest_file)
            file_time = os.path.getmtime(file_path)
            hours_old = (datetime.now().timestamp() - file_time) / 3600
            
            # Force fresh if file is more than 12 hours old
            if hours_old > 12:
                logger.info(f"🔄 Data file is {hours_old:.1f}h old, forcing fresh fetch")
                return True
            
            # Check training history for data staleness
            history_file = 'ml_training_history.json'
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                if history:
                    last_training = history[-1]
                    data_period = last_training.get('data_period', '')
                    if 'to' in data_period:
                        try:
                            end_date_str = data_period.split('to')[-1].strip()
                            end_date = datetime.strptime(end_date_str.split()[0], '%Y-%m-%d')
                            days_since_data = (datetime.now() - end_date).days
                            
                            if days_since_data > 2:
                                logger.info(f"🔄 Training data is {days_since_data} days old, forcing fresh fetch")
                                return True
                        except Exception:
                            pass
            
            return False  # Data seems fresh enough
            
        except Exception as e:
            logger.warning(f"Error checking data freshness: {e}")
            return True  # When in doubt, fetch fresh
    
    def scheduled_training_job(self):
        """Main scheduled training job"""
        if not self.training_enabled:
            return
        
        if not self.is_off_peak_hours():
            return
        
        if not self.should_retrain_models():
            return
        
        logger.info("🎯 Starting scheduled ML training...")
        self.execute_ml_training()
    
    def start_scheduler(self):
        """Start the scheduler"""
        if self.is_running:
            return
        
        # Simple daily schedule
        schedule.every().day.at(f"{self.training_start_hour:02d}:00").do(self.scheduled_training_job)
        
        self.is_running = True
        
        def run_scheduler():
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(60)
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                    time.sleep(60)
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("✅ ML Scheduler started")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.is_running = False
        logger.info("🛑 ML Scheduler stopped")

# Global scheduler instance
_scheduler = None

def start_automated_ml_training():
    """Start the ML training scheduler"""
    global _scheduler
    if _scheduler is None:
        _scheduler = SimpleMLScheduler()
    _scheduler.start_scheduler()
    return _scheduler

def stop_automated_ml_training():
    """Stop the ML training scheduler"""
    global _scheduler
    if _scheduler:
        _scheduler.stop_scheduler()

def get_training_status():
    """Get basic training status"""
    global _scheduler
    if _scheduler is None:
        return {"scheduler_running": False, "training_enabled": getattr(config, 'ML_TRAINING_ENABLED', True)}
    
    return {
        "scheduler_running": _scheduler.is_running,
        "training_enabled": _scheduler.training_enabled,
        "training_in_progress": _scheduler.training_in_progress,
        "training_hour": _scheduler.training_start_hour
    }

def force_training_execution():
    """Force immediate training"""
    global _scheduler
    if _scheduler is None:
        _scheduler = SimpleMLScheduler()
    
    _scheduler.execute_ml_training()
    return {"success": True, "message": "Training executed"}

if __name__ == "__main__":
    # For standalone testing
    scheduler = SimpleMLScheduler()
    scheduler.start_scheduler()
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.stop_scheduler()
