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
        """Check if models need retraining"""
        try:
            if not self.training_enabled:
                return False
            
            history_file = 'ml_training_history.json'
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                if history:
                    last_training = history[-1]
                    last_time = datetime.fromisoformat(last_training['timestamp'])
                    days_since_training = (datetime.now() - last_time).days
                    
                    return days_since_training >= self.retrain_days
            
            return True  # No history, should train
            
        except Exception as e:
            logger.error(f"Error checking retrain status: {e}")
            return True
    
    def execute_ml_training(self):
        """Execute ML training"""
        if self.training_in_progress:
            return
        
        self.training_in_progress = True
        
        try:
            logger.info("🚀 Starting ML training...")
            
            if self.trainer is None:
                self.trainer = EnhancedMLTrainer()
            
            # Execute training with Render-friendly settings
            results = self.trainer.train_all_models(
                days_back=60,           # Smaller dataset for Render
                force_refresh=False,
                incremental=True
            )
            
            if results['success']:
                logger.info(f"✅ ML training completed - {len(results['models_trained'])} models trained")
            else:
                logger.error(f"❌ ML training failed")
                
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
        finally:
            self.training_in_progress = False
    
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
