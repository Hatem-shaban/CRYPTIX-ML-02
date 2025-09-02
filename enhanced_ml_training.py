"""
Enhanced ML Training Module for CRYPTIX Trading Bot
Handles training with comprehensive historical data from Binance
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Safe ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available. ML training will be limited.")

# Import our enhanced data fetcher
try:
    from enhanced_historical_data import EnhancedHistoricalDataFetcher
    ENHANCED_DATA_AVAILABLE = True
except ImportError:
    ENHANCED_DATA_AVAILABLE = False
    print("⚠️ Enhanced data fetcher not available")

# Import data cleaner
try:
    from data_cleaner import DataCleaner
    DATA_CLEANER_AVAILABLE = True
except ImportError:
    DATA_CLEANER_AVAILABLE = False
    print("⚠️ Data cleaner not available")

import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMLTrainer:
    """
    Enhanced ML Training Manager using comprehensive historical data
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.training_history = []
        self.model_performance = {}
        
        # Initialize data cleaner
        if DATA_CLEANER_AVAILABLE:
            self.data_cleaner = DataCleaner()
        else:
            self.data_cleaner = None
            logger.warning("⚠️ Data cleaner not available - training may fail with dirty data")
        
        # Enhanced model paths
        self.model_paths = {
            'trend_model': 'rf_price_trend_model.pkl',
            'regime_model': 'rf_market_regime_model.pkl',
            'pattern_model': 'rf_pattern_recognition_model.pkl',
            'signal_model': 'rf_signal_success_model.pkl',
            'volatility_model': 'rf_volatility_model.pkl',
            
            'trend_scaler': 'rf_scaler.pkl',
            'regime_scaler': 'rf_regime_scaler.pkl',
            'pattern_scaler': 'rf_pattern_scaler.pkl',
            'signal_scaler': 'rf_signal_scaler.pkl',
            'volatility_scaler': 'rf_volatility_scaler.pkl',
            
            'trend_selector': 'rf_trend_selector.pkl',
            'regime_selector': 'rf_regime_selector.pkl',
            'pattern_selector': 'rf_pattern_selector.pkl',
            'signal_selector': 'rf_signal_selector.pkl',
            'volatility_selector': 'rf_volatility_selector.pkl'
        }
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
    
    def fetch_fresh_training_data(self, days_back: int = 90, force_refresh: bool = False, 
                                incremental: bool = True) -> pd.DataFrame:
        """
        Fetch training data with smart incremental loading
        
        Args:
            days_back: Number of days of historical data to fetch (reduced default)
            force_refresh: Force fresh download
            incremental: Use incremental loading for efficiency
            
        Returns:
            DataFrame with training data
        """
        # Check for existing recent data
        data_files = [f for f in os.listdir('logs') if f.startswith('ml_training_data_') and f.endswith('.csv')]
        
        if not force_refresh and data_files:
            # Use most recent file if it's less than 4 hours old (more frequent updates)
            latest_file = max(data_files)
            file_path = os.path.join('logs', latest_file)
            file_time = os.path.getmtime(file_path)
            
            if (datetime.now().timestamp() - file_time) < 4 * 3600:  # Less than 4 hours
                logger.info(f"📥 Using recent training data: {latest_file}")
                return pd.read_csv(file_path)
            
            # For incremental mode, try to update existing data
            elif incremental:
                logger.info(f"🔄 Attempting incremental update from: {latest_file}")
                existing_df = pd.read_csv(file_path)
                return self._fetch_incremental_update(existing_df, days_back)
        
        # Fetch fresh data
        if ENHANCED_DATA_AVAILABLE:
            logger.info("🔄 Fetching fresh training data from Binance...")
            fetcher = EnhancedHistoricalDataFetcher()
            df = fetcher.fetch_comprehensive_data(days_back=days_back)
            
            if not df.empty:
                # Save the data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"logs/ml_training_data_{timestamp}.csv"
                fetcher.save_training_data(df, filename)
                logger.info(f"✅ Fresh training data saved: {filename}")
                return df
            else:
                logger.error("❌ Failed to fetch fresh data")
                return pd.DataFrame()
        else:
            logger.warning("⚠️ Enhanced data fetcher not available, using fallback")
            return self.load_fallback_data()

    def _fetch_incremental_update(self, existing_df: pd.DataFrame, days_back: int) -> pd.DataFrame:
        """Fetch incremental updates to existing training data"""
        try:
            if ENHANCED_DATA_AVAILABLE:
                fetcher = EnhancedHistoricalDataFetcher()
                
                # Use incremental fetch for each symbol/timeframe combination
                updated_data = []
                symbols_processed = set()
                
                for _, row in existing_df.groupby(['symbol', 'timeframe']).first().iterrows():
                    symbol = row['symbol']
                    timeframe = row['timeframe']
                    
                    if f"{symbol}_{timeframe}" not in symbols_processed:
                        # Get existing data for this symbol/timeframe
                        existing_subset = existing_df[
                            (existing_df['symbol'] == symbol) & 
                            (existing_df['timeframe'] == timeframe)
                        ].copy()
                        
                        # Fetch incremental data
                        updated_subset = fetcher.fetch_incremental_data(
                            symbol, timeframe, existing_subset
                        )
                        
                        if not updated_subset.empty:
                            updated_data.append(updated_subset)
                        
                        symbols_processed.add(f"{symbol}_{timeframe}")
                
                if updated_data:
                    combined_df = pd.concat(updated_data, ignore_index=True)
                    
                    # Save incremental update
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"logs/ml_training_data_incremental_{timestamp}.csv"
                    fetcher.save_training_data(combined_df, filename)
                    logger.info(f"✅ Incremental training data saved: {filename}")
                    return combined_df
                
            # Fallback to existing data if incremental fails
            logger.warning("⚠️ Incremental update failed, using existing data")
            return existing_df
            
        except Exception as e:
            logger.error(f"Error in incremental update: {e}")
            return existing_df
    
    def load_fallback_data(self) -> pd.DataFrame:
        """Load fallback data from existing CSV files"""
        try:
            # Try to load the most comprehensive existing data
            if os.path.exists('logs/trade_history_combined.csv'):
                df = pd.read_csv('logs/trade_history_combined.csv')
                logger.info(f"📥 Loaded fallback data: {len(df)} records")
                return df
            else:
                logger.error("❌ No fallback data available")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading fallback data: {e}")
            return pd.DataFrame()
    
    def prepare_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare comprehensive ML features from the dataset
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Tuple of (feature_df, feature_names)
        """
        logger.info("🔧 Preparing ML features...")
        
        # Clean data first if cleaner is available
        if self.data_cleaner:
            df_clean = self.data_cleaner.clean_data(df)
        else:
            df_clean = df.copy()
            # Basic cleaning without data cleaner
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
        
        # Core feature columns (technical indicators)
        feature_columns = [
            # Price and returns
            'returns', 'log_returns', 'high_low_ratio', 'open_close_ratio',
            
            # Volume features
            'volume_ratio', 'price_volume',
            
            # Technical indicators
            'rsi', 'rsi_oversold', 'rsi_overbought',
            'macd', 'macd_signal', 'macd_histogram', 'macd_trend', 'macd_crossover',
            
            # Moving averages
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
            
            # Moving average signals
            'price_above_sma_20', 'price_above_sma_50', 'price_above_sma_100', 'price_above_sma_200',
            'price_above_ema_20', 'price_above_ema_50', 'price_above_ema_100', 'price_above_ema_200',
            
            # Bollinger Bands
            'bb_width', 'bb_position', 'bb_squeeze',
            
            # Volatility
            'atr', 'volatility', 'volatility_ratio',
            
            # Momentum
            'stoch_k', 'stoch_d', 'williams_r', 'roc', 'momentum',
            
            # Trend strength
            'adx', 'plus_di', 'minus_di', 'di_diff', 'trend_strength',
            
            # VWAP
            'vwap_distance',
            
            # Pattern recognition
            'doji', 'hammer', 'shooting_star', 'engulfing',
            
            # Support/Resistance
            'resistance_distance', 'support_distance'
        ]
        
        # Filter columns that exist in the dataframe
        available_features = [col for col in feature_columns if col in df_clean.columns]
        
        if not available_features:
            logger.error("❌ No feature columns found in data!")
            return pd.DataFrame(), []
        
        # Create feature DataFrame
        feature_df = df_clean[available_features + ['symbol', 'timeframe']].copy()
        
        # Handle categorical variables
        if 'symbol' in feature_df.columns:
            # Create symbol dummy variables
            symbol_dummies = pd.get_dummies(feature_df['symbol'], prefix='symbol')
            feature_df = pd.concat([feature_df, symbol_dummies], axis=1)
            feature_df = feature_df.drop('symbol', axis=1)
        
        if 'timeframe' in feature_df.columns:
            # Create timeframe dummy variables
            timeframe_dummies = pd.get_dummies(feature_df['timeframe'], prefix='timeframe')
            feature_df = pd.concat([feature_df, timeframe_dummies], axis=1)
            feature_df = feature_df.drop('timeframe', axis=1)
        
        # Final cleaning of feature DataFrame
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        feature_df[numeric_cols] = feature_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0)
        
        logger.info(f"✅ Prepared {len(feature_df.columns)} features for {len(feature_df)} samples")
        return feature_df, list(feature_df.columns)
    
    def train_trend_prediction_model(self, df: pd.DataFrame) -> Dict:
        """Train price trend prediction model"""
        logger.info("🎯 Training trend prediction model...")
        
        try:
            # Prepare features
            features_df, feature_names = self.prepare_ml_features(df)
            
            if features_df.empty:
                return {'success': False, 'error': 'No features available'}
            
            # Create trend target (future price direction)
            if 'future_return_4h' in df.columns:
                y = (df['future_return_4h'] > 0).astype(int)
            else:
                # Fallback: use next period return
                y = (df['close'].shift(-1) / df['close'] > 1).astype(int)
            
            # Remove rows with NaN targets
            valid_mask = ~y.isna()
            X = features_df[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 100:
                return {'success': False, 'error': 'Insufficient training data'}
            
            # Validate and clean features if data cleaner is available
            if self.data_cleaner:
                X, y = self.data_cleaner.validate_features(X, y)
            
            # Split data (time-aware split)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Feature selection
            selector = SelectKBest(score_func=f_classif, k=min(50, len(feature_names)))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Save model components
            joblib.dump(model, self.model_paths['trend_model'])
            joblib.dump(scaler, self.model_paths['trend_scaler'])
            joblib.dump(selector, self.model_paths['trend_selector'])
            
            self.models['trend'] = model
            self.scalers['trend'] = scaler
            self.feature_selectors['trend'] = selector
            
            result = {
                'success': True,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'features_used': selector.get_support().sum(),
                'training_samples': len(X_train)
            }
            
            logger.info(f"✅ Trend model trained - Test accuracy: {test_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training trend model: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_signal_success_model(self, df: pd.DataFrame) -> Dict:
        """Train signal success prediction model"""
        logger.info("🎯 Training signal success model...")
        
        try:
            # Prepare features
            features_df, feature_names = self.prepare_ml_features(df)
            
            if features_df.empty:
                return {'success': False, 'error': 'No features available'}
            
            # Use signal_success target if available
            if 'signal_success' in df.columns:
                y = df['signal_success']
            else:
                # Create signal success target (2% gain in next 4 periods)
                if 'future_return_4h' in df.columns:
                    y = (df['future_return_4h'] > 0.02).astype(int)
                else:
                    y = (df['close'].shift(-4) / df['close'] > 1.02).astype(int)
            
            # Remove rows with NaN targets
            valid_mask = ~y.isna()
            X = features_df[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 100:
                return {'success': False, 'error': 'Insufficient training data'}
            
            # Validate and clean features if data cleaner is available
            if self.data_cleaner:
                X, y = self.data_cleaner.validate_features(X, y)
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Feature selection
            selector = SelectKBest(score_func=mutual_info_classif, k=min(40, len(feature_names)))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Train model with class balancing
            model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=30,
                min_samples_leaf=15,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Calculate AUC
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Save model components
            joblib.dump(model, self.model_paths['signal_model'])
            joblib.dump(scaler, self.model_paths['signal_scaler'])
            joblib.dump(selector, self.model_paths['signal_selector'])
            
            self.models['signal'] = model
            self.scalers['signal'] = scaler
            self.feature_selectors['signal'] = selector
            
            result = {
                'success': True,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'auc_score': auc_score,
                'features_used': selector.get_support().sum(),
                'training_samples': len(X_train),
                'positive_rate': y_train.mean()
            }
            
            logger.info(f"✅ Signal model trained - Test AUC: {auc_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training signal model: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_market_regime_model(self, df: pd.DataFrame) -> Dict:
        """Train market regime classification model"""
        logger.info("🎯 Training market regime model...")
        
        try:
            # Prepare features
            features_df, feature_names = self.prepare_ml_features(df)
            
            if features_df.empty:
                return {'success': False, 'error': 'No features available'}
            
            # Use market_regime target if available
            if 'market_regime' in df.columns:
                y = df['market_regime']
            else:
                # Create market regime based on moving averages
                sma_20 = df['close'].rolling(20).mean()
                sma_50 = df['close'].rolling(50).mean()
                sma_100 = df['close'].rolling(100).mean()
                
                y = np.where(
                    (sma_20 > sma_50) & (sma_50 > sma_100), 'uptrend',
                    np.where(
                        (sma_20 < sma_50) & (sma_50 < sma_100), 'downtrend',
                        'sideways'
                    )
                )
                y = pd.Series(y)
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Remove rows with NaN targets
            valid_mask = ~pd.isna(y_encoded)
            X = features_df[valid_mask]
            y_encoded = y_encoded[valid_mask]
            
            if len(X) < 100:
                return {'success': False, 'error': 'Insufficient training data'}
            
            # Validate and clean features if data cleaner is available
            if self.data_cleaner:
                X, y_encoded = self.data_cleaner.validate_features(X, pd.Series(y_encoded))
                y_encoded = y_encoded.values  # Convert back to array
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]
            
            # Feature selection
            selector = SelectKBest(score_func=f_classif, k=min(35, len(feature_names)))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=180,
                max_depth=12,
                min_samples_split=25,
                min_samples_leaf=12,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Save model components
            joblib.dump(model, self.model_paths['regime_model'])
            joblib.dump(scaler, self.model_paths['regime_scaler'])
            joblib.dump(selector, self.model_paths['regime_selector'])
            joblib.dump(label_encoder, 'rf_regime_label_encoder.pkl')
            
            self.models['regime'] = model
            self.scalers['regime'] = scaler
            self.feature_selectors['regime'] = selector
            
            result = {
                'success': True,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'features_used': selector.get_support().sum(),
                'training_samples': len(X_train),
                'regime_classes': list(label_encoder.classes_)
            }
            
            logger.info(f"✅ Regime model trained - Test accuracy: {test_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training regime model: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_all_models(self, days_back: int = 90, force_refresh: bool = False, 
                        incremental: bool = True) -> Dict:
        """
        Train all ML models with smart incremental data loading
        
        Args:
            days_back: Days of historical data to use (reduced default)
            force_refresh: Force fresh data download
            incremental: Use incremental loading for efficiency
            
        Returns:
            Training results summary
        """
        logger.info("🚀 Starting enhanced ML model training...")
        
        if not SKLEARN_AVAILABLE:
            return {'success': False, 'error': 'Scikit-learn not available'}
        
        # Fetch training data with incremental loading
        df = self.fetch_fresh_training_data(days_back, force_refresh, incremental)
        
        if df.empty:
            return {'success': False, 'error': 'No training data available'}
        
        logger.info(f"📊 Training with {len(df)} samples from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        results = {}
        
        # Train each model
        models_to_train = [
            ('trend', self.train_trend_prediction_model),
            ('signal', self.train_signal_success_model),
            ('regime', self.train_market_regime_model)
        ]
        
        for model_name, train_func in models_to_train:
            logger.info(f"\n🔄 Training {model_name} model...")
            result = train_func(df)
            results[model_name] = result
            
            if result['success']:
                logger.info(f"✅ {model_name} model training completed successfully")
            else:
                logger.error(f"❌ {model_name} model training failed: {result.get('error', 'Unknown error')}")
        
        # Save training history (with JSON serialization fix)
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'data_samples': int(len(df)),
            'data_period': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'results': {k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv 
                           for kk, vv in v.items()} for k, v in results.items()},
            'models_trained': [k for k, v in results.items() if v['success']],
            'training_duration': 'completed'
        }
        
        # Save to file
        history_file = 'ml_training_history.json'
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
        except (json.JSONDecodeError, FileNotFoundError):
            # Start fresh if file is corrupted
            history = []
        
        history.append(training_record)
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Summary
        successful_models = [k for k, v in results.items() if v['success']]
        
        summary = {
            'success': len(successful_models) > 0,
            'models_trained': successful_models,
            'total_models': len(models_to_train),
            'training_data_size': len(df),
            'results': results
        }
        
        logger.info(f"\n🎉 Training completed! {len(successful_models)}/{len(models_to_train)} models trained successfully")
        
        return summary

def main():
    """Main training execution"""
    logger.info("🤖 CRYPTIX Enhanced ML Training System")
    
    trainer = EnhancedMLTrainer()
    
    # Train all models with incremental approach
    results = trainer.train_all_models(days_back=90, force_refresh=False, incremental=True)
    
    if results['success']:
        logger.info(f"\n✅ Training completed successfully!")
        logger.info(f"🎯 Models trained: {', '.join(results['models_trained'])}")
        logger.info(f"📊 Training data: {results['training_data_size']:,} samples")
    else:
        logger.error("❌ Training failed!")
        
    return results

if __name__ == "__main__":
    main()
