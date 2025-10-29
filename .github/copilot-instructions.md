# CRYPTIX-ML AI Trading Bot - Copilot Instructions

## Project Overview

CRYPTIX-ML is an advanced cryptocurrency trading bot with machine learning capabilities, built for production deployment on Render cloud platform. The system combines real-time technical analysis, incremental ML learning, and intelligent risk management.

## Architecture & Core Components

### Entry Points
- **Local Development**: `python web_bot.py` - Main Flask application
- **Production (Render)**: `python render_launcher.py` - Memory-optimized cloud launcher
- **ML Training**: `python enhanced_ml_training.py` - Comprehensive model training
- **Configuration Check**: Built-in validation via `initialize_client()` and `/health` endpoint

### Key Architectural Patterns

#### 1. Centralized Configuration (`config.py`)
All trading parameters, risk limits, and features are environment-variable driven:
```python
# Environment-first configuration pattern
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '50.0'))
ML_ENABLED = os.getenv('ML_ENABLED', 'true').lower() == 'true'
DEFAULT_STRATEGY = os.getenv('DEFAULT_STRATEGY', 'ML_PURE')
```

#### 2. Global Bot State (`web_bot.py`)
Single source of truth for runtime state in `bot_status` dictionary:
```python
bot_status = {
    'running': False,
    'api_connected': False,
    'monitored_pairs': {},  # Multi-symbol tracking
    'trading_strategy': config.DEFAULT_STRATEGY,
    'consecutive_losses': 0,
    'daily_loss': 0.0
}
```

#### 3. Modular Service Architecture
Core services are singleton instances obtained via getter functions:
- `get_position_tracker()` - Portfolio management (Supabase + file fallback)
- `get_risk_manager()` - Risk controls and position sizing
- `get_signal_filter()` - Trading signal validation
- `get_market_intelligence()` - Market regime detection

#### 4. Incremental ML Learning System
Unique cumulative learning approach where models build knowledge over time:
```python
# Models accumulate samples: 30k → 60k → 90k...
trainer = EnhancedMLTrainer(use_incremental=True)
result = trainer.train_all_models(incremental=True)  # Default behavior
```

## Critical Development Workflows

### Environment Setup
```bash
# Required for any development
cp .env.example .env
# Edit .env with actual API credentials
# Validation happens automatically during bot startup
```

### API Credentials (Multiple Fallback Pattern)
The system supports multiple environment variable names for Render compatibility:
```python
api_key = (
    os.getenv("API_KEY") or 
    os.getenv("BINANCE_API_KEY") or
    os.environ.get("API_KEY")  # Direct environ access
)
```

### Testing Workflows
- **Safe Testing**: Always set `USE_TESTNET=true` for development
- **Configuration Validation**: Built into `initialize_client()` function
- **Health Check**: Access `http://localhost:5000/health` for system status
- **Web Interface**: Access `http://localhost:5000` for bot control and monitoring
- **Telegram Test**: Built-in connection diagnostics in `telegram_notify.py`

### Deployment Commands
```bash
# Local development
python web_bot.py

# Render production (memory optimized)
python render_launcher.py

# Force ML retraining
python enhanced_ml_training.py

# Migrate old models to incremental format
python migrate_to_incremental.py
```

## Project-Specific Patterns

### 1. Memory-Optimized Cloud Deployment
Render deployment uses specialized memory management:
- `render_launcher.py` - Main entry point with memory monitoring
- `render_memory_optimizer.py` - Aggressive memory cleanup and optimization
- `render_compatibility.py` - Handles missing dependencies (TA-Lib, etc.)
- `auto_memory_manager.py` - Background memory monitoring

### 2. Dual Data Storage Strategy
Production system uses Supabase with file-based fallback:
```python
# Pattern: Always try Supabase first, fallback to files
tracker = get_position_tracker()  # Returns SupabasePositionTracker or file-based
```

### 3. Multi-Strategy Trading System
Strategy switching without restart:
```python
# Runtime strategy changes via web interface
bot_status['trading_strategy'] = 'ADAPTIVE'  # or 'ML_PURE'
```

### 4. Emergency Mode System
Built-in API ban protection and rate limiting:
```python
# Emergency mode detection
if hasattr(config, 'EMERGENCY_MODE') and config.EMERGENCY_MODE:
    # Ultra-conservative rate limits, skip API calls
```

### 5. Incremental Learning vs Batch Training
Unique approach where models accumulate knowledge:
```python
# Incremental (default): Builds on previous training
trainer.train_all_models(incremental=True)

# Batch: Retrains from scratch (monthly refresh)
trainer.train_all_models(force_batch=True)
```

## Integration Points

### External APIs
- **Binance API**: Primary trading data source
- **Telegram Bot**: Real-time notifications
- **Supabase**: Cloud database for trades/positions
- **Coinbase API**: Fallback price data during high volatility

### Internal Service Communication
- **Flask Routes**: `/api/status`, `/api/ml-training/force`, `/start`, `/stop`
- **Background Threads**: Trading loop, ML scheduler, memory management
- **Shared State**: Global `bot_status` dictionary for cross-component communication

### File System Patterns
- **Models**: `models/*.pkl` - Trained ML models with centralized path management
- **Logs**: `logs/*.csv` - Trading history, errors, signals
- **Configuration**: `.env` + `config.py` - Environment-driven settings

## Development Philosophy

### Think Holistically & Reuse First
**Before writing new code, always:**
1. **Search existing codebase** - Look for similar functionality already implemented
2. **Leverage service getters** - Use `get_position_tracker()`, `get_risk_manager()`, etc.
3. **Extend existing patterns** - Build on established conventions rather than creating new ones
4. **Minimal viable solution** - Choose the simplest approach that solves the problem

### Code Reuse Opportunities
```python
# GOOD: Reuse existing services
risk_manager = get_risk_manager()
position_size = risk_manager.calculate_position_size(signal, balance)

# AVOID: Reimplementing risk calculations
position_size = balance * 0.02  # Don't hardcode risk logic
```

### Leverage Existing Infrastructure
- **Configuration**: Use `config.py` environment variables instead of hardcoding
- **Logging**: Use `log_error_to_csv()` instead of print statements
- **State**: Update `bot_status` dictionary rather than creating separate state
- **API patterns**: Follow existing Binance client usage patterns

## Key Development Guidelines

### Error Handling Pattern
Always use the established error logging system:
```python
log_error_to_csv(str(e), "ERROR_TYPE", "function_name", "SEVERITY")
```

### API Rate Limiting
Respect Binance limits with built-in protection:
```python
# Check bot_status before API calls
if not bot_status.get('api_connected', False):
    return  # Skip API operation
```

### Configuration Changes
Prefer environment variables over hardcoded values:
```python
# Good: Environment-driven
THRESHOLD = float(os.getenv('RISK_THRESHOLD', '5.0'))

# Avoid: Hardcoded values
THRESHOLD = 5.0  # Hard to modify in production
```

### ML Model Integration
Always check model availability before use:
```python
if SKLEARN_AVAILABLE and model_exists('trend_model'):
    prediction = ml_predictor.predict_trend(df)
```

### Memory Considerations
For Render deployment, use memory-efficient patterns:
```python
# Use memory-efficient data processing
df = optimize_dataframe_memory(df)  # Convert dtypes
gc.collect()  # Explicit cleanup after heavy operations
```

### Simplicity-First Problem Solving
**When implementing new features:**
1. **Check existing functions** - Search for similar logic before writing new code
2. **Use established patterns** - Follow existing ML, trading, or API patterns
3. **Minimize complexity** - Prefer simple, readable solutions over clever optimizations
4. **Reuse components** - Leverage existing classes, functions, and utilities

### Code Reuse Checklist
- [ ] Does this functionality exist in another form?
- [ ] Can I extend an existing service instead of creating new one?
- [ ] Am I following established configuration patterns?
- [ ] Can I reuse existing error handling/logging?
- [ ] Does this fit into the current architecture?

### Critical: No Fabricated Data
**NEVER create, invent, or use:**
- Mock/demo/placeholder data
- Fabricated historical prices or trading data
- Made-up API responses or test fixtures
- Simulated trading results or performance metrics
- Fake timestamps, balances, or transaction records

**Always use:**
- Real data from actual sources (Binance API, Supabase, CSV files)
- Actual configuration values from `.env` or `config.py`
- Live API responses during development/testing
- Real-time market data or historical data from legitimate sources

### Documentation Guidelines
**Avoid creating new .md files** - Reuse and update existing documentation:
- README.md - Main project overview
- RENDER_DEPLOYMENT.md - Cloud deployment
- INCREMENTAL_LEARNING_GUIDE.md - ML training
- SIMPLE_ML_AUTOMATION.md - ML scheduler
- SUPABASE_SETUP_GUIDE.md - Database setup

**Only create new documentation when:**
- Introducing entirely new subsystems
- Major architectural changes requiring detailed explanation
- Creating guides for complex workflows not covered elsewhere

This bot is a production system handling real money trades. Always test thoroughly in testnet mode before deploying changes that affect trading logic.