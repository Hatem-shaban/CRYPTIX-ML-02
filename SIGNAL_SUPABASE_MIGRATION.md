# 📊 Signal Logging Migration to Supabase

## Overview
This update migrates signal logging from CSV files to Supabase database, providing:
- **Cloud-based persistence** - Signals stored in PostgreSQL
- **Better performance** - Indexed queries for fast retrieval
- **Reliability** - No file system dependencies
- **Scalability** - Unlimited signal history
- **Dual logging** - Continues CSV backup for safety

## 🚀 Quick Setup

### Step 1: Create Signals Table in Supabase
1. Log in to your [Supabase Dashboard](https://app.supabase.com)
2. Select your CRYPTIX-ML project
3. Go to **SQL Editor**
4. Copy and paste the contents of `supabase_signals_table.sql`
5. Click **Run** to create the table

### Step 2: Deploy Updated Code
The code changes are already integrated:
- ✅ `supabase_position_tracker.py` - Added `log_signal()` and `get_signal_history()` methods
- ✅ `web_bot.py` - Updated signal logging and display functions
- ✅ `/logs/signals` route - Now fetches from Supabase

Simply deploy or restart your application.

## 📋 What Changed

### New Supabase Methods
```python
# In SupabasePositionTracker class:
tracker.log_signal(signal, symbol, price, indicators, reason)
tracker.get_signal_history(symbol=None, limit=100, days=0)
```

### Updated Functions
- **`log_signal_to_csv()`** - Now logs to both Supabase AND CSV (backup)
- **`get_supabase_signal_history()`** - Fetches signals from Supabase
- **`view_signal_logs()`** - Route updated to use Supabase

### Signal Table Schema
```sql
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    signal VARCHAR(10) NOT NULL,  -- BUY, SELL, HOLD
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    rsi DECIMAL(10, 2),
    macd DECIMAL(20, 8),
    macd_trend VARCHAR(20),
    sentiment VARCHAR(20),
    sma5 DECIMAL(20, 8),
    sma20 DECIMAL(20, 8),
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 🔄 Migration Strategy

### Automatic Dual Logging
All new signals are automatically logged to:
1. **Supabase** (primary) - Cloud database
2. **CSV file** (backup) - Local file system

### No Data Loss
- Existing CSV signals remain accessible
- New signals logged to both systems
- Fallback to CSV if Supabase unavailable

### Gradual Migration
You can optionally import historical CSV signals to Supabase using a migration script (not included, but can be created if needed).

## 📊 Usage Examples

### View Signals on Web Interface
Navigate to: `https://your-app.onrender.com/logs/signals`
- Shows latest 100 signals
- Sorted by timestamp (newest first)
- Includes all technical indicators

### Query Signals Programmatically
```python
from web_bot import get_position_tracker

tracker = get_position_tracker()

# Get all signals for a symbol
btc_signals = tracker.get_signal_history(symbol='BTCUSDT', limit=50)

# Get recent signals across all symbols
recent_signals = tracker.get_signal_history(limit=100)

# Get signals from last 7 days
week_signals = tracker.get_signal_history(days=7)
```

## 🎯 Benefits

### Performance
- **Fast Queries**: Indexed database vs. file parsing
- **Efficient Filtering**: SQL-based filtering by symbol, date, signal type
- **Scalability**: No file size limitations

### Reliability
- **Cloud Persistence**: Data survives server restarts/redeployments
- **Automatic Backups**: Supabase handles database backups
- **Dual Logging**: CSV backup ensures no data loss

### Features
- **Real-time Access**: Multiple processes can read signals simultaneously
- **Historical Analysis**: Query signals by any timeframe
- **Better UI**: Web interface loads faster with database queries

## 🛠️ Troubleshooting

### Signals Not Appearing
1. Check Supabase connection in logs
2. Verify `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` environment variables
3. Check SQL Editor for table creation errors
4. CSV backup should still work regardless

### Fallback to CSV
If Supabase is unavailable:
- System automatically falls back to CSV logging
- Web interface shows CSV data instead
- No functionality is lost

### Database Errors
Check the application logs for:
```
✅ Signal logged to Supabase: BUY for BTCUSDT
⚠️ Failed to log signal to Supabase: [error], falling back to CSV
```

## 📈 Next Steps

### Optional Enhancements
1. **CSV Import Tool**: Migrate historical CSV signals to Supabase
2. **Signal Analytics**: Dashboard with signal statistics and performance metrics
3. **Signal Filters**: Web UI filters for symbol, signal type, date range
4. **Export Options**: Download signals as CSV/Excel from web interface

## 🔗 Related Files
- `supabase_signals_table.sql` - Database schema
- `supabase_position_tracker.py` - Supabase integration layer
- `web_bot.py` - Signal logging and web interface
- `supabase_schema.sql` - Original database schema

## ✅ Verification Checklist
- [ ] Run `supabase_signals_table.sql` in Supabase SQL Editor
- [ ] Restart your application
- [ ] Navigate to `/logs/signals` to verify it loads
- [ ] Trigger a signal (manually or wait for bot)
- [ ] Check logs for "✅ Signal logged to Supabase" message
- [ ] Verify signal appears in web interface
- [ ] (Optional) Check Supabase Table Editor to see the signal record

## 📞 Support
If you encounter issues:
1. Check application logs for error messages
2. Verify Supabase table creation was successful
3. Ensure environment variables are set correctly
4. CSV backup continues to work as fallback
