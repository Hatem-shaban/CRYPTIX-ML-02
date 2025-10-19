# Quick Fix for Supabase Migration Error

## The Issue
You're getting the error:
```
value too long for type character varying(20)
```

This is because the database schema has a 20-character limit on the `source` field, but the migration script is trying to insert `'positions_json_import'` which is 21 characters.

## Quick Solution

### Option 1: Use the SQL Fix (Recommended)

1. **Go to your Supabase Dashboard** → SQL Editor
2. **Run this SQL command:**

```sql
-- Fix the source field length issue
ALTER TABLE trades DROP CONSTRAINT IF EXISTS trades_source_check;
ALTER TABLE trades ALTER COLUMN source TYPE VARCHAR(50);
ALTER TABLE trades ADD CONSTRAINT trades_source_check 
    CHECK (source IN ('bot', 'binance_history', 'manual', 'positions_json_import'));
```

3. **Re-run the migration:**
```bash
python migrate_to_supabase.py
```

### Option 2: Temporary Workaround (if SQL fix doesn't work)

Edit the `supabase_position_tracker.py` file and change line where it says:
```python
source='positions_json_import'
```
to:
```python
source='json_import'
```

## Environment Variables

Make sure you have these set in your environment:
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_SERVICE_KEY` - Your service role key (not anon key)

You can check if they're set by running:
```bash
echo $SUPABASE_URL
echo $SUPABASE_SERVICE_KEY
```

## Next Steps

After fixing the schema issue, the migration should complete successfully and you'll be able to use Supabase for position tracking.

The error indicates that 6 positions were found for migration, so once this issue is resolved, your historical data should be properly imported.