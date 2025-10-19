# CRYPTIX-ML Migration Issue Fix Guide

## Problem
The migration script is failing with the error:
```
value too long for type character varying(20)
```

This happens because the `source` field in the `trades` table was initially set to VARCHAR(20), but the migration script uses `'positions_json_import'` (21 characters) as the source value.

## Solution

### Step 1: Fix the Database Schema

1. **Open your Supabase Dashboard**
   - Go to https://supabase.com/dashboard
   - Navigate to your CRYPTIX-ML project

2. **Open SQL Editor**
   - Click on "SQL Editor" in the left sidebar
   - Click "New query"

3. **Run the Schema Fix**
   Copy and paste the following SQL code and run it:

```sql
-- Migration to fix source field length issue
-- Run this in Supabase SQL Editor before running the migration script

-- 1. First, drop the constraint if it exists
ALTER TABLE trades DROP CONSTRAINT IF EXISTS trades_source_check;

-- 2. Increase the VARCHAR length for source field
ALTER TABLE trades ALTER COLUMN source TYPE VARCHAR(50);

-- 3. Add back the constraint with new allowed values
ALTER TABLE trades ADD CONSTRAINT trades_source_check 
    CHECK (source IN ('bot', 'binance_history', 'manual', 'positions_json_import'));

-- 4. Verify the change
SELECT column_name, data_type, character_maximum_length 
FROM information_schema.columns 
WHERE table_name = 'trades' AND column_name = 'source';
```

4. **Verify the Fix**
   The last query should show that the `source` field now has a `character_maximum_length` of 50.

### Step 2: Re-run the Migration

After fixing the schema, run the migration script again:

```bash
python migrate_to_supabase.py
```

## Alternative: Reset Migration Status

If you need to re-run the migration after partial completion:

```sql
-- Reset migration status to allow re-running
UPDATE migration_status 
SET status = 'pending', completed_at = NULL, error_message = NULL 
WHERE migration_name IN ('positions_json_import', 'binance_historical_import');
```

## Prevention

This issue has been fixed in the updated schema files. New deployments should use the corrected schema with VARCHAR(50) for the source field.

## Files Modified

- `supabase_schema.sql` - Updated to use VARCHAR(50) for source field
- `fix_source_field_migration.sql` - Schema fix for existing databases
- `migrate_to_supabase.py` - Added better error handling and instructions

## Support

If you continue to have issues:

1. Check that your SUPABASE_URL and SUPABASE_SERVICE_KEY are correctly set
2. Verify you have the necessary permissions in Supabase
3. Check the migration logs in `logs/migration.log`
4. Ensure your Supabase project has the correct table structure

For more help, refer to the SUPABASE_SETUP_GUIDE.md file.