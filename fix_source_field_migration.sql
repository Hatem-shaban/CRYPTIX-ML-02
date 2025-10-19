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