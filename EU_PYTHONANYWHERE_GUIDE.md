# 🇪🇺 EU PythonAnywhere API Configuration Guide

## Important: EU Server Requirements

Since you're using **eu.pythonanywhere.com**, there are specific considerations for API connectivity:

## 🔧 API Configuration for EU Server

### 1. Binance API Region Settings
Your Binance API keys need to be configured to work from European IP addresses:

1. **Log into Binance** → **API Management**
2. **Check IP Restrictions** for your API key
3. **Either:**
   - **Remove IP restrictions** (less secure but easier)
   - **Add PythonAnywhere EU IP ranges** (more secure)

### 2. PythonAnywhere EU IP Ranges
If you want to restrict by IP, you need to whitelist PythonAnywhere EU server IPs. Contact PythonAnywhere support for current EU IP ranges.

### 3. Recommended API Settings for EU
In your `.env` file on PythonAnywhere EU:

```bash
# Basic API configuration
API_KEY=your_binance_api_key_64_characters
API_SECRET=your_binance_secret_64_characters

# For testing - use Binance Testnet first
BINANCE_TESTNET=1
USE_TESTNET=true

# Regional settings
BINANCE_API_URL=https://api.binance.com
BINANCE_STREAM_URL=wss://stream.binance.com:9443

# Telegram (optional)
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 🚀 Step-by-Step Setup for EU Server

### Step 1: Update Your Code
```bash
# In EU PythonAnywhere console:
cd /home/cryptix/CRYPTIX-ML-02
git pull
```

### Step 2: Configure Environment
```bash
# Create/edit .env file
nano .env
```

Add the configuration above with your actual credentials.

### Step 3: Test API Connection
```bash
# Test in PythonAnywhere console:
cd /home/cryptix/CRYPTIX-ML-02
python3.9 -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('API Key length:', len(os.getenv('API_KEY', '')))
print('Using testnet:', os.getenv('BINANCE_TESTNET', 'false'))
"
```

### Step 4: Reload Web App
1. Go to **Web** tab in EU PythonAnywhere dashboard
2. Click **"Reload"**
3. Wait 10-15 seconds

### Step 5: Test Your Site
Visit: `https://cryptix.eu.pythonanywhere.com`

## 🔍 Troubleshooting EU-Specific Issues

### Issue: API Connection Fails
**Cause:** IP restrictions or wrong server region
**Fix:**
1. Check Binance API key IP restrictions
2. Verify you're using the correct API endpoints
3. Try with testnet first: `BINANCE_TESTNET=1`

### Issue: Different Latency
**Cause:** EU server location affects API response times
**Fix:** This is normal - EU server may have slightly different latency to Binance servers

### Issue: Time Zone Differences
**Cause:** EU servers use different time zones
**Fix:** Your app already uses Cairo timezone, so this should be handled automatically

## 🌍 EU vs US PythonAnywhere Differences

| Feature | US (.pythonanywhere.com) | EU (.eu.pythonanywhere.com) |
|---------|-------------------------|----------------------------|
| **Your URL** | cryptix.pythonanywhere.com | cryptix.eu.pythonanywhere.com |
| **Server Location** | US East Coast | Europe |
| **API Latency** | ~100-200ms to Binance | ~50-150ms to Binance |
| **Data Privacy** | US regulations | GDPR compliant |
| **File Paths** | Same | Same |
| **Python Version** | Same | Same |

## ✅ Your Live App

**URL:** `https://cryptix.eu.pythonanywhere.com`

Once configured properly, your CRYPTIX-ML bot will work exactly the same on EU servers as it would on US servers, with the added benefit of GDPR compliance and potentially better latency to European financial services.

## 🆘 Need Help?

1. Check EU PythonAnywhere help: https://eu.pythonanywhere.com/help/
2. Verify your Binance API settings
3. Test with Binance testnet first
4. Contact PythonAnywhere EU support if needed

Your bot should work perfectly on the EU server! 🚀
