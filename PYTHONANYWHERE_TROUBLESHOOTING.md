# 🚨 CRYPTIX-ML PythonAnywhere Troubleshooting Guide

## Current Issue: "Something went wrong" Error

Based on the error screen, your PythonAnywhere app is not loading properly. Let's fix this step by step.

## 🔍 Step 1: Check Error Logs

1. **Log into PythonAnywhere dashboard**
2. **Go to Web tab**
3. **Click on your web app**
4. **Scroll down to "Log files" section**
5. **Check these logs:**
   - **Error log** - Most important for debugging
   - **Server log** - Shows startup issues
   - **Access log** - Shows if requests are reaching your app

## 🔧 Step 2: Most Common Issues & Fixes

### Issue #1: WSGI Configuration Problem
**Symptoms:** "Something went wrong" error
**Fix:**
1. Go to **Web** tab → **WSGI configuration file**
2. Replace the entire content with this corrected version:

```python
import sys
import os

# Add your project directory to Python path
project_home = '/home/cryptix/CRYPTIX-ML-02'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Change to project directory
os.chdir(project_home)

# Import your Flask app
from web_bot import app as application

# This is what PythonAnywhere will use
if __name__ == "__main__":
    application.run()
```

### Issue #2: Missing Dependencies
**Symptoms:** Import errors in logs
**Fix:**
```bash
# In PythonAnywhere console:
cd CRYPTIX-ML-02
pip3.9 install --user -r requirements.txt
```

### Issue #3: Environment Variables Not Set
**Symptoms:** API connection errors
**Fix:**
1. Create/edit `.env` file in your project directory
2. Add your actual credentials:
```
API_KEY=your_actual_binance_api_key
API_SECRET=your_actual_binance_secret
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Issue #4: File Permissions
**Symptoms:** Permission denied errors
**Fix:**
```bash
# In PythonAnywhere console:
chmod -R 755 /home/cryptix/CRYPTIX-ML-02
```

### Issue #5: Python Path Issues
**Symptoms:** Module not found errors
**Fix:**
1. Verify your project is in the correct location: `/home/cryptix/CRYPTIX-ML-02`
2. Update WSGI file path to match exactly

## 🔍 Step 3: Debug Process

### Check #1: Verify Files Uploaded
```bash
# In PythonAnywhere console:
ls -la /home/cryptix/CRYPTIX-ML-02/
```
Should show: `web_bot.py`, `config.py`, `requirements.txt`, etc.

### Check #2: Test Import
```bash
# In PythonAnywhere console:
cd /home/cryptix/CRYPTIX-ML-02
python3.9 -c "from web_bot import app; print('Import successful!')"
```

### Check #3: Test Basic Flask
```bash
# In PythonAnywhere console:
cd /home/cryptix/CRYPTIX-ML-02
python3.9 -c "from web_bot import app; print(f'Flask app: {app}')"
```

## 🛠️ Step 4: Create Minimal Test App

If the main app still fails, create a test file to verify basic setup:

1. Create `/home/cryptix/test_app.py`:
```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "CRYPTIX-ML Test - Flask is working!"

if __name__ == "__main__":
    app.run()
```

2. Update WSGI file temporarily:
```python
from test_app import app as application
```

3. Reload web app and test

## 🔄 Step 5: Reload and Test

After making changes:
1. **Go to Web tab**
2. **Click "Reload" button**
3. **Wait 10-15 seconds**
4. **Visit your site again**

## 📋 Step 6: Specific Error Solutions

### If you see "ImportError: No module named 'xyz'"
```bash
pip3.9 install --user xyz
```

### If you see "Permission denied"
```bash
chmod 755 /home/cryptix/CRYPTIX-ML-02/web_bot.py
```

### If you see "Flask app not found"
Check WSGI file imports exactly match your file structure.

## 🎯 Quick Fix Checklist

Run these commands in PythonAnywhere console:

```bash
# 1. Navigate to project
cd /home/cryptix/CRYPTIX-ML-02

# 2. Check files exist
ls -la web_bot.py config.py

# 3. Install dependencies
pip3.9 install --user flask python-binance pandas python-dotenv

# 4. Test import
python3.9 -c "import web_bot; print('Success!')"

# 5. Check Flask app
python3.9 -c "from web_bot import app; print('Flask app loaded!')"
```

## 🆘 If Still Not Working

### Option 1: Start Fresh
1. Delete current web app in PythonAnywhere
2. Create new web app
3. Follow deployment guide step by step

### Option 2: Simplified Deployment
1. Upload only essential files: `web_bot.py`, `config.py`, `requirements.txt`
2. Install minimal dependencies first
3. Add features gradually

### Option 3: Alternative Hosting
If PythonAnywhere continues to have issues:
- **Render.com** - Better for complex Flask apps
- **Railway.app** - Modern deployment platform
- **Heroku** - Reliable but paid

## 📞 Get Help

1. **Check PythonAnywhere help**: https://help.pythonanywhere.com/
2. **PythonAnywhere forums**: Active community support
3. **Contact PythonAnywhere support**: If you have paid account

---

**Next Step: Check your error logs first, then apply the appropriate fix above.**
