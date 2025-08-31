# CRYPTIX-ML Deployment Guide for PythonAnywhere

## 🐍 Why PythonAnywhere is Perfect for Your Project

PythonAnywhere is specifically designed for Python applications and provides:
- ✅ Full Python environment support
- ✅ Pre-installed scientific libraries (numpy, pandas, scikit-learn)
- ✅ Easy Flask deployment with WSGI
- ✅ Free tier available
- ✅ File manager and console access
- ✅ Scheduled tasks support

## 🚀 Step-by-Step Deployment

### Step 1: Create PythonAnywhere Account
1. Go to [PythonAnywhere](https://www.pythonanywhere.com)
2. Sign up for a free account (or upgrade for more features)
3. Verify your email and log in

### Step 2: Upload Your Code

**Option A: Using Git (Recommended)**
1. Open a **Bash console** in PythonAnywhere
2. Clone your repository:
   ```bash
   git clone https://github.com/Hatem-shaban/CRYPTIX-ML-02.git
   cd CRYPTIX-ML-02
   ```

**Option B: Upload Files**
1. Use the **Files** tab in PythonAnywhere dashboard
2. Create a new directory: `/home/cryptix/CRYPTIX-ML`
3. Upload all your Python files

### Step 3: Install Dependencies
1. Open a **Bash console**
2. Navigate to your project directory:
   ```bash
   cd CRYPTIX-ML-02  # or your project directory
   ```
3. Install requirements:
   ```bash
   pip3.9 install --user -r requirements.txt
   ```

### Step 4: Set Up Environment Variables
1. Edit your `.env` file or create one:
   ```bash
   nano .env
   ```
2. Add your credentials:
   ```
   API_KEY=your_binance_api_key
   API_SECRET=your_binance_secret
   TELEGRAM_BOT_TOKEN=your_telegram_token
   TELEGRAM_CHAT_ID=your_chat_id
   FLASK_DEBUG=0
   ```

### Step 5: Create WSGI File
PythonAnywhere will automatically create a WSGI file, but you need to configure it:

1. Go to **Web** tab in PythonAnywhere dashboard
2. Click **"Add a new web app"**
3. Choose **Manual configuration** and **Python 3.9**
4. Edit the WSGI configuration file (it will be at `/var/www/cryptix_pythonanywhere_com_wsgi.py`)

The WSGI file content should be:
```python
import sys
import os

# Add your project directory to Python path
path = '/home/cryptix/CRYPTIX-ML-02'  # Update this path
if path not in sys.path:
    sys.path.append(path)

# Set environment variables
os.environ['API_KEY'] = 'your_binance_api_key'
os.environ['API_SECRET'] = 'your_binance_secret'
os.environ['TELEGRAM_BOT_TOKEN'] = 'your_telegram_token'
os.environ['TELEGRAM_CHAT_ID'] = 'your_chat_id'

from web_bot import app as application

if __name__ == "__main__":
    application.run()
```

### Step 6: Configure Web App
1. In the **Web** tab:
   - **Source code**: `/home/cryptix/CRYPTIX-ML-02`
   - **Working directory**: `/home/cryptix/CRYPTIX-ML-02`
   - **WSGI configuration file**: Auto-generated path
2. Click **"Reload"** to start your app

### Step 7: Test Your Deployment
1. Visit your app URL: `https://cryptix.pythonanywhere.com`
2. Check that the dashboard loads
3. Test API endpoints
4. Verify Telegram notifications work

## 🔧 Advanced Configuration

### Static Files (if needed)
If you have CSS/JS files, configure them in the **Web** tab:
- **URL**: `/static/`
- **Directory**: `/home/cryptix/CRYPTIX-ML-02/static/`

### Scheduled Tasks
For automated trading, set up scheduled tasks:
1. Go to **Tasks** tab
2. Create a new scheduled task
3. Command: `python3.9 /home/cryptix/CRYPTIX-ML-02/trading_scheduler.py`

### Database (if needed)
PythonAnywhere provides MySQL databases for persistent storage.

## 🐛 Troubleshooting

### Common Issues:

**1. Import Errors**
- Check that all files are uploaded
- Verify Python path in WSGI file
- Install missing packages with `pip3.9 install --user package_name`

**2. Environment Variables Not Loading**
- Add them directly to WSGI file
- Or use `python-dotenv` in your code
- Check file permissions

**3. API Connection Issues**
- Verify credentials in environment variables
- Check if IP is whitelisted on Binance
- Test API connection in console

**4. Web App Won't Start**
- Check error logs in **Web** tab
- Verify WSGI file syntax
- Check file paths are correct

### Error Logs
- Check error logs in **Web** tab → **Log files**
- Use `print()` statements for debugging
- View logs in real-time

## 💰 Pricing & Limits

### Free Account:
- 1 web app
- 512MB disk space
- Always-on tasks: No
- Custom domains: No

### Paid Plans ($5/month+):
- Multiple web apps
- More disk space
- Always-on tasks
- Custom domains
- SSH access

## 📈 Production Tips

1. **Use Always-On Tasks** (paid feature) for continuous trading
2. **Set up monitoring** with error notifications
3. **Regular backups** of your trading data
4. **Monitor resource usage** to avoid limits
5. **Use scheduled tasks** for periodic maintenance

## 🔗 Useful Links

- [PythonAnywhere Help](https://help.pythonanywhere.com/)
- [Flask on PythonAnywhere](https://help.pythonanywhere.com/pages/Flask/)
- [Environment Variables Guide](https://help.pythonanywhere.com/pages/environment-variables-for-web-apps/)

---

**🎯 Your app will be live at: `https://cryptix.eu.pythonanywhere.com`** (EU server)

**Important Note:** Since you're using EU PythonAnywhere, your URLs and configurations will use the `.eu.pythonanywhere.com` domain instead of the regular `.pythonanywhere.com`

Follow these steps and your CRYPTIX-ML bot will be running smoothly on PythonAnywhere!
