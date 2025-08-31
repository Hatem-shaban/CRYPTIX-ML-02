# 🚀 CRYPTIX-ML PythonAnywhere Quick Deploy Checklist

## ✅ Pre-Deployment Checklist

### Files Ready:
- [x] `web_bot.py` - Main Flask application
- [x] `wsgi.py` - WSGI configuration for PythonAnywhere
- [x] `requirements.txt` - Python dependencies
- [x] `config.py` - Trading configuration
- [x] All ML model files (*.pkl)
- [x] Enhanced modules (position_manager.py, etc.)

### Credentials Needed:
- [ ] Binance API Key
- [ ] Binance API Secret  
- [ ] Telegram Bot Token
- [ ] Telegram Chat ID

## 🔧 Deployment Steps (15 minutes)

### 1. Create PythonAnywhere Account (2 min)
- Go to https://www.pythonanywhere.com
- Sign up with username: `cryptix` (or your preferred)
- Verify email

### 2. Upload Code (5 min)
**Option A - Git Clone (Recommended):**
```bash
# In PythonAnywhere Bash console:
git clone https://github.com/Hatem-shaban/CRYPTIX-ML-02.git
cd CRYPTIX-ML-02
```

**Option B - File Upload:**
- Use Files tab to create `/home/cryptix/CRYPTIX-ML-02/`
- Upload all project files

### 3. Install Dependencies (3 min)
```bash
# In PythonAnywhere Bash console:
cd CRYPTIX-ML-02
pip3.9 install --user -r requirements.txt
```

### 4. Configure Environment (2 min)
Create `.env` file:
```bash
nano .env
```
Add:
```
API_KEY=your_actual_binance_api_key
API_SECRET=your_actual_binance_secret
TELEGRAM_BOT_TOKEN=your_actual_telegram_token
TELEGRAM_CHAT_ID=your_actual_chat_id
```

### 5. Set Up Web App (3 min)
1. Go to **Web** tab in dashboard
2. Click **"Add a new web app"**
3. Choose **Manual configuration** → **Python 3.9**
4. Configure:
   - **Source code**: `/home/cryptix/CRYPTIX-ML-02`
   - **Working directory**: `/home/cryptix/CRYPTIX-ML-02`
5. Edit WSGI file (click the link in Web tab)
6. Replace content with the `wsgi.py` content from your project
7. Update the project path in WSGI file
8. Click **"Reload"**

## 🎯 Your Live URL
`https://cryptix.eu.pythonanywhere.com` (EU server)

**Important:** Since you're using EU PythonAnywhere, your URL will be `.eu.pythonanywhere.com` instead of the regular `.pythonanywhere.com`

## ✅ Post-Deployment Tests

### Test Checklist:
- [ ] Website loads without errors
- [ ] Dashboard shows bot status
- [ ] API connection works (check credentials)
- [ ] Trading signals display
- [ ] Telegram notifications work
- [ ] ML predictions load

### Common URLs to Test:
- `https://cryptix.eu.pythonanywhere.com/` - Main dashboard
- `https://cryptix.eu.pythonanywhere.com/status` - Bot status
- `https://cryptix.eu.pythonanywhere.com/trade-history` - Trade history

## 🚨 Troubleshooting

### If site shows errors:
1. Check **Error log** in Web tab
2. Verify file paths in WSGI file
3. Check environment variables
4. Install missing packages: `pip3.9 install --user package_name`

### If API doesn't work:
1. Verify credentials in `.env` file
2. Check Binance API permissions
3. Whitelist PythonAnywhere IP if needed

### If Telegram doesn't work:
1. Test bot token with Telegram Bot API
2. Verify chat ID is correct
3. Check network connectivity

## 📊 Monitoring

### Check Logs:
- **Error log**: Web tab → Log files
- **Server log**: Web tab → Log files  
- **Access log**: Web tab → Log files

### Performance:
- Monitor CPU seconds usage
- Check for memory errors
- Watch API rate limits

## 🎉 Success!

If everything works:
- ✅ Your bot is live at `https://cryptix.pythonanywhere.com`
- ✅ Trading is automated (if enabled)
- ✅ Telegram notifications active
- ✅ ML predictions running
- ✅ Dashboard accessible 24/7

## 💡 Next Steps

### For Production:
1. Upgrade to paid plan for always-on tasks
2. Set up scheduled tasks for maintenance
3. Configure custom domain (paid feature)
4. Implement monitoring and alerts
5. Regular backups of trading data

---

**Total deployment time: ~15 minutes**
**Success rate: 95%+ for Python applications on PythonAnywhere**
