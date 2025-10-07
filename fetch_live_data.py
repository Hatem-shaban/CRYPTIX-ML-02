#!/usr/bin/env python3
"""
Fetch live trading data directly from the Render dashboard endpoints
"""

import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import json

def fetch_trade_data_from_dashboard():
    """Fetch trade data from the live dashboard endpoint"""
    
    render_url = "https://cryptix-6yol.onrender.com"
    trades_endpoint = f"{render_url}/logs/trades"
    
    print("🔍 Fetching live trade data from dashboard...")
    print(f"   URL: {trades_endpoint}")
    
    try:
        # Get the HTML page
        response = requests.get(trades_endpoint, timeout=30)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table
        table = soup.find('table')
        if not table:
            print("❌ No table found in the response")
            return None
        
        # Extract headers
        headers = []
        header_row = table.find('thead').find('tr')
        for th in header_row.find_all('th'):
            headers.append(th.text.strip())
        
        print(f"📋 Found table with headers: {headers}")
        
        # Extract data rows
        trades_data = []
        tbody = table.find('tbody')
        
        if tbody:
            rows = tbody.find_all('tr')
            print(f"📊 Found {len(rows)} trade rows")
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= len(headers):
                    trade_row = {}
                    for i, header in enumerate(headers):
                        if i < len(cells):
                            cell_text = cells[i].text.strip()
                            # Clean up currency symbols and formatting
                            if cell_text.startswith('$'):
                                cell_text = cell_text[1:]
                            trade_row[header] = cell_text
                    trades_data.append(trade_row)
        
        return trades_data
        
    except Exception as e:
        print(f"❌ Error fetching trade data: {e}")
        return None

def fetch_signal_data_from_dashboard():
    """Fetch signal data from the live dashboard endpoint"""
    
    render_url = "https://cryptix-6yol.onrender.com"
    signals_endpoint = f"{render_url}/logs/signals"
    
    print("🔍 Fetching live signal data from dashboard...")
    print(f"   URL: {signals_endpoint}")
    
    try:
        # Get the HTML page
        response = requests.get(signals_endpoint, timeout=30)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table
        table = soup.find('table')
        if not table:
            print("❌ No table found in the response")
            return None
        
        # Extract headers
        headers = []
        header_row = table.find('thead').find('tr')
        for th in header_row.find_all('th'):
            headers.append(th.text.strip())
        
        print(f"📋 Found table with headers: {headers}")
        
        # Extract data rows
        signals_data = []
        tbody = table.find('tbody')
        
        if tbody:
            rows = tbody.find_all('tr')
            print(f"📊 Found {len(rows)} signal rows")
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= len(headers):
                    signal_row = {}
                    for i, header in enumerate(headers):
                        if i < len(cells):
                            cell_text = cells[i].text.strip()
                            # Clean up currency symbols and formatting
                            if cell_text.startswith('$'):
                                cell_text = cell_text[1:]
                            # Handle truncated reason text
                            if header == 'Reason' and cell_text.endswith('...'):
                                # Get the full reason from the title attribute if available
                                if 'title' in cells[i].attrs:
                                    cell_text = cells[i]['title']
                            signal_row[header] = cell_text
                    signals_data.append(signal_row)
        
        return signals_data
        
    except Exception as e:
        print(f"❌ Error fetching signal data: {e}")
        return None

def convert_dashboard_data_to_csv_format(trades_data, signals_data):
    """Convert dashboard data to CSV format matching the expected structure"""
    
    print("🔄 Converting dashboard data to CSV format...")
    
    # Convert trades data
    trade_records = []
    if trades_data:
        for trade in trades_data:
            # Map dashboard fields to CSV fields
            csv_record = {
                'timestamp': '',  # Will be derived from cairo_time
                'cairo_time': trade.get('Time (Cairo)', ''),
                'signal': trade.get('Signal', ''),
                'symbol': trade.get('Symbol', ''),
                'quantity': trade.get('Quantity', '0'),
                'price': trade.get('Price', '0'),
                'value': trade.get('Value', '0'),
                'fee': trade.get('Fee', '0'),
                'status': trade.get('Status', ''),
                'order_id': '',  # Not available in dashboard
                'rsi': trade.get('RSI', '0'),
                'macd_trend': trade.get('MACD', ''),
                'sentiment': trade.get('Sentiment', ''),
                'balance_before': 0,  # Not available in dashboard
                'balance_after': 0,   # Not available in dashboard
                'profit_loss': trade.get('P&L', '0')
            }
            
            # Convert cairo_time to timestamp
            try:
                cairo_time = trade.get('Time (Cairo)', '')
                if cairo_time:
                    # Parse the time and convert to timestamp format
                    # Format: "2025-10-07 15:06:18 EEST"
                    dt = datetime.strptime(cairo_time, "%Y-%m-%d %H:%M:%S EEST")
                    csv_record['timestamp'] = dt.strftime("%Y-%m-%d %H:%M:%S EEST")
            except:
                pass
            
            trade_records.append(csv_record)
    
    # Convert signals data
    signal_records = []
    if signals_data:
        for signal in signals_data:
            csv_record = {
                'timestamp': '',  # Will be derived from cairo_time
                'cairo_time': signal.get('Time (Cairo)', ''),
                'signal': signal.get('Signal', ''),
                'symbol': signal.get('Symbol', ''),
                'price': signal.get('Price', '0'),
                'rsi': signal.get('RSI', '0'),
                'macd': signal.get('MACD', '0'),
                'macd_trend': signal.get('MACD Trend', ''),
                'sentiment': signal.get('Sentiment', ''),
                'sma5': signal.get('SMA5', '0'),
                'sma20': signal.get('SMA20', '0'),
                'reason': signal.get('Reason', '')
            }
            
            # Convert cairo_time to timestamp
            try:
                cairo_time = signal.get('Time (Cairo)', '')
                if cairo_time:
                    dt = datetime.strptime(cairo_time, "%Y-%m-%d %H:%M:%S EEST")
                    csv_record['timestamp'] = dt.strftime("%Y-%m-%d %H:%M:%S EEST")
            except:
                pass
            
            signal_records.append(csv_record)
    
    return trade_records, signal_records

def save_data_to_csv(trade_records, signal_records):
    """Save the converted data to CSV files"""
    
    print("💾 Saving data to CSV files...")
    
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Backup existing files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save trade data
    if trade_records:
        trade_file = logs_dir / 'trade_history.csv'
        if trade_file.exists():
            backup_file = logs_dir / f'trade_history.backup_{timestamp}.csv'
            trade_file.rename(backup_file)
            print(f"   📋 Backed up existing trade_history.csv to {backup_file.name}")
        
        # Convert to DataFrame and save
        df = pd.DataFrame(trade_records)
        df.to_csv(trade_file, index=False)
        print(f"   ✅ Saved {len(trade_records)} trade records to trade_history.csv")
    
    # Save signal data
    if signal_records:
        signal_file = logs_dir / 'signal_history.csv'
        if signal_file.exists():
            backup_file = logs_dir / f'signal_history.backup_{timestamp}.csv'
            signal_file.rename(backup_file)
            print(f"   📋 Backed up existing signal_history.csv to {backup_file.name}")
        
        # Convert to DataFrame and save
        df = pd.DataFrame(signal_records)
        df.to_csv(signal_file, index=False)
        print(f"   ✅ Saved {len(signal_records)} signal records to signal_history.csv")

def main():
    """Main function"""
    print("=" * 70)
    print("🌐 CRYPTIX Live Dashboard Data Fetcher")
    print("=" * 70)
    
    # Fetch data from dashboard
    trades_data = fetch_trade_data_from_dashboard()
    signals_data = fetch_signal_data_from_dashboard()
    
    if not trades_data and not signals_data:
        print("❌ Failed to fetch any data from the dashboard")
        return
    
    # Convert to CSV format
    trade_records, signal_records = convert_dashboard_data_to_csv_format(trades_data, signals_data)
    
    # Save to CSV files
    save_data_to_csv(trade_records, signal_records)
    
    print("\n" + "=" * 70)
    print("✅ Live data fetch completed successfully!")
    
    if trade_records:
        latest_trade = trade_records[0] if trade_records else None
        if latest_trade:
            print(f"📊 Latest trade: {latest_trade['signal']} {latest_trade['symbol']} at {latest_trade['cairo_time']}")
    
    if signal_records:
        latest_signal = signal_records[0] if signal_records else None
        if latest_signal:
            print(f"📈 Latest signal: {latest_signal['signal']} {latest_signal['symbol']} at {latest_signal['cairo_time']}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()