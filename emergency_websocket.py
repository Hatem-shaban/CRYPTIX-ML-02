#!/usr/bin/env python3
"""
WebSocket Price Tracker - Replaces REST API calls for price monitoring
Recommended by Binance to avoid rate limits
"""

import websocket
import json
import threading
import time
from typing import Dict, Callable

class BinanceWebSocketManager:
    """
    WebSocket manager to replace frequent REST API calls
    Uses Binance WebSocket streams for live price updates
    """
    
    def __init__(self):
        self.prices = {}
        self.callbacks = []
        self.ws = None
        self.running = False
    
    def add_price_callback(self, callback: Callable):
        """Add callback for price updates"""
        self.callbacks.append(callback)
    
    def start_price_stream(self, symbols: list):
        """Start WebSocket stream for given symbols"""
        # Convert symbols to lowercase for WebSocket
        streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
        stream_name = "/".join(streams)
        
        url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if isinstance(data, list):
                    for item in data:
                        self._process_ticker_data(item)
                else:
                    self._process_ticker_data(data)
            except Exception as e:
                print(f"WebSocket error: {e}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket closed")
            self.running = False
        
        def on_open(ws):
            print("WebSocket connected - Live price updates active")
            self.running = True
        
        self.ws = websocket.WebSocketApp(url,
                                       on_open=on_open,
                                       on_message=on_message,
                                       on_error=on_error,
                                       on_close=on_close)
        
        # Start in background thread
        def run_websocket():
            self.ws.run_forever()
        
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
    
    def _process_ticker_data(self, data):
        """Process ticker data from WebSocket"""
        if 's' in data and 'c' in data:  # symbol and current price
            symbol = data['s']
            price = float(data['c'])
            self.prices[symbol] = {
                'price': price,
                'timestamp': time.time(),
                'volume': float(data.get('v', 0)),
                'high': float(data.get('h', 0)),
                'low': float(data.get('l', 0))
            }
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(symbol, price)
                except Exception as e:
                    print(f"Callback error: {e}")
    
    def get_price(self, symbol: str) -> float:
        """Get cached price from WebSocket data"""
        if symbol in self.prices:
            return self.prices[symbol]['price']
        return 0.0
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all cached prices"""
        return {symbol: data['price'] for symbol, data in self.prices.items()}
    
    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()

# Global instance
ws_manager = BinanceWebSocketManager()

def start_emergency_price_monitoring(symbols: list):
    """Start emergency WebSocket price monitoring"""
    print("Starting emergency WebSocket price monitoring...")
    ws_manager.start_price_stream(symbols)
    return ws_manager
