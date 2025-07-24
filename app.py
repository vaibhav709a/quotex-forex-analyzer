import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page config
st.set_page_config(
    page_title="Quotex Forex Analyzer - TwelveData",
    page_icon="💱",
    layout="wide"
)

# API Key Management
def get_api_key():
    """Get API key from multiple sources with priority"""
    # 1. Try Streamlit secrets (for cloud deployment)
    try:
        if "TWELVEDATA_API_KEY" in st.secrets:
            return st.secrets["TWELVEDATA_API_KEY"]
    except:
        pass
    
    # 2. Try environment variable
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if api_key:
        return api_key
    
    # 3. Ask user for input as fallback
    return None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f1f1f;
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    .signal-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .signal-up {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #4CAF50;
    }
    .signal-down {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: 2px solid #f44336;
    }
    .confidence-high {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: 2px solid #2196F3;
    }
    .live-price {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .api-status {
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
    }
    .status-connected {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .setup-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

class TwelveDataAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.session = requests.Session()
        
    def test_connection(self):
        """Test API connection and get quota status"""
        try:
            url = f"{self.base_url}/quota"
            params = {"apikey": self.api_key}
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return True, data
            else:
                return False, f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            return False, str(e)
    
    def get_real_time_price(self, symbol):
        """Get real-time forex price"""
        try:
            url = f"{self.base_url}/price"
            params = {
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data:
                    return {
                        'price': float(data['price']),
                        'timestamp': datetime.now(),
                        'symbol': symbol
                    }
                else:
                    st.error(f"TwelveData Error: {data}")
                    return None
            else:
                st.error(f"API Error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Real-time price error: {e}")
            return None
    
    def get_quote(self, symbol):
        """Get detailed quote with bid/ask"""
        try:
            url = f"{self.base_url}/quote"
            params = {
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'close' in data:
                    return {
                        'price': float(data['close']),
                        'open': float(data['open']),
                        'high': float(data['high']),
                        'low': float(data['low']),
                        'volume': int(data.get('volume', 0)),
                        'change': float(data.get('change', 0)),
                        'percent_change': float(data.get('percent_change', 0)),
                        'timestamp': data.get('datetime', datetime.now().isoformat())
                    }
                else:
                    st.error(f"Quote Error: {data}")
                    return None
        except Exception as e:
            st.error(f"Quote error: {e}")
            return None
    
    def get_time_series(self, symbol, interval="5min", outputsize=100):
        """Get historical time series data"""
        try:
            url = f"{self.base_url}/time_series"
            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "apikey": self.api_key
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'values' in data and data['values']:
                    df_data = []
                    for item in data['values']:
                        df_data.append({
                            'timestamp': pd.to_datetime(item['datetime']),
                            'Open': float(item['open']),
                            'High': float(item['high']),
                            'Low': float(item['low']),
                            'Close': float(item['close']),
                            'Volume': int(item.get('volume', 0))
                        })
                    
                    df = pd.DataFrame(df_data)
                    df.set_index('timestamp', inplace=True)
                    df = df.sort_index()
                    return df
                else:
                    st.error(f"Time series error: {data}")
                    return None
        except Exception as e:
            st.error(f"Time series error: {e}")
            return None

# Technical Indicators without TA-Lib
def calculate_sma(prices, period):
    """Simple Moving Average"""
    return prices.rolling(window=period).mean()

def calculate_ema(prices, period):
    """Exponential Moving Average"""
    return prices.ewm(span=period).mean()

def calculate_rsi(prices, period=14):
    """Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD Indicator"""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd = ema_fast - ema_slow
    macd_signal = calculate_ema(macd, signal)
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Bollinger Bands"""
    sma = calculate_sma(prices, period)
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_technical_indicators(data):
    """Calculate technical indicators for forex analysis"""
    if data is None or len(data) < 50:
        return None
    
    indicators = {}
    
    try:
        # Price data
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Moving Averages
        indicators['EMA_9'] = calculate_ema(close, 9).values
        indicators['EMA_21'] = calculate_ema(close, 21).values
        indicators['SMA_50'] = calculate_sma(close, 50).values
        
        # MACD
        macd, macd_signal, macd_hist = calculate_macd(close)
        indicators['MACD'] = macd.values
        indicators['MACD_signal'] = macd_signal.values
        indicators['MACD_hist'] = macd_hist.values
        
        # RSI
        indicators['RSI'] = calculate_rsi(close).values
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
        indicators['BB_upper'] = bb_upper.values
        indicators['BB_middle'] = bb_middle.values
        indicators['BB_lower'] = bb_lower.values
        
        # Stochastic
        stoch_k, stoch_d = calculate_stochastic(high, low, close)
        indicators['STOCH_K'] = stoch_k.values
        indicators['STOCH_D'] = stoch_d.values
        
        # ATR
        indicators['ATR'] = calculate_atr(high, low, close).values
        
        # Simple trend indicators
        indicators['SMA_20'] = calculate_sma(close, 20).values
        indicators['EMA_12'] = calculate_ema(close, 12).values
        
        return indicators
    except Exception as e:
        st.error(f"Indicator calculation error: {e}")
        return None

def analyze_forex_signals(data, indicators):
    """Advanced signal analysis for forex trading"""
    if indicators is None or data is None:
        return None, 0, "NEUTRAL", []
    
    signals = []
    weights = []
    reasons = []
    
    latest_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2]
    
    # 1. EMA Crossover (High weight)
    if not np.isnan(indicators['EMA_9'][-1]) and not np.isnan(indicators['EMA_21'][-1]):
        ema9_curr = indicators['EMA_9'][-1]
        ema21_curr = indicators['EMA_21'][-1]
        ema9_prev = indicators['EMA_9'][-2]
        ema21_prev = indicators['EMA_21'][-2]
        
        if ema9_curr > ema21_curr and ema9_prev <= ema21_prev:
            signals.append(1)
            weights.append(0.25)
            reasons.append("🟢 EMA Bullish Crossover")
        elif ema9_curr < ema21_curr and ema9_prev >= ema21_prev:
            signals.append(-1)
            weights.append(0.25)
            reasons.append("🔴 EMA Bearish Crossover")
    
    # 2. MACD Signal
    if not np.isnan(indicators['MACD'][-1]) and not np.isnan(indicators['MACD_signal'][-1]):
        macd_curr = indicators['MACD'][-1]
        signal_curr = indicators['MACD_signal'][-1]
        macd_prev = indicators['MACD'][-2]
        signal_prev = indicators['MACD_signal'][-2]
        
        if macd_curr > signal_curr and macd_prev <= signal_prev:
            signals.append(1)
            weights.append(0.20)
            reasons.append("🟢 MACD Bullish Cross")
        elif macd_curr < signal_curr and macd_prev >= signal_prev:
            signals.append(-1)
            weights.append(0.20)
            reasons.append("🔴 MACD Bearish Cross")
    
    # 3. RSI Momentum
    if not np.isnan(indicators['RSI'][-1]):
        rsi = indicators['RSI'][-1]
        rsi_prev = indicators['RSI'][-2]
        
        if rsi < 30 and rsi > rsi_prev:
            signals.append(1)
            weights.append(0.15)
            reasons.append("🟢 RSI Oversold Bounce")
        elif rsi > 70 and rsi < rsi_prev:
            signals.append(-1)
            weights.append(0.15)
            reasons.append("🔴 RSI Overbought Drop")
        elif 40 < rsi < 60:
            if rsi > rsi_prev and latest_close > prev_close:
                signals.append(0.5)
                weights.append(0.10)
                reasons.append("🟡 RSI Neutral Bullish")
            elif rsi < rsi_prev and latest_close < prev_close:
                signals.append(-0.5)
                weights.append(0.10)
                reasons.append("🟡 RSI Neutral Bearish")
    
    # 4. Stochastic
    if not np.isnan(indicators['STOCH_K'][-1]) and not np.isnan(indicators['STOCH_D'][-1]):
        k_curr = indicators['STOCH_K'][-1]
        d_curr = indicators['STOCH_D'][-1]
        k_prev = indicators['STOCH_K'][-2]
        d_prev = indicators['STOCH_D'][-2]
        
        if k_curr > d_curr and k_prev <= d_prev and k_curr < 80:
            signals.append(1)
            weights.append(0.15)
            reasons.append("🟢 Stochastic Bullish Cross")
        elif k_curr < d_curr and k_prev >= d_prev and k_curr > 20:
            signals.append(-1)
            weights.append(0.15)
            reasons.append("🔴 Stochastic Bearish Cross")
    
    # 5. Bollinger Bands
    if not np.isnan(indicators['BB_upper'][-1]) and not np.isnan(indicators['BB_lower'][-1]):
        bb_upper = indicators['BB_upper'][-1]
        bb_lower = indicators['BB_lower'][-1]
        bb_middle = indicators['BB_middle'][-1]
        
        if latest_close <= bb_lower:
            signals.append(1)
            weights.append(0.12)
            reasons.append("🟢 BB Lower Band Touch")
        elif latest_close >= bb_upper:
            signals.append(-1)
            weights.append(0.12)
            reasons.append("🔴 BB Upper Band Touch")
        elif latest_close > bb_middle and prev_close <= bb_middle:
            signals.append(0.5)
            weights.append(0.08)
            reasons.append("🟡 BB Middle Cross Up")
        elif latest_close < bb_middle and prev_close >= bb_middle:
            signals.append(-0.5)
            weights.append(0.08)
            reasons.append("🟡 BB Middle Cross Down")
    
    # 6. Price Momentum
    price_change_pct = ((latest_close - prev_close) / prev_close) * 100
    if abs(price_change_pct) > 0.1:  # Significant move
        if price_change_pct > 0:
            signals.append(0.7)
            weights.append(0.08)
            reasons.append("🟢 Strong Price Momentum Up")
        else:
            signals.append(-0.7)
            weights.append(0.08)
            reasons.append("🔴 Strong Price Momentum Down")
    
    # Calculate weighted signal
    if signals and weights:
        weighted_signal = np.average(signals, weights=weights)
        
        # Enhanced confidence calculation
        signal_strength = abs(weighted_signal)
        signal_count = len([s for s in signals if abs(s) >= 0.5])
        convergence_factor = min(signal_count / 5.0, 1.0)
        
        base_confidence = signal_strength * 85
        convergence_boost = convergence_factor * 15
        
        confidence = min(base_confidence + convergence_boost, 100)
        
        # Adjust for very strong signals
        if signal_strength > 0.8 and signal_count >= 4:
            confidence = max(confidence, 98)
        elif signal_strength > 0.6 and signal_count >= 3:
            confidence = max(confidence, 92)
        elif signal_strength > 0.4:
            confidence = max(confidence, 85)
        
        # Next candle direction
        if weighted_signal > 0.3:
            next_candle = "UP 📈"
        elif weighted_signal < -0.3:
            next_candle = "DOWN 📉"
        else:
            next_candle = "SIDEWAYS ➡️"
        
        return weighted_signal, confidence, next_candle, reasons
    
    return 0, 50, "NEUTRAL ➡️", []

def create_forex_chart(data, indicators, symbol):
    """Create professional forex chart"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price Action', 'MACD', 'RSI', 'Stochastic'),
        row_heights=[0.5, 0.2, 0.15, 0.15]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4757'
        ),
        row=1, col=1
    )
    
    # Add EMAs
    if indicators:
        fig.add_trace(
            go.Scatter(x=data.index, y=indicators['EMA_9'], name='EMA 9', 
                      line=dict(color='#ffa502', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=indicators['EMA_21'], name='EMA 21', 
                      line=dict(color='#3742fa', width=2)),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=data.index, y=indicators['BB_upper'], name='BB Upper', 
                      line=dict(color='gray', dash='dash'), opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=indicators['BB_lower'], name='BB Lower', 
                      line=dict(color='gray', dash='dash'), opacity=0.7),
            row=1, col=1
        )
        
        # MACD
        if 'MACD' in indicators:
            fig.add_trace(
                go.Scatter(x=data.index, y=indicators['MACD'], name='MACD', 
                          line=dict(color='#00d2d3')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=indicators['MACD_signal'], name='Signal', 
                          line=dict(color='#ff6348')),
                row=2, col=1
            )
            
            # MACD Histogram
            colors = ['#00ff88' if x >= 0 else '#ff4757' for x in indicators['MACD_hist']]
            fig.add_trace(
                go.Bar(x=data.index, y=indicators['MACD_hist'], name='Histogram', 
                      marker_color=colors, opacity=0.7),
                row=2, col=1
            )
        
        # RSI
        if 'RSI' in indicators:
            fig.add_trace(
                go.Scatter(x=data.index, y=indicators['RSI'], name='RSI', 
                          line=dict(color='#8c7ae6')),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="#ff4757", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#00ff88", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
        # Stochastic
        if 'STOCH_K' in indicators:
            fig.add_trace(
                go.Scatter(x=data.index, y=indicators['STOCH_K'], name='%K', 
                          line=dict(color='#ff9ff3')),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=indicators['STOCH_D'], name='%D', 
                          line=dict(color='#54a0ff')),
                row=4, col=1
            )
            fig.add_hline(y=80, line_dash="dash", line_color="#ff4757", row=4, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="#00ff88", row=4, col=1)
    
    fig.update_layout(
        title=f"{symbol} TwelveData Analysis",
        xaxis_rangeslider_visible=False,
        height=900,
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#16213e',
        font=dict(color='#eee')
    )
    
    return fig

# Main Application
st.markdown('<h1 class="main-header">💱 QUOTEX FOREX ANALYZER - TwelveData Edition</h1>', unsafe_allow_html=True)

# API Key Setup
api_key = get_api_key()

if not api_key:
    st.markdown("""
    <div class="setup-box">
        <h3>🔑 TwelveData API Setup Required</h3>
        <p>To use this Forex analyzer, you need a TwelveData API key.</p>
        
        <h4>📝 How to get your FREE API key:</h4>
        <ol>
            <li>Visit <a href="https://twelvedata.com/" target="_blank">twelvedata.com</a></li>
            <li>Click "Sign Up" and create a free account</li>
            <li>Go to your dashboard after signup</li>
            <li>Copy your API key</li>
            <li>Enter it below</li>
        </ol>
        
        <p><strong>Free Tier Benefits:</strong></p>
        <ul>
            <li>✅ 800 API calls per day</li>
            <li>✅ Real-time forex data</li>
            <li>✅ Multiple timeframes</li>
            <li>✅ Professional-grade data quality</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Manual API key input
    with st.sidebar:
        st.header("🔑 Enter API Key")
        manual_api_key = st.text_input(
            "TwelveData API Key",
            type="password",
            help="Paste your TwelveData API key here"
        )
        
        if manual_api_key:
            api_key = manual_api_key
        else:
            st.stop()

# Initialize API with the key
api = TwelveDataAPI(api_key)

# Sidebar
with st.sidebar:
    st.header("🔌 API Status")
    
    # Test connection
    connection_status, quota_info = api.test_connection()
    
    if connection_status:
        st.markdown('<div class="api-status status-connected">✅ API Connected!</div>', 
                   unsafe_allow_html=True)
        if isinstance(quota_info, dict):
            st.write(f"**Quota Used:** {quota_info.get('api_calls_used', 'N/A')}")
            st.write(f"**Quota Limit:** {quota_info.get('api_calls_limit', 'N/A')}")
    else:
        st.markdown('<div class="api-status status-error">❌ API Connection Failed</div>', 
                   unsafe_allow_html=True)
        st.error(f"Error: {quota_info}")
        st.stop()
    
    # Forex pairs for TwelveData
    st.header("⚙️ Trading Settings")
    
    major_pairs = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD",
        "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "EUR/AUD", "GBP/AUD",
        "EUR/CAD", "GBP/CAD", "AUD/CAD", "EUR/CHF", "GBP/CHF", "CHF/JPY",
        "CAD/JPY", "AUD/CHF", "NZD/JPY"
    ]
    
    selected_pair = st.selectbox("Select Forex Pair", major_pairs)
    
    # Timeframes for TwelveData
    timeframes = {
        "1 Minute": "1min",
        "5 Minutes": "5min",
        "15 Minutes": "15min",
        "30 Minutes": "30min",
        "1 Hour": "1h",
        "4 Hours": "4h"
    }
    
    selected_timeframe = st.selectbox("Select Timeframe", list(timeframes.keys()))
    interval = timeframes[selected_timeframe]
    
    # Trading options
    st.markdown("---")
    st.markdown("### 🎯 Quotex Settings")
    expiry_times = ["1 min", "5 min", "15 min", "30 min", "1 hour"]
    expiry_time = st.selectbox("Expiry Time", expiry_times)
    
    # Auto refresh
    auto_refresh = st.checkbox("🔄 Auto Refresh (30s)", value=False)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📊 {selected_pair} Live Analysis")
    
    # Get real-time data
    with st.spinner("🔄 Fetching live data from TwelveData..."):
        # Get current quote
        current_quote = api.get_quote(selected_pair)
        
        # Get historical data
        historical_data = api.get_time_series(selected_pair, interval, outputsize=100)
        
        if current_quote and historical_data is not None:
            # Display live price
            price_change = current_quote['change']
            price_change_pct = current_quote['percent_change']
            
            trend_icon = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
            trend_color = "#00ff88" if price_change > 0 else "#ff4757" if price_change < 0 else "gray"
            
            st.markdown(f'''
            <div class="live-price" style="border: 3px solid {trend_color};">
                🎯 LIVE {selected_pair}: {current_quote['price']:.5f} {trend_icon}
                <br>
                <small>Change: {price_change:+.5f} ({price_change_pct:+.2f}%)</small>
            </div>
            ''', unsafe_allow_html=True)
            
            # Calculate indicators
            indicators = calculate_technical_indicators(historical_data)
            
            # Analyze signals
            if indicators:
                signal, confidence, next_candle, reasons = analyze_forex_signals(historical_data, indicators)
                
                # Signal display
                st.subheader("🎯 AI Signal Analysis")
                
                col_sig1, col_sig2 = st.columns(2)
                
                with col_sig1:
                    signal_class = "signal-up" if signal > 0 else "signal-down"
                    signal_direction = "BUY 📈" if signal > 0 else "SELL 📉"
                    
                    st.markdown(f'''
                    <div class="{signal_class} signal-box">
                        <h3>NEXT CANDLE: {next_candle}</h3>
                        <p>Signal: {signal_direction}</p>
                        <p>Strength: {abs(signal):.3f}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col_sig2:
                    confidence_class = "confidence-high"
                    if confidence >= 95:
                        confidence_text = "🔥 VERY HIGH"
                    elif confidence >= 85:
                        confidence_text = "💪 HIGH"
                    elif confidence >= 75:
                        confidence_text = "👍 GOOD"
                    else:
                        confidence_text = "⚠️ LOW"
                    
                    st.markdown(f'''
                    <div class="{confidence_class} signal-box">
                        <h3>Confidence: {confidence:.1f}%</h3>
                        <p>{confidence_text}</p>
                        <p>Expiry: {expiry_time}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Signal reasons
                if reasons:
                    st.subheader("📋 Signal Analysis Details")
                    for i, reason in enumerate(reasons, 1):
                        st.markdown(f"**{i}.** {reason}")
                
                # Create chart
                chart = create_forex_chart(historical_data, indicators, selected_pair)
                st.plotly_chart(chart, use_container_width=True)
            
            else:
                st.error("❌ Unable to calculate indicators. Insufficient data.")
        
        else:
            st.error("❌ Unable to fetch data. Please check your API key or try a different pair.")

with col2:
    st.subheader("📈 Market Information")
    
    if 'current_quote' in locals() and current_quote:
        # Market stats
        st.markdown("### 📊 Session Stats")
        st.metric("Current Price", f"{current_quote['price']:.5f}")
        st.metric("Session Open", f"{current_quote['open']:.5f}")
        st.metric("Session High", f"{current_quote['high']:.5f}")
        st.metric("Session Low", f"{current_quote['low']:.5f}")
        
        # Price change metrics
        change_color = "normal" if current_quote['change'] >= 0 else "inverse"
        st.metric(
            "Price Change", 
            f"{current_quote['change']:+.5f}",
            f"{current_quote['percent_change']:+.2f}%",
            delta_color=change_color
        )
        
        # Calculate daily range
        daily_range = current_quote['high'] - current_quote['low']
        st.metric("Daily Range", f"{daily_range:.5f}")
        
        # Key indicators summary
        if 'indicators' in locals() and indicators:
            st.markdown("### 🎯 Key Indicators")
            
            # RSI Status
            if not np.isnan(indicators['RSI'][-1]):
                rsi_val = indicators['RSI'][-1]
                if rsi_val > 70:
                    rsi_status = "🔴 Overbought"
                    rsi_color = "#ff4757"
                elif rsi_val < 30:
                    rsi_status = "🟢 Oversold"
                    rsi_color = "#00ff88"
                else:
                    rsi_status = "🟡 Neutral"
                    rsi_color = "#ffa502"
                
                st.markdown(f"**RSI (14):** {rsi_val:.1f}")
                st.markdown(f'<div style="color: {rsi_color}; font-weight: bold;">{rsi_status}</div>', 
                           unsafe_allow_html=True)
            
            # MACD Trend
            if not np.isnan(indicators['MACD'][-1]) and not np.isnan(indicators['MACD_signal'][-1]):
                macd_val = indicators['MACD'][-1]
                macd_signal = indicators['MACD_signal'][-1]
                
                if macd_val > macd_signal:
                    macd_status = "🟢 Bullish"
                    macd_color = "#00ff88"
                else:
                    macd_status = "🔴 Bearish"
                    macd_color = "#ff4757"
                
                st.markdown(f"**MACD:** {macd_val:.6f}")
                st.markdown(f'<div style="color: {macd_color}; font-weight: bold;">{macd_status}</div>', 
                           unsafe_allow_html=True)
            
            # Stochastic
            if not np.isnan(indicators['STOCH_K'][-1]):
                stoch_k = indicators['STOCH_K'][-1]
                if stoch_k > 80:
                    stoch_status = "🔴 Overbought"
                    stoch_color = "#ff4757"
                elif stoch_k < 20:
                    stoch_status = "🟢 Oversold"
                    stoch_color = "#00ff88"
                else:
                    stoch_status = "🟡 Mid-range"
                    stoch_color = "#ffa502"
                
                st.markdown(f"**Stochastic:** {stoch_k:.1f}")
                st.markdown(f'<div style="color: {stoch_color}; font-weight: bold;">{stoch_status}</div>', 
                           unsafe_allow_html=True)
        
        # Trading recommendation
        st.markdown("### 💡 Trading Recommendation")
        if 'confidence' in locals():
            if confidence >= 95:
                rec_bg = "#00ff88"
                rec_text = "🎯 EXCELLENT SIGNAL"
                rec_action = "Strong entry recommended"
            elif confidence >= 85:
                rec_bg = "#ffa502"
                rec_text = "💪 GOOD SIGNAL"
                rec_action = "Consider entry"
            elif confidence >= 75:
                rec_bg = "#54a0ff"
                rec_text = "👍 MODERATE SIGNAL"
                rec_action = "Wait for confirmation"
            else:
                rec_bg = "#ff4757"
                rec_text = "⚠️ WEAK SIGNAL"
                rec_action = "Avoid trading"
            
            st.markdown(f'''
            <div style="background: {rec_bg}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                <h4>{rec_text}</h4>
                <p>{rec_action}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # TwelveData API usage
        st.markdown("---")
        st.markdown("### 📊 API Usage")
        if connection_status and isinstance(quota_info, dict):
            used = quota_info.get('api_calls_used', 0)
            limit = quota_info.get('api_calls_limit', 800)
            remaining = limit - used
            usage_pct = (used / limit) * 100 if limit > 0 else 0
            
            st.progress(usage_pct / 100)
            st.write(f"**Used:** {used} / {limit} calls")
            st.write(f"**Remaining:** {remaining} calls")
            
            if usage_pct > 80:
                st.warning("⚠️ API usage is high!")
            elif usage_pct > 90:
                st.error("🚨 API limit nearly reached!")

# Trading Tools Section
st.markdown("---")
st.markdown("### 🛠️ Trading Tools")

col_tools1, col_tools2, col_tools3 = st.columns(3)

with col_tools1:
    st.markdown("#### 💰 Position Calculator")
    account_balance = st.number_input("Account Balance ($)", min_value=10.0, value=100.0, step=10.0)
    risk_percent = st.slider("Risk per Trade (%)", min_value=1, max_value=10, value=2)
    
    trade_amount = (account_balance * risk_percent) / 100
    quotex_payout = st.slider("Quotex Payout (%)", min_value=70, max_value=95, value=85)
    potential_profit = trade_amount * (quotex_payout / 100)
    
    st.info(f"💡 Trade Amount: ${trade_amount:.2f}")
    st.success(f"🎯 Potential Profit: ${potential_profit:.2f}")

with col_tools2:
    st.markdown("#### ⏰ Market Sessions")
    current_utc = datetime.utcnow().hour
    
    # Major forex sessions
    sessions = {
        "Sydney": {"start": 21, "end": 6, "active": False},
        "Tokyo": {"start": 0, "end": 9, "active": False}, 
        "London": {"start": 8, "end": 17, "active": False},
        "New York": {"start": 13, "end": 22, "active": False}
    }
    
    for session, times in sessions.items():
        if times["start"] < times["end"]:
            active = times["start"] <= current_utc < times["end"]
        else:  # Crosses midnight
            active = current_utc >= times["start"] or current_utc < times["end"]
        
        status = "🟢 OPEN" if active else "🔴 CLOSED"
        st.write(f"**{session}:** {status}")

with col_tools3:
    st.markdown("#### 📈 Quick Stats")
    if 'historical_data' in locals() and historical_data is not None:
        # Volatility analysis
        price_changes = historical_data['Close'].pct_change().dropna()
        volatility = price_changes.std() * 100
        
        avg_range = ((historical_data['High'] - historical_data['Low']) / historical_data['Close'] * 100).mean()
        
        st.metric("Volatility", f"{volatility:.3f}%")
        st.metric("Avg Range", f"{avg_range:.3f}%")
        
        # Trend direction
        sma_short = historical_data['Close'].rolling(10).mean().iloc[-1]
        sma_long = historical_data['Close'].rolling(20).mean().iloc[-1]
        
        if sma_short > sma_long:
            trend = "📈 Uptrend"
            trend_color = "#00ff88"
        else:
            trend = "📉 Downtrend" 
            trend_color = "#ff4757"
        
        st.markdown(f'<div style="color: {trend_color}; font-weight: bold;">{trend}</div>', 
                   unsafe_allow_html=True)

# Auto-refresh functionality
if auto_refresh:
    time.sleep(30)
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
    <h3>🚀 QUOTEX FOREX ANALYZER - TwelveData Edition</h3>
    <p><strong>Professional Real-Time Forex Signals for Binary Options</strong></p>
    <p>🔗 <em>Powered by TwelveData API - No TA-Lib Dependencies</em></p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
### ⚖️ Important Disclaimers
- **Real-Time Data**: TwelveData provides professional-grade real-time forex data
- **API Usage**: Monitor your API quota to avoid service interruption  
- **Trading Risk**: Forex and binary options trading involves substantial risk
- **Educational Purpose**: This tool is for analysis and educational purposes
- **Professional Advice**: Consider consulting qualified financial advisors
- **Responsibility**: Users are responsible for their trading decisions

### 💡 Fixed Issues
- ✅ **Removed TA-Lib dependency** - Now works on Streamlit Cloud
- ✅ **Custom technical indicators** - Pure Python/Pandas implementation
- ✅ **Optimized for deployment** - No system-level dependencies
- ✅ **Streamlit Cloud compatible** - Ready for immediate deployment

### 🔧 Technical Indicators Included
- **Moving Averages** (EMA 9/21, SMA 20/50)
- **MACD** with signal line and histogram
- **RSI** (Relative Strength Index)
- **Bollinger Bands** (20-period, 2 std dev)
- **Stochastic Oscillator** (%K and %D)
- **ATR** (Average True Range)
- **Price Momentum** analysis
""")

st.markdown("*🎯 Now fully compatible with Streamlit Cloud deployment!*")