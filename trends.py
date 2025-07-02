import yfinance as yf
import pandas as pd
import numpy as np
import functools
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalAnalysisConfig:
    """Configuration class for technical analysis parameters"""
    
    def __init__(self):
        self.sma_short = 20
        self.sma_long = 50
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.bb_period = 20
        self.bb_std = 2
        self.volume_threshold = 1.5
        self.data_period = "3mo"
        self.cache_minutes = 5
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.stoch_k = 14
        self.stoch_d = 3
        self.williams_period = 14
    
    def update_params(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")

# Global configuration instance
config = TechnicalAnalysisConfig()

def get_stock_data(ticker, period="3mo", interval="1d", retries=3):
    """Enhanced data fetching with retry logic and better error handling"""
    for attempt in range(retries):
        try:
            logger.info(f"Fetching data for {ticker} (attempt {attempt + 1})")
            
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Download data with explicit parameters
            data = stock.history(
                period=period,
                interval=interval,
                auto_adjust=True,  # Automatically adjust for splits and dividends
                prepost=False,     # Don't include pre/post market data
                repair=True        # Repair data inconsistencies
            )
            
            if data.empty:
                logger.warning(f"No data returned for {ticker} on attempt {attempt + 1}")
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retry
                continue
            
            # Validate data quality
            if len(data) < 30:  # Need at least 30 days for meaningful analysis
                logger.warning(f"Insufficient data for {ticker}: only {len(data)} rows")
                if attempt < retries - 1:
                    # Try a longer period
                    if period == "3mo":
                        period = "6mo"
                    elif period == "6mo":
                        period = "1y"
                    time.sleep(1)
                    continue
            
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing columns for {ticker}: {missing_columns}")
                continue
            
            # Remove any rows with all NaN values
            data = data.dropna(how='all')
            
            if data.empty:
                logger.warning(f"Data is empty after cleaning for {ticker}")
                continue
            
            logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker} on attempt {attempt + 1}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2)  # Wait longer between retries
            else:
                logger.error(f"Failed to fetch data for {ticker} after {retries} attempts")
    
    return pd.DataFrame()


@functools.lru_cache(maxsize=100)
def cached_download(ticker, period, interval, timestamp):
    """Cache yfinance data for a few minutes"""
    try:
        return yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {str(e)}")
        return pd.DataFrame()

def get_cached_data(ticker, period="3mo", interval="1d", cache_minutes=5):
    """Get data with caching"""
    current_time = datetime.now()
    cache_key = current_time.replace(second=0, microsecond=0)
    # Round to cache_minutes intervals
    cache_key = cache_key.replace(minute=cache_key.minute // cache_minutes * cache_minutes)
    
    return cached_download(ticker, period, interval, cache_key.timestamp())

def get_optimal_period(indicators_needed):
    """Determine optimal data period based on indicators"""
    if 'SMA50' in indicators_needed or 'long_trend' in indicators_needed:
        return "6mo"  # Need more data for SMA50
    elif 'SMA20' in indicators_needed:
        return "3mo"
    else:
        return "2mo"

def compute_RSI(data, period=14):
    """Enhanced RSI calculation with better validation"""
    try:
        if data.empty or len(data) < period + 1:
            return pd.Series(dtype=float)
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Handle edge cases more robustly
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        # Clean up infinite and invalid values
        rsi = rsi.replace([np.inf, -np.inf], np.nan)
        rsi = rsi.fillna(50)  # Neutral RSI for invalid calculations
        
        return rsi
    except Exception as e:
        logger.error(f"RSI calculation error: {e}")
        return pd.Series(dtype=float)

def compute_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands with enhanced error handling"""
    try:
        if data.empty or len(data) < window:
            return pd.Series(dtype=float), pd.Series(dtype=float)
            
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        # Replace infinite values with NaN
        upper_band = upper_band.replace([np.inf, -np.inf], np.nan)
        lower_band = lower_band.replace([np.inf, -np.inf], np.nan)
        
        return upper_band, lower_band
    except Exception as e:
        logger.error(f"Bollinger Bands calculation error: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float)

def compute_MACD(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    try:
        if data.empty or len(data) < slow:
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
            
        exp1 = data['Close'].ewm(span=fast).mean()
        exp2 = data['Close'].ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        # Clean up infinite values
        macd_line = macd_line.replace([np.inf, -np.inf], np.nan)
        signal_line = signal_line.replace([np.inf, -np.inf], np.nan)
        histogram = histogram.replace([np.inf, -np.inf], np.nan)
        
        return macd_line, signal_line, histogram
    except Exception as e:
        logger.error(f"MACD calculation error: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

def compute_stochastic(data, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    try:
        if data.empty or len(data) < k_period:
            return pd.Series(dtype=float), pd.Series(dtype=float)
            
        high_max = data['High'].rolling(window=k_period).max()
        low_min = data['Low'].rolling(window=k_period).min()
        
        # Avoid division by zero
        denominator = high_max - low_min
        denominator = denominator.replace(0, np.nan)
        
        k_percent = ((data['Close'] - low_min) / denominator) * 100
        d_percent = k_percent.rolling(window=d_period).mean()
        
        # Clean up infinite values
        k_percent = k_percent.replace([np.inf, -np.inf], np.nan)
        d_percent = d_percent.replace([np.inf, -np.inf], np.nan)
        
        return k_percent, d_percent
    except Exception as e:
        logger.error(f"Stochastic calculation error: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float)

def compute_williams_r(data, period=14):
    """Calculate Williams %R"""
    try:
        if data.empty or len(data) < period:
            return pd.Series(dtype=float)
            
        high_max = data['High'].rolling(window=period).max()
        low_min = data['Low'].rolling(window=period).min()
        
        # Avoid division by zero
        denominator = high_max - low_min
        denominator = denominator.replace(0, np.nan)
        
        williams_r = ((high_max - data['Close']) / denominator) * -100
        williams_r = williams_r.replace([np.inf, -np.inf], np.nan)
        
        return williams_r
    except Exception as e:
        logger.error(f"Williams %R calculation error: {e}")
        return pd.Series(dtype=float)

def detect_support_resistance(data, window=20):
    """Detect potential support and resistance levels"""
    try:
        if data.empty or len(data) < window * 2:
            return pd.Series(dtype=float), pd.Series(dtype=float)
            
        highs = data['High'].rolling(window=window, center=True).max() == data['High']
        lows = data['Low'].rolling(window=window, center=True).min() == data['Low']
        
        resistance_levels = data.loc[highs, 'High'].tail(5)
        support_levels = data.loc[lows, 'Low'].tail(5)
        
        return support_levels, resistance_levels
    except Exception as e:
        logger.error(f"Support/Resistance detection error: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float)

def calculate_signal_strength(indicators):
    """Calculate overall signal strength from multiple indicators"""
    try:
        score = 0
        max_score = 0
        
        # RSI scoring
        if indicators.get('RSI') is not None:
            max_score += 3
            rsi = indicators['RSI']
            if rsi > config.rsi_overbought:
                score -= 2  # Overbought
            elif rsi < config.rsi_oversold:
                score += 2  # Oversold
            elif 40 <= rsi <= 60:
                score += 1  # Neutral is slightly positive
        
        # SMA scoring
        if indicators.get('SMA20') and indicators.get('SMA50'):
            max_score += 3
            if indicators['SMA20'] > indicators['SMA50']:
                score += 3  # Bullish
            else:
                score -= 1  # Bearish
        
        # MACD scoring
        if indicators.get('MACD') and indicators.get('MACD_Signal'):
            max_score += 2
            if indicators['MACD'] > indicators['MACD_Signal']:
                score += 2  # Bullish
            else:
                score -= 1  # Bearish
        
        # Normalize score
        if max_score > 0:
            strength = (score / max_score) * 100
            return max(-100, min(100, strength))  # Clamp between -100 and 100
        return 0
    except Exception as e:
        logger.error(f"Signal strength calculation error: {e}")
        return 0

def check_alerts(ticker, indicators, current_price):
    """Check for alert conditions"""
    alerts = []
    
    try:
        # Golden Cross alert
        if indicators.get('golden_cross'):
            alerts.append({
                'type': 'GOLDEN_CROSS',
                'ticker': ticker,
                'message': f"🚀 {ticker}: Golden Cross detected!",
                'priority': 'HIGH'
            })
        
        # Death Cross alert
        if indicators.get('death_cross'):
            alerts.append({
                'type': 'DEATH_CROSS',
                'ticker': ticker,
                'message': f"💀 {ticker}: Death Cross detected!",
                'priority': 'HIGH'
            })
        
        # RSI extreme alerts
        rsi = indicators.get('RSI')
        if rsi and (rsi > 80 or rsi < 20):
            alerts.append({
                'type': 'RSI_EXTREME',
                'ticker': ticker,
                'message': f"⚠️ {ticker}: Extreme RSI level: {rsi:.1f}",
                'priority': 'MEDIUM'
            })
        
        # Bollinger Band breakouts
        if indicators.get('bb_breakout_upper'):
            alerts.append({
                'type': 'BB_BREAKOUT_UPPER',
                'ticker': ticker,
                'message': f"🔥 {ticker}: Price broke above upper Bollinger Band!",
                'priority': 'MEDIUM'
            })
        
        if indicators.get('bb_breakout_lower'):
            alerts.append({
                'type': 'BB_BREAKOUT_LOWER',
                'ticker': ticker,
                'message': f"❄️ {ticker}: Price broke below lower Bollinger Band!",
                'priority': 'MEDIUM'
            })
        
        return alerts
    except Exception as e:
        logger.error(f"Alert checking error for {ticker}: {e}")
        return []

def check_indicators(ticker):
    """Get all technical indicators for a given ticker"""
    try:
        period = get_optimal_period(['SMA20', 'SMA50', 'RSI', 'BB'])
        data = get_cached_data(ticker, period=period, interval="1d")
        
        if data.empty:
            return f"❌ No data found for {ticker}"
        
        # Calculate indicators
        data['SMA20'] = data['Close'].rolling(window=config.sma_short).mean()
        data['SMA50'] = data['Close'].rolling(window=config.sma_long).mean()
        
        rsi = compute_RSI(data, config.rsi_period)
        upper_band, lower_band = compute_bollinger_bands(data, config.bb_period, config.bb_std)
        macd_line, signal_line, histogram = compute_MACD(data, config.macd_fast, config.macd_slow, config.macd_signal)
        
        latest_data = data.iloc[-1]
        
        indicators = {
            'Ticker': ticker,
            'Close': round(float(latest_data['Close']), 2),
            'SMA20': round(float(latest_data['SMA20']), 2) if not pd.isna(latest_data['SMA20']) else None,
            'SMA50': round(float(latest_data['SMA50']), 2) if not pd.isna(latest_data['SMA50']) else None,
            'RSI': round(float(rsi.iloc[-1]), 2) if not pd.isna(rsi.iloc[-1]) else None,
            'Upper_Band': round(float(upper_band.iloc[-1]), 2) if not pd.isna(upper_band.iloc[-1]) else None,
            'Lower_Band': round(float(lower_band.iloc[-1]), 2) if not pd.isna(lower_band.iloc[-1]) else None,
            'MACD': round(float(macd_line.iloc[-1]), 4) if not pd.isna(macd_line.iloc[-1]) else None,
            'MACD_Signal': round(float(signal_line.iloc[-1]), 4) if not pd.isna(signal_line.iloc[-1]) else None,
            'MACD_Histogram': round(float(histogram.iloc[-1]), 4) if not pd.isna(histogram.iloc[-1]) else None,
        }
        
        # Calculate signal strength
        indicators['Signal_Strength'] = calculate_signal_strength(indicators)
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error fetching indicators for {ticker}: {str(e)}")
        return f"❌ Error fetching data for {ticker}: {str(e)}"

def check_sma_crossover_fixed(ticker):
    """Fixed SMA crossover detection with better error handling"""
    try:
        print(f"\n🔍 Analyzing SMA crossover for {ticker}")
        
        # Get data with longer period to ensure we have enough for SMA50
        data = get_stock_data(ticker, period="6mo", interval="1d")
        
        if data.empty:
            return f"❌ {ticker}: No data available"
        
        print(f"📊 Retrieved {len(data)} rows of data")
        
        if len(data) < config.sma_long:
            return f"❌ {ticker}: Insufficient data for SMA{config.sma_long} analysis (need {config.sma_long}, got {len(data)})"
        
        # Calculate SMAs
        data['SMA20'] = data['Close'].rolling(window=config.sma_short).mean()
        data['SMA50'] = data['Close'].rolling(window=config.sma_long).mean()
        
        # Get the most recent valid data
        # Find the last row where both SMAs are not NaN
        valid_data = data.dropna(subset=['SMA20', 'SMA50'])
        
        if len(valid_data) < 2:
            return f"❌ {ticker}: Insufficient valid SMA data after calculation"
        
        current_sma20 = float(valid_data['SMA20'].iloc[-1])
        current_sma50 = float(valid_data['SMA50'].iloc[-1])
        prev_sma20 = float(valid_data['SMA20'].iloc[-2])
        prev_sma50 = float(valid_data['SMA50'].iloc[-2])
        
        # Calculate trend strength
        trend_strength = abs(current_sma20 - current_sma50) / current_sma50 * 100
        
        # Golden Cross: SMA20 crosses above SMA50
        if prev_sma20 <= prev_sma50 and current_sma20 > current_sma50:
            return f"🚀 {ticker}: GOLDEN CROSS! SMA20 (${current_sma20:.2f}) crossed above SMA50 (${current_sma50:.2f}) - Strong bullish signal! Trend strength: {trend_strength:.1f}%"
        
        # Death Cross: SMA20 crosses below SMA50
        elif prev_sma20 >= prev_sma50 and current_sma20 < current_sma50:
            return f"💀 {ticker}: DEATH CROSS! SMA20 (${current_sma20:.2f}) crossed below SMA50 (${current_sma50:.2f}) - Strong bearish signal! Trend strength: {trend_strength:.1f}%"
        
        # No crossover, but show current trend
        elif current_sma20 > current_sma50:
            return f"📈 {ticker}: Bullish trend - SMA20 (${current_sma20:.2f}) above SMA50 (${current_sma50:.2f}) - Trend strength: {trend_strength:.1f}%"
        
        else:
            return f"📉 {ticker}: Bearish trend - SMA20 (${current_sma20:.2f}) below SMA50 (${current_sma50:.2f}) - Trend strength: {trend_strength:.1f}%"
            
    except Exception as e:
        logger.error(f"Error analyzing SMA crossover for {ticker}: {str(e)}")
        return f"❌ Error analyzing {ticker}: {str(e)}"


def get_comprehensive_analysis(ticker):
    """Enhanced comprehensive technical analysis with signal strength"""
    try:
        period = get_optimal_period(['SMA20', 'SMA50', 'RSI', 'BB', 'MACD'])
        data = get_cached_data(ticker, period=period, interval="1d")
        
        if data.empty:
            return f"❌ No data available for {ticker}"
        
        # Calculate all indicators
        data['SMA20'] = data['Close'].rolling(window=config.sma_short).mean()
        data['SMA50'] = data['Close'].rolling(window=config.sma_long).mean()
        
        rsi = compute_RSI(data, config.rsi_period)
        upper_band, lower_band = compute_bollinger_bands(data, config.bb_period, config.bb_std)
        macd_line, signal_line, histogram = compute_MACD(data, config.macd_fast, config.macd_slow, config.macd_signal)
        
        latest = data.iloc[-1]
        current_price = float(latest['Close'].iloc[0])
        
        # Prepare indicators dict for signal strength calculation
        indicators_dict = {
            'RSI': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
            'SMA20': float(latest['SMA20']) if not pd.isna(latest['SMA20']) else None,
            'SMA50': float(latest['SMA50']) if not pd.isna(latest['SMA50']) else None,
            'MACD': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
            'MACD_Signal': float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None,
        }
        
        # Check for crossover signals
        if len(data) >= 2:
            prev_sma20 = float(data['SMA20'].iloc[-2]) if not pd.isna(data['SMA20'].iloc[-2]) else None
            prev_sma50 = float(data['SMA50'].iloc[-2]) if not pd.isna(data['SMA50'].iloc[-2]) else None
            
            if all([prev_sma20, prev_sma50, indicators_dict['SMA20'], indicators_dict['SMA50']]):
                if prev_sma20 <= prev_sma50 and indicators_dict['SMA20'] > indicators_dict['SMA50']:
                    indicators_dict['golden_cross'] = True
                elif prev_sma20 >= prev_sma50 and indicators_dict['SMA20'] < indicators_dict['SMA50']:
                    indicators_dict['death_cross'] = True
        
        # Check for Bollinger Band breakouts
        if not upper_band.empty and not lower_band.empty:
            upper_val = upper_band.iloc[-1]
            lower_val = lower_band.iloc[-1]

            if pd.notna(upper_val):
                upper_val_float = float(upper_val)
                if current_price > upper_val_float:
                    indicators_dict['bb_breakout_upper'] = True

            if pd.notna(lower_val):
                lower_val_float = float(lower_val)
                if current_price < lower_val_float:
                    indicators_dict['bb_breakout_lower'] = True
        
        # Calculate signal strength
        signal_strength = calculate_signal_strength(indicators_dict)
        
        # Check for alerts
        alerts = check_alerts(ticker, indicators_dict, current_price)

        # Build analysis
        analysis = [f"📊 **{ticker} Comprehensive Technical Analysis**"]
        analysis.append(f"💰 Current Price: ${current_price:.2f}")
        
        # Signal strength indicator
        if signal_strength > 50:
            analysis.append(f"🔥 Overall Signal: STRONG BULLISH ({signal_strength:.0f}/100)")
        elif signal_strength > 20:
            analysis.append(f"📈 Overall Signal: BULLISH ({signal_strength:.0f}/100)")
        elif signal_strength > -20:
            analysis.append(f"⚖️ Overall Signal: NEUTRAL ({signal_strength:.0f}/100)")
        elif signal_strength > -50:
            analysis.append(f"📉 Overall Signal: BEARISH ({signal_strength:.0f}/100)")
        else:
            analysis.append(f"🔻 Overall Signal: STRONG BEARISH ({signal_strength:.0f}/100)")
        
        # RSI Analysis
        if indicators_dict['RSI'] is not None:
            current_rsi = indicators_dict['RSI']
            if current_rsi > config.rsi_overbought:
                analysis.append(f"📈 RSI: {current_rsi:.1f} - OVERBOUGHT ⚠️")
            elif current_rsi < config.rsi_oversold:
                analysis.append(f"📉 RSI: {current_rsi:.1f} - OVERSOLD 🔥")
            else:
                analysis.append(f"⚖️ RSI: {current_rsi:.1f} - NEUTRAL")
        else:
            analysis.append("❌ RSI: Insufficient data")
        
        # Bollinger Bands Analysis
        if not upper_band.empty and not lower_band.empty:
            upper_val = upper_band.iloc[-1]
            lower_val = lower_band.iloc[-1]
            
            if pd.notna(upper_val) and pd.notna(lower_val):
                upper_val = float(upper_val)
                lower_val = float(lower_val)
                
                if current_price > upper_val:
                    analysis.append(f"🔥 Price above upper Bollinger Band (${upper_val:.2f}) - Potentially overbought")
                elif current_price < lower_val:
                    analysis.append(f"❄️ Price below lower Bollinger Band (${lower_val:.2f}) - Potentially oversold")
                else:
                    analysis.append(f"📊 Price within Bollinger Bands (${lower_val:.2f} - ${upper_val:.2f}) - Normal range")
            else:
                analysis.append("❌ Bollinger Bands: Insufficient data")
        else:
            analysis.append("❌ Bollinger Bands: Insufficient data")
        
        # MACD Analysis
        if indicators_dict['MACD'] is not None and indicators_dict['MACD_Signal'] is not None:
            macd_val = indicators_dict['MACD']
            signal_val = indicators_dict['MACD_Signal']
            
            if macd_val > signal_val:
                analysis.append(f"📈 MACD: Bullish ({macd_val:.4f} > {signal_val:.4f})")
            else:
                analysis.append(f"📉 MACD: Bearish ({macd_val:.4f} < {signal_val:.4f})")
        else:
            analysis.append("❌ MACD: Insufficient data")
        
        # SMA Trend
        sma_signal = check_sma_crossover_fixed(ticker)
        analysis.append(sma_signal)
        
        # Add alerts if any
        if alerts:
            analysis.append("\n🚨 **ALERTS**:")
            for alert in alerts:
                analysis.append(f"• {alert['message']}")
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis for {ticker}: {str(e)}")
        return f"❌ Error in comprehensive analysis for {ticker}: {str(e)}"

# Crypto Analysis
def get_crypto_analysis(ticker):
    """Perform comprehensive crypto analysis for a given cryptocurrency ticker"""
    try: 
        logger.info(f"Analyzing cryptocurrency: {ticker}")

        # You can likely reuse the get_cached_data function
        # Ensure the ticker is in a format that yfinance understands for crypto
        # For example, for Bitcoin against USD, the ticker might be "BTC-USD"
        period = config.data_period
        data = get_cached_data(ticker, period=period, interval="1d", cache_minutes=config.cache_minutes)

        if data.empty:
            return f"❌ No data available for cryptocurrency: {ticker}"

        # Perform similar technical analysis as you do for stocks
        data['SMA20'] = data['Close'].rolling(window=config.sma_short).mean()
        data['SMA50'] = data['Close'].rolling(window=config.sma_long).mean()

        rsi = compute_RSI(data, config.rsi_period)
        upper_band, lower_band = compute_bollinger_bands(data, config.bb_period, config.bb_std)
        macd_line, signal_line, histogram = compute_MACD(data, config.macd_fast, config.macd_slow, config.macd_signal)
        k_percent, d_percent = compute_stochastic(data, config.stoch_k, config.stoch_d)
        williams_r = compute_williams_r(data, config.williams_period)

        latest = data.iloc[-1]
        current_price = float(latest['Close'])

        analysis = [f"🪙 **{ticker} Cryptocurrency Analysis**"]
        analysis.append(f"💰 Current Price: ${current_price:.2f}")

        # Add relevant indicators to the analysis output
        if not pd.isna(latest['SMA20']):
            analysis.append(f"SMA20: {latest['SMA20']:.2f}")
        if not pd.isna(latest['SMA50']):
            analysis.append(f"SMA50: {latest['SMA50']:.2f}")
        if not pd.isna(rsi.iloc[-1]):
            analysis.append(f"RSI: {rsi.iloc[-1]:.2f}")
        if not upper_band.empty and not pd.isna(upper_band.iloc[-1]):
            analysis.append(f"Bollinger Upper: {upper_band.iloc[-1]:.2f}")
        if not lower_band.empty and not pd.isna(lower_band.iloc[-1]):
            analysis.append(f"Bollinger Lower: {lower_band.iloc[-1]:.2f}")
        if not macd_line.empty and not pd.isna(macd_line.iloc[-1]):
            analysis.append(f"MACD: {macd_line.iloc[-1]:.4f}")
        if not signal_line.empty and not pd.isna(signal_line.iloc[-1]):
            analysis.append(f"MACD Signal: {signal_line.iloc[-1]:.4f}")
        if not histogram.empty and not pd.isna(histogram.iloc[-1]):
            analysis.append(f"MACD Histogram: {histogram.iloc[-1]:.4f}")
        if not k_percent.empty and not pd.isna(k_percent.iloc[-1]):
            analysis.append(f"Stochastic %K: {k_percent.iloc[-1]:.2f}")
        if not d_percent.empty and not pd.isna(d_percent.iloc[-1]):
            analysis.append(f"Stochastic %D: {d_percent.iloc[-1]:.2f}")
        if not williams_r.empty and not pd.isna(williams_r.iloc[-1]):
            analysis.append(f"Williams %R: {williams_r.iloc[-1]:.2f}")

        # You can also include logic for signal strength or specific crypto-related indicators here

        return "\n".join(analysis)

    except Exception as e:
        logger.error(f"Error analyzing cryptocurrency {ticker}: {str(e)}")
        return f"❌ Error analyzing cryptocurrency {ticker}: {str(e)}"

# Enhanced volume analysis with better metrics and insights
def check_volume_analysis(ticker):
   
    try:
        period = get_optimal_period(['volume'])
        data = get_cached_data(ticker, period=period, interval="1d")
        
        if data.empty or 'Volume' not in data.columns:
            return f"❌ {ticker}: No volume data available"
        
        # Calculate volume moving averages
        data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
        data['Volume_MA50'] = data['Volume'].rolling(window=50).mean()
        
        # Calculate On-Balance Volume (OBV)
        data['OBV'] = (data['Volume'] * ((data['Close'] - data['Close'].shift(1)) > 0).astype(int) * 2 - data['Volume']).cumsum()
        
        latest = data.iloc[-1]
        current_volume = int(latest['Volume'])
        volume_ma20 = int(latest['Volume_MA20']) if not pd.isna(latest['Volume_MA20']) else 0
        volume_ma50 = int(latest['Volume_MA50']) if not pd.isna(latest['Volume_MA50']) else 0
        
        # Price and volume relationship
        price_change = (latest['Close'] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
        volume_ratio = current_volume / volume_ma20 if volume_ma20 > 0 else 0
        
        # OBV trend
        obv_trend = "BULLISH" if latest['OBV'] > data['OBV'].iloc[-10] else "BEARISH"
        
        analysis = [f"📊 **{ticker} Enhanced Volume Analysis**"]
        analysis.append(f"📈 Current Volume: {current_volume:,}")
        analysis.append(f"📊 20-day Avg Volume: {volume_ma20:,}")
        analysis.append(f"📊 50-day Avg Volume: {volume_ma50:,}")
        analysis.append(f"📈 OBV Trend: {obv_trend}")
        
        # Volume classification
        if volume_ratio > 2.0:
            analysis.append(f"🚀 EXTREME HIGH volume ({volume_ratio:.1f}x average) - Major interest!")
        elif volume_ratio > config.volume_threshold:
            analysis.append(f"🔥 High volume ({volume_ratio:.1f}x average) - Strong interest!")
        elif volume_ratio < 0.5:
            analysis.append(f"😴 Low volume ({volume_ratio:.1f}x average) - Weak interest")
        else:
            analysis.append(f"⚖️ Normal volume ({volume_ratio:.1f}x average)")
        
        # Volume-Price relationship analysis
        if price_change > 2 and volume_ratio > 1.5:
            analysis.append(f"💪 STRONG BULLISH: Price up {price_change:.1f}% on high volume")
        elif price_change < -2 and volume_ratio > 1.5:
            analysis.append(f"⚠️ STRONG BEARISH: Price down {price_change:.1f}% on high volume")
        elif price_change > 0 and volume_ratio > 1.2:
            analysis.append(f"📈 Bullish: Price up {price_change:.1f}% on above-average volume")
        elif price_change < 0 and volume_ratio > 1.2:
            analysis.append(f"📉 Bearish: Price down {price_change:.1f}% on above-average volume")
        elif abs(price_change) > 3 and volume_ratio < 0.8:
            analysis.append(f"🤔 Suspicious: Large price move ({price_change:+.1f}%) on low volume")
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error in volume analysis for {ticker}: {str(e)}")
        return f"❌ Error in volume analysis for {ticker}: {str(e)}"

def check_momentum_indicators(ticker):
    """Enhanced momentum indicators with better signal interpretation"""
    try:
        period = get_optimal_period(['MACD', 'Stochastic', 'Williams'])
        data = get_cached_data(ticker, period=period, interval="1d")
        
        if data.empty:
            return f"❌ No data available for {ticker}"
        
        # Calculate indicators
        macd_line, signal_line, histogram = compute_MACD(data, config.macd_fast, config.macd_slow, config.macd_signal)
        k_percent, d_percent = compute_stochastic(data, config.stoch_k, config.stoch_d)
        williams_r = compute_williams_r(data, config.williams_period)
        
        # Check if any indicators are empty
        if any(pd.Series([macd_line.empty, signal_line.empty, histogram.empty, k_percent.empty, d_percent.empty, williams_r.empty]).any()):
            return f"❌ {ticker}: Insufficient data for momentum analysis"
        
        latest_idx = -1
        analysis = [f"📊 **{ticker} Momentum Indicators Analysis**"]
        
        # MACD Analysis
        current_macd = float(macd_line.iloc[latest_idx])
        current_signal = float(signal_line.iloc[latest_idx])
        current_histogram = float(histogram.iloc[latest_idx])
        prev_histogram = float(histogram.iloc[latest_idx-1]) if len(histogram) > 1 else current_histogram
        
        macd_trend = "BULLISH" if current_macd > current_signal else "BEARISH"
        momentum_direction = "STRENGTHENING" if current_histogram > prev_histogram else "WEAKENING"
       
        analysis.append(f"📈 MACD: {macd_trend} - Momentum {momentum_direction}")
        analysis.append(f"   MACD Line: {current_macd:.4f} | Signal: {current_signal:.4f} | Histogram: {current_histogram:.4f}")
        
        # Stochastic Analysis
        current_k = float(k_percent.iloc[latest_idx])
        current_d = float(d_percent.iloc[latest_idx])
        
        if current_k > 80 and current_d > 80:
            stoch_signal = "OVERBOUGHT - Potential reversal"
        elif current_k < 20 and current_d < 20:
            stoch_signal = "OVERSOLD - Potential bounce"
        elif current_k > current_d:
            stoch_signal = "BULLISH - %K above %D"
        else:
            stoch_signal = "BEARISH - %K below %D"
        
        analysis.append(f"📊 Stochastic: {stoch_signal}")
        analysis.append(f"   %K: {current_k:.1f} | %D: {current_d:.1f}")
        
        # Williams %R Analysis
        current_williams = float(williams_r.iloc[latest_idx])
        
        if current_williams > -20:
            williams_signal = "OVERBOUGHT - Potential reversal"
        elif current_williams < -80:
            williams_signal = "OVERSOLD - Potential bounce"
        else:
            williams_signal = "NEUTRAL range"
        
        analysis.append(f"📈 Williams %R: {williams_signal}")
        analysis.append(f"   Current: {current_williams:.1f}")
        
        # Overall momentum assessment
        bullish_signals = sum([
            current_macd > current_signal,
            current_k > current_d,
            current_histogram > prev_histogram,
            current_williams < -50  # Less oversold is more bullish for Williams %R
        ])
        
        if bullish_signals >= 3:
            analysis.append("🚀 **OVERALL MOMENTUM: STRONG BULLISH**")
        elif bullish_signals == 2:
            analysis.append("📈 **OVERALL MOMENTUM: BULLISH**")
        elif bullish_signals == 1:
            analysis.append("📉 **OVERALL MOMENTUM: BEARISH**")
        else:
            analysis.append("🔻 **OVERALL MOMENTUM: STRONG BEARISH**")
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error in momentum analysis for {ticker}: {str(e)}")
        return f"❌ Error in momentum analysis for {ticker}: {str(e)}"
    
def check_price_action_patterns(ticker):
    """Analyze price action patterns and candlestick formations"""
    try:
        data = get_cached_data(ticker, period="1mo", interval="1d")
        
        if data.empty or len(data) < 10:
            return f"❌ {ticker}: Insufficient data for price action analysis"
        
        analysis = [f"📊 **{ticker} Price Action Analysis**"]
        
        # Calculate recent price movements
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        current_price = float(latest['Close'])
        daily_change = (current_price - float(prev['Close'])) / float(prev['Close']) * 100
        
        # Weekly and monthly performance
        week_ago_price = float(data.iloc[-5]['Close']) if len(data) >= 5 else float(data.iloc[0]['Close'])
        weekly_change = (current_price - week_ago_price) / week_ago_price * 100
        
        month_ago_price = float(data.iloc[0]['Close'])
        monthly_change = (current_price - month_ago_price) / month_ago_price * 100
        
        analysis.append(f"💰 Current Price: ${current_price:.2f}")
        analysis.append(f"📈 Daily Change: {daily_change:+.2f}%")
        analysis.append(f"📊 Weekly Change: {weekly_change:+.2f}%")
        analysis.append(f"📅 Monthly Change: {monthly_change:+.2f}%")
        
        # Volatility analysis
        daily_returns = data['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        analysis.append(f"📊 Annualized Volatility: {volatility:.1f}%")
        
        # Simple candlestick patterns
        body_size = abs(latest['Close'] - latest['Open'])
        candle_range = latest['High'] - latest['Low']
        upper_shadow = latest['High'] - max(latest['Close'], latest['Open'])
        lower_shadow = min(latest['Close'], latest['Open']) - latest['Low']
        
        if body_size / candle_range < 0.3:
            if upper_shadow > lower_shadow * 2:
                analysis.append("🕯️ Bearish: Shooting Star pattern detected")
            elif lower_shadow > upper_shadow * 2:
                analysis.append("🕯️ Bullish: Hammer pattern detected")
            else:
                analysis.append("🕯️ Neutral: Doji pattern - Indecision")
        else:
            candle_type = "Bullish" if latest['Close'] > latest['Open'] else "Bearish"
            analysis.append(f"🕯️ {candle_type}: Strong body candle")
        
        # Support and resistance levels
        try:
            support_levels, resistance_levels = detect_support_resistance(data)
            
            if len(support_levels) > 0:
                nearest_support = float(support_levels.iloc[-1])
                support_distance = (current_price - nearest_support) / current_price * 100
                analysis.append(f"🛡️ Nearest Support: ${nearest_support:.2f} ({support_distance:+.1f}%)")
            
            if len(resistance_levels) > 0:
                nearest_resistance = float(resistance_levels.iloc[-1])
                resistance_distance = (nearest_resistance - current_price) / current_price * 100
                analysis.append(f"⚔️ Nearest Resistance: ${nearest_resistance:.2f} ({resistance_distance:+.1f}%)")
        except Exception as e:
            logger.error(f"Error in support/resistance analysis: {e}")
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error in price action analysis for {ticker}: {str(e)}")
        return f"❌ Error in price action analysis for {ticker}: {str(e)}"

def batch_analysis(tickers, analysis_type="comprehensive"):
    """Perform batch analysis on multiple tickers"""
    try:
        if isinstance(tickers, str):
            tickers = [t.strip().upper() for t in tickers.split(',')]
        
        results = {}
        
        def analyze_ticker(ticker):
            try:
                if analysis_type == "comprehensive":
                    return ticker, get_comprehensive_analysis(ticker)
                elif analysis_type == "sma":
                     return ticker, check_sma_crossover_fixed(ticker)
                elif analysis_type == "volume":
                    return ticker, check_volume_analysis(ticker)
                elif analysis_type == "momentum":
                    return ticker, check_momentum_indicators(ticker)
                elif analysis_type == "price_action":
                    return ticker, check_price_action_patterns(ticker)
                else:
                    return ticker, check_indicators(ticker)
            except Exception as e:
                return ticker, f"❌ Error analyzing {ticker}: {str(e)}"
        
        # Use ThreadPoolExecutor for concurrent analysis
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_results = [executor.submit(analyze_ticker, ticker) for ticker in tickers]
            
            for future in future_results:
                ticker, analysis = future.result()
                results[ticker] = analysis
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        return {ticker: f"❌ Error: {str(e)}" for ticker in tickers}

def get_market_summary():
    """Get a market summary for major indices or custom ticker list"""
    try:
        if tickers is None:
            tickers = ["SPY", "QQQ", "IWM", "VTI", "GLD", "TLT", "DXY"]
        
        analysis = ["📊 **MARKET SUMMARY**", "=" * 40]
        
        results = batch_analysis(tickers, "indicators")
        
        for ticker, result in results.items():
            if isinstance(result, dict):
                signal_strength = result.get('Signal_Strength', 0)
                close_price = result.get('Close', 'N/A')
                
                if signal_strength > 50:
                    signal_emoji = "🚀"
                elif signal_strength > 0:
                    signal_emoji = "📈"
                elif signal_strength > -50:
                    signal_emoji = "📉"
                else:
                    signal_emoji = "🔻"
                
                analysis.append(f"{signal_emoji} {ticker}: ${close_price} | Signal: {signal_strength:.0f}/100")
            else:
                analysis.append(f"❌ {ticker}: Error in analysis")
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error in market summary: {str(e)}")
        return f"❌ Error generating market summary: {str(e)}"
    
def get_alternative_market_analysis():
    """Alternative market indicators analysis"""
    try:
        alt_indicators = {
            'GLD': 'Gold',
            'UUP': 'US Dollar',
            'IEF': '10-Year Treasury',
            'BTC-USD': 'Bitcoin',
            'USO': 'Oil'
        }
        
        analysis = ["📈 **ALTERNATIVE MARKET ANALYSIS**", "=" * 35]
        
        for symbol, name in alt_indicators.items():
            try:
                data = get_cached_data(symbol, period="1mo", interval="1d")
                if not data.empty:
                    current = float(data['Close'].iloc[-1])
                    week_ago = float(data['Close'].iloc[-5]) if len(data) >= 5 else float(data['Close'].iloc[0])
                    change = ((current - week_ago) / week_ago) * 100
                    
                    status = "📈" if change > 2 else "📉" if change < -2 else "⚖️"
                    analysis.append(f"{status} {name}: {change:+.1f}% (1 week)")
                else:
                    analysis.append(f"❌ {name}: Data unavailable")
                    
            except Exception as e:
                logger.error(f"Error analyzing {name}: {str(e)}")
                analysis.append(f"❌ {name}: Data unavailable")
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error in alternative analysis: {str(e)}")
        return f"❌ Error in alternative analysis: {str(e)}"

def create_watchlist_alert(tickers, conditions=None):
    """Create alerts for a watchlist based on specified conditions"""
    try:
        if isinstance(tickers, str):
            tickers = [t.strip().upper() for t in tickers.split(',')]
        
        if conditions is None:
            conditions = {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'volume_spike': 2.0,
                'price_change': 5.0
            }
        
        alerts = []
        
        for ticker in tickers:
            try:
                indicators = check_indicators(ticker)
                if isinstance(indicators, dict):
                    # Check RSI conditions
                    rsi = indicators.get('RSI')
                    if rsi and rsi <= conditions['rsi_oversold']:
                        alerts.append(f"🔥 {ticker}: RSI Oversold ({rsi:.1f}) - Potential buy opportunity")
                    elif rsi and rsi >= conditions['rsi_overbought']:
                        alerts.append(f"⚠️ {ticker}: RSI Overbought ({rsi:.1f}) - Potential sell signal")
                    
                    # Check signal strength
                    signal_strength = indicators.get('Signal_Strength', 0)
                    if signal_strength > 70:
                        alerts.append(f"🚀 {ticker}: Strong bullish signal ({signal_strength:.0f}/100)")
                    elif signal_strength < -70:
                        alerts.append(f"🔻 {ticker}: Strong bearish signal ({signal_strength:.0f}/100)")
                
                # Check for volume spikes and price changes
                volume_analysis = check_volume_analysis(ticker)
                if "EXTREME HIGH volume" in volume_analysis:
                    alerts.append(f"📊 {ticker}: Extreme volume spike detected")
                
            except Exception as e:
                logger.error(f"Error checking alerts for {ticker}: {str(e)}")
                continue
        
        if alerts:
            result = ["🚨 **WATCHLIST ALERTS**", "=" * 30]
            result.extend(alerts)
            return "\n".join(result)
        else:
            return "✅ No alerts triggered for your watchlist"
        
    except Exception as e:
        logger.error(f"Error creating watchlist alerts: {str(e)}")
        return f"❌ Error creating alerts: {str(e)}"

def get_sector_rotation_analysis():
    """Analyze sector rotation using sector ETFs"""
    try:
        sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials', 
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLC': 'Communication Services',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLU': 'Utilities'
        }
        
        analysis = ["📊 **SECTOR ROTATION ANALYSIS**", "=" * 40]
        
        sector_performance = []
        
        for etf, sector_name in sector_etfs.items():
            try:
                data = get_cached_data(etf, period="1mo", interval="1d")
                if not data.empty:
                    current_price = float(data['Close'].iloc[-1])
                    month_ago_price = float(data['Close'].iloc[0])
                    monthly_return = (current_price - month_ago_price) / month_ago_price * 100
                    
                    # Get signal strength
                    indicators = check_indicators(etf)
                    signal_strength = 0
                    if isinstance(indicators, dict):
                        signal_strength = indicators.get('Signal_Strength', 0)
                    
                    sector_performance.append({
                        'etf': etf,
                        'sector': sector_name,
                        'return': monthly_return,
                        'signal': signal_strength
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing {etf}: {str(e)}")
                continue
        
        # Sort by monthly performance
        sector_performance.sort(key=lambda x: x['return'], reverse=True)
        
        analysis.append("🏆 **TOP PERFORMING SECTORS (1 Month)**")
        for i, sector in enumerate(sector_performance[:5]):
            signal_emoji = "🚀" if sector['signal'] > 0 else "📉"
            analysis.append(f"{i+1}. {sector['sector']} ({sector['etf']}): {sector['return']:+.2f}% {signal_emoji}")
        
        analysis.append("\n📉 **UNDERPERFORMING SECTORS (1 Month)**")
        for i, sector in enumerate(sector_performance[-3:]):
            signal_emoji = "🚀" if sector['signal'] > 0 else "📉"
            analysis.append(f"{len(sector_performance)-2+i}. {sector['sector']} ({sector['etf']}): {sector['return']:+.2f}% {signal_emoji}")
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error in sector rotation analysis: {str(e)}")
        return f"❌ Error in sector analysis: {str(e)}"

# Enhanced sector analysis with more robust error handling
def get_enhanced_sector_analysis():
    """Enhanced sector rotation analysis"""
    try:
        # Major sector ETFs
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV', 
            'Financials': 'XLF',
            'Energy': 'XLE',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Communications': 'XLC'
        }
        
        results = []
        results.append("🏭 ENHANCED SECTOR ROTATION ANALYSIS")
        results.append("=" * 50)
        
        sector_performance = {}
        
        for sector, etf in sector_etfs.items():
            try:
                data = get_stock_data(etf, period="3mo")
                if data.empty:
                    continue
                    
                # Calculate performance metrics
                current_price = data['Close'].iloc[-1]
                price_30d = data['Close'].iloc[-20] if len(data) >= 20 else data['Close'].iloc[0]
                price_5d = data['Close'].iloc[-5] if len(data) >= 5 else data['Close'].iloc[0]
                
                perf_30d = ((current_price - price_30d) / price_30d) * 100
                perf_5d = ((current_price - price_5d) / price_5d) * 100
                
                # Calculate RSI using your existing check_indicators function
                indicators = check_indicators(etf)
                rsi = indicators.get('RSI') if isinstance(indicators, dict) else None
                
                # Volume analysis
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                sector_performance[sector] = {
                    'perf_30d': perf_30d,
                    'perf_5d': perf_5d,
                    'rsi': rsi,
                    'volume_ratio': volume_ratio,
                    'etf': etf
                }
                
            except Exception as e:
                continue
        
        if not sector_performance:
            return "❌ Unable to fetch sector data. Please try again later."

       
    except Exception as e:
        logger.error(f"Error in alternative analysis: {str(e)}")
        return f"❌ Error in alternative analysis: {str(e)}"
    

    # Sort by 30-day performance    
    sorted_sectors = sorted(sector_performance.items(), 
                          key=lambda x: x[1]['perf_30d'], reverse=True) 
    
    results.append("\n🏆 TOP PERFORMING SECTORS (30-day):")
    results.append("-" * 35)
    
    for i, (sector, data) in enumerate(sorted_sectors[:5]):
        status = "🔥" if data['perf_30d'] > 5 else "📈" if data['perf_30d'] > 0 else "📉"
        rsi_status = "⚠️" if data['rsi'] and data['rsi'] > 70 else "🔥" if data['rsi'] and data['rsi'] < 30 else "⚖️"
        vol_status = "🚀" if data['volume_ratio'] > 1.5 else ""
        
        results.append(f"{i+1}. {status} {sector}")
        results.append(f"   ETF: {data['etf']} | 30d: {data['perf_30d']:+.1f}% | 5d: {data['perf_5d']:+.1f}%")
        if data['rsi']:
            results.append(f"   RSI: {data['rsi']:.1f} {rsi_status} | Volume: {data['volume_ratio']:.1f}x {vol_status}")
        results.append("")
    
    results.append("\n📉 UNDERPERFORMING SECTORS:")
    results.append("-" * 30)
    
    for i, (sector, data) in enumerate(sorted_sectors[-3:]):
        results.append(f"• {sector}: {data['perf_30d']:+.1f}% (30d)")
    
    # Market breadth analysis
    positive_sectors = sum(1 for _, data in sector_performance.items() if data['perf_30d'] > 0)
    total_sectors = len(sector_performance)
    breadth = (positive_sectors / total_sectors) * 100
    
    results.append(f"\n📊 MARKET BREADTH:")
    results.append(f"   Positive Sectors: {positive_sectors}/{total_sectors} ({breadth:.0f}%)")
    
    if breadth > 70:
        results.append("   🟢 BROAD MARKET STRENGTH")
    elif breadth > 50:
        results.append("   🟡 MIXED MARKET CONDITIONS")
    else:
        results.append("   🔴 MARKET WEAKNESS")

    # Rotation signals
    results.append(f"\n🔄 ROTATION SIGNALS:")
    
    # Find sectors with strong 5d vs 30d performance divergence
    momentum_shifts = []
    for sector, data in sector_performance.items():
        if data['perf_5d'] > data['perf_30d'] + 2:  # Recent acceleration
            momentum_shifts.append(f"📈 {sector}: Recent acceleration")
        elif data['perf_5d'] < data['perf_30d'] - 2:  # Recent deceleration
            momentum_shifts.append(f"📉 {sector}: Losing momentum")
    
    if momentum_shifts:
        for signal in momentum_shifts[:5]:  # Top 5 signals
            results.append(f"   {signal}")
    else:
        results.append("   ⚖️ No significant rotation signals detected")
    
    return "\n".join(results)

#except Exception as e:
#logger.error(f"Error in sector analysis: {str(e)}")
#return f"❌ Error in sector analysis: {str(e)}"

# Main execution functions
def enhanced_main():
    """Enhanced main function with more robust analysis options"""
    print("📊 Enhanced Technical Analysis Tool")
    print("=" * 50)
    print("🚀 Now with improved sector analysis and error handling!")
    
    while True:
        print("\n📋 Available Commands:")
        print("-" * 30)
        print("📈 INDIVIDUAL STOCK ANALYSIS:")
        print("  comp <ticker>     - Comprehensive analysis")
        print("  sma <ticker>      - SMA crossover analysis") 
        print("  vol <ticker>      - Volume analysis")
        print("  mom <ticker>      - Momentum indicators")
        print("  price <ticker>    - Price action patterns")
        print("  rsi <ticker>      - RSI analysis only")
        
        print("\n📊 MARKET & SECTOR ANALYSIS:")
        print("  sectors           - Enhanced sector rotation")
        print("  market            - Major indices summary")
        print("  alt-market        - Alternative market analysis")
        print("  crypto           - Cryptocurrency analysis")
        
        print("\n🔍 BATCH & SCREENING:")
        print("  batch <tickers>   - Batch analysis (comma-separated)")
        print("  screen-rsi        - Screen for RSI extremes")
        print("  screen-vol        - Screen for volume spikes")
        print("  alerts <tickers>  - Watchlist alerts")
        
        print("\n⚙️ CONFIGURATION & UTILITIES:")
        print("  config            - View/modify settings")
        print("  test <ticker>     - Test data availability")
        print("  help             - Show detailed help")
        print("  quit/exit        - Exit program")
        
        command = input("\n🎯 Enter command: ").strip().lower()
        
        try:
            if command.startswith('quit') or command == 'exit':
                print("👋 Thanks for using the Technical Analysis Tool!")
                break
                
            elif command.startswith('comp '):
                ticker = command.split(' ', 1)[1].upper()
                print(f"\n🔍 Analyzing {ticker}...")
                print(get_comprehensive_analysis(ticker))
                
            elif command.startswith('sma '):
                ticker = command.split(' ', 1)[1].upper()
                print(f"\n📈 SMA Analysis for {ticker}...")
                print(check_sma_crossover_fixed(ticker))
                
            elif command.startswith('vol '):
                ticker = command.split(' ', 1)[1].upper()
                print(f"\n📊 Volume Analysis for {ticker}...")
                print(check_volume_analysis(ticker))
                
            elif command.startswith('mom '):
                ticker = command.split(' ', 1)[1].upper()
                print(f"\n🚀 Momentum Analysis for {ticker}...")
                print(check_momentum_indicators(ticker))
                
            elif command.startswith('price '):
                ticker = command.split(' ', 1)[1].upper()
                print(f"\n🕯️ Price Action Analysis for {ticker}...")
                print(check_price_action_patterns(ticker))
                
            elif command.startswith('rsi '):
                ticker = command.split(' ', 1)[1].upper()
                print(f"\n📈 RSI Analysis for {ticker}...")
                indicators = check_indicators(ticker)
                if isinstance(indicators, dict) and indicators.get('RSI'):
                    rsi = indicators['RSI']
                    if rsi > 70:
                        print(f"📈 {ticker} RSI: {rsi:.1f} - OVERBOUGHT ⚠️")
                    elif rsi < 30:
                        print(f"📉 {ticker} RSI: {rsi:.1f} - OVERSOLD 🔥")
                    else:
                        print(f"⚖️ {ticker} RSI: {rsi:.1f} - NEUTRAL")
                else:
                    print(f"❌ Could not calculate RSI for {ticker}")
                    
            elif command == 'sectors':
                print("\n🏭 Running Enhanced Sector Analysis...")
                print("⏳ This may take a moment...")
                print(get_enhanced_sector_analysis())
                
            elif command == 'market':
                print("\n📊 Market Summary...")
                print(get_market_summary())
                
            elif command == 'alt-market':
                print("\n📈 Alternative Market Analysis...")
                print(get_alternative_market_analysis())
                
            elif command == 'crypto':
                print("\n₿ Cryptocurrency Analysis...")
                print(get_crypto_analysis())
                
            elif command.startswith('batch '):
                tickers = command.split(' ', 1)[1]
                print(f"\n📊 Batch Analysis for: {tickers}")
                print("⏳ Processing...")
                results = batch_analysis(tickers, "comprehensive")
                for ticker, result in results.items():
                    print(f"\n{'-'*50}")
                    print(result)
                    
            elif command == 'screen-rsi':
                print("\n🔍 Screening for RSI Extremes...")
                # Screen popular stocks for RSI extremes
                popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA']
                results = batch_analysis(popular_stocks, "comprehensive")
                
                oversold = []
                overbought = []
                
                for ticker, result in results.items():
                    if isinstance(result, str) and "RSI:" in result:
                        if "OVERSOLD" in result:
                            oversold.append(ticker)
                        elif "OVERBOUGHT" in result:
                            overbought.append(ticker)
                
                if oversold:
                    print(f"🔥 OVERSOLD (RSI < 30): {', '.join(oversold)}")
                if overbought:
                    print(f"⚠️ OVERBOUGHT (RSI > 70): {', '.join(overbought)}")
                if not oversold and not overbought:
                    print("⚖️ No extreme RSI levels found in screened stocks")
                    
            elif command == 'screen-vol':
                print("\n📊 Screening for Volume Spikes...")
                popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA']
                
                volume_spikes = []
                for ticker in popular_stocks:
                    try:
                        vol_analysis = check_volume_analysis(ticker)
                        if "High volume" in vol_analysis or "EXTREME HIGH volume" in vol_analysis:
                            volume_spikes.append(ticker)
                    except:
                        continue
                
                if volume_spikes:
                    print(f"🚀 VOLUME SPIKES DETECTED: {', '.join(volume_spikes)}")
                else:
                    print("😴 No significant volume spikes detected")
                    
            elif command.startswith('alerts '):
                tickers = command.split(' ', 1)[1]
                print(f"\n🚨 Checking alerts for: {tickers}")
                print(create_watchlist_alert(tickers))
                
            elif command.startswith('test '):
                ticker = command.split(' ', 1)[1].upper()
                print(f"\n🧪 Testing data availability for {ticker}...")
                
                try:
                    data = get_stock_data(ticker, period="1mo")
                    if not data.empty:
                        print(f"✅ {ticker}: {len(data)} days of data available")
                        print(f"   Latest close: ${float(data['Close'].iloc[-1]):.2f}")
                        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
                    else:
                        print(f"❌ {ticker}: No data available")
                        
                    # Test indicators
                    indicators = check_indicators(ticker)
                    if isinstance(indicators, dict):
                        print(f"✅ Technical indicators calculated successfully")
                    else:
                        print(f"⚠️ Issue with technical indicators: {indicators}")
                        
                except Exception as e:
                    print(f"❌ Error testing {ticker}: {str(e)}")
                    
            elif command == 'config':
                print(f"\n⚙️ Current Configuration:")
                print(f"   SMA Short Period: {config.sma_short}")
                print(f"   SMA Long Period: {config.sma_long}")
                print(f"   RSI Period: {config.rsi_period}")
                print(f"   RSI Overbought Level: {config.rsi_overbought}")
                print(f"   RSI Oversold Level: {config.rsi_oversold}")
                print(f"   Volume Spike Threshold: {config.volume_threshold}x")
                print(f"   Data Period: {config.data_period}")
                print(f"   Cache Duration: {config.cache_minutes} minutes")
                
                modify = input("\n🔧 Modify settings? (y/n): ").lower()
                if modify == 'y':
                    try:
                        new_rsi_period = input(f"RSI Period (current: {config.rsi_period}): ")
                        if new_rsi_period.strip():
                            config.rsi_period = int(new_rsi_period)
                            
                        new_vol_threshold = input(f"Volume Threshold (current: {config.volume_threshold}): ")
                        if new_vol_threshold.strip():
                            config.volume_threshold = float(new_vol_threshold)
                            
                        print("✅ Configuration updated!")
                    except ValueError:
                        print("❌ Invalid input. Configuration unchanged.")
                        
            elif command == 'help':
                print("\n📚 DETAILED HELP")
                print("=" * 40)
                print("📈 ANALYSIS TYPES:")
                print("  • Comprehensive: All indicators + signal strength")
                print("  • SMA: Moving average crossovers (Golden/Death Cross)")
                print("  • Volume: Trading volume analysis with OBV")
                print("  • Momentum: MACD, Stochastic, Williams %R")
                print("  • Price Action: Candlestick patterns, S/R levels")
                print("\n📊 SCREENING:")
                print("  • RSI screening finds oversold/overbought stocks")
                print("  • Volume screening finds unusual trading activity")
                print("\n🎯 TIPS:")
                print("  • Use 'test <ticker>' if having data issues")
                print("  • Sector analysis works best during market hours")
                print("  • Batch analysis: separate tickers with commas")
                print("  • Configuration changes apply immediately")
                
            else:
                print("❌ Unknown command. Type 'help' for detailed instructions.")
                
        except KeyboardInterrupt:
            print("\n\n⏸️ Operation cancelled by user.")
            continue
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("💡 Tip: Try 'test <ticker>' to check data availability")
            continue


def quick_test():
    """Quick test function to verify the system is working"""
    print("🧪 Running Quick System Test...")
    print("-" * 30)
    
    test_ticker = "AAPL"
    
    try:
        # Test data fetching
        print(f"1. Testing data fetch for {test_ticker}...")
        data = get_stock_data(test_ticker, period="1mo")
        if not data.empty:
            print(f"   ✅ Success: {len(data)} days of data")
        else:
            print("   ❌ No data received")
            return False
            
        # Test indicators
        print("2. Testing technical indicators...")
        indicators = check_indicators(test_ticker)
        if isinstance(indicators, dict):
            print("   ✅ Indicators calculated successfully")
        else:
            print(f"   ❌ Indicator error: {indicators}")
            return False
            
        # Test comprehensive analysis
        print("3. Testing comprehensive analysis...")
        analysis = get_comprehensive_analysis(test_ticker)
        if "Error" not in analysis:
            print("   ✅ Comprehensive analysis working")
        else:
            print("   ❌ Analysis error")
            return False
            
        print("\n🎉 All tests passed! System is ready.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


# Alternative entry point that runs the test first
def safe_main():
    """Safe main function that tests the system first"""
    print("🚀 Technical Analysis Tool - Enhanced Edition")
    print("=" * 50)
    
    # Run quick test
    if not quick_test():
        print("\n⚠️ System test failed. Some features may not work properly.")
        proceed = input("Continue anyway? (y/n): ").lower()
        if proceed != 'y':
            print("👋 Exiting. Please check your system setup and try again.")
            return
        else:
            print("⚠️ Proceeding with potential issues...")
    
    # If test passed or user chose to continue, run the main application
    print("\n🎯 Starting main application...")
    try:
        enhanced_main()
    except KeyboardInterrupt:
        print("\n\n👋 Application terminated by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Critical error in main application: {str(e)}")
        print("💡 Please restart the application or check your setup.")


# Entry point
if __name__ == "__main__":
    safe_main()
