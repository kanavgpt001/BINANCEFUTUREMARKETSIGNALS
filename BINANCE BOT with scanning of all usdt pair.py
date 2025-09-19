# ==============================================================================
# Enhanced Scalping Bot with Leading Indicators for High Volatility
# ==============================================================================
import pandas as pd
from binance.client import Client
from binance import BinanceSocketManager
import ta
import numpy as np
import time
import os
import warnings
import logging
from threading import Thread, Lock
from datetime import datetime, timedelta
from ta.utils import dropna
from scipy.signal import argrelextrema
import tkinter as tk
from tkinter import scrolledtext
import sys
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress SSL warnings and pandas warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option('mode.chained_assignment', None)

# --- Configuration ---
API_KEY = "YOUR BINANCE API KEY"
API_SECRET = "YOUR BINANCE API SECRET"

# Analysis Parameters
TIME_FRAMES = ['1m', '3m', '5m', '15m']
HISTORY_KLINE_LIMIT = 100
SCAN_LIMIT = 50  # Limit number of pairs to scan for performance

# Leading Indicator Parameters
RSI_PERIOD = 7  # Shorter period for faster signals
STOCH_OVERBOUGHT = 80
STOCH_OVERSOLD = 20
WILLIAMS_PERIOD = 10  # Shorter for faster signals
CCI_PERIOD = 14  # Shorter for faster reaction
CCI_OVERBOUGHT = 100
CCI_OVERSOLD = -100

# Volatility-based Parameters
ATR_PERIOD = 7  # Shorter for faster volatility measurement
BOLLINGER_PERIOD = 14  # Shorter period
BOLLINGER_STD = 1.8  # Tighter bands for scalping
KELTNER_PERIOD = 14
KELTNER_ATR_MULTIPLIER = 1.5

# Momentum Parameters
MOMENTUM_PERIOD = 5  # Very short for scalping
ROC_PERIOD = 5  # Rate of Change
ULTIMATE_OSCILLATOR_PERIODS = [7, 14, 28]

# Scalping Parameters
MIN_CONFIDENCE = 75  # Minimum confidence percentage for signals
MAX_SPREAD_PCT = 0.1  # Maximum spread percentage for entry
MIN_RISK_REWARD = 1.5  # Minimum risk/reward ratio
STOP_LOSS_ATR_MULTIPLIER = 1.2  # Tighter stop loss for scalping
TAKE_PROFIT_ATR_MULTIPLIER = 2.0  # Adjusted take profit

# Support/Resistance Parameters
SR_LOOKBACK = 8  # Shorter lookback for scalping
SR_MIN_DISTANCE = 3  # Closer levels for scalping

# Global variable to store last signals
last_signals = []
gui_instance = None
print_lock = Lock()
best_pairs = []
sure_shot_signals = []

# --- Initialize Binance REST Client ---
try:
    client = Client(API_KEY, API_SECRET, {"verify": False, "timeout": 30})
    client.ping()
    logger.info("Successfully connected to Binance REST API.")
except Exception as e:
    logger.error(f"Error connecting to Binance REST API: {e}")
    exit()

# GUI Setup
class TradingBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Crypto Scalping Bot - Leading Indicators")
        self.root.configure(bg='black')
        self.root.geometry("1200x800")
        
        # Create text area with scrollbar
        self.text_area = scrolledtext.ScrolledText(
            root, 
            wrap=tk.WORD, 
            width=140, 
            height=40,
            bg='black',
            fg='white',
            font=('Courier New', 10)
        )
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.text_area.configure(state="disabled")
        
        # Create a lock for thread-safe GUI updates
        self.gui_lock = Lock()
        
    def update_display(self, text):
        with self.gui_lock:
            self.text_area.configure(state="normal")
            self.text_area.insert(tk.END, text + "\n")
            self.text_area.see(tk.END)
            self.text_area.configure(state="disabled")
            self.text_area.update()

class ThreadSafePrint:
    def __init__(self, gui):
        self.gui = gui
        
    def write(self, text):
        with print_lock:
            if self.gui:
                self.gui.update_display(text.strip())
            else:
                sys.__stdout__.write(text)
                
    def flush(self):
        pass

def get_latest_historical_data(symbol, interval):
    """Fetches the latest historical data for analysis."""
    try:
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=HISTORY_KLINE_LIMIT
        )
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = dropna(df)
        logger.info(f"Fetched {len(df)} latest candles for {symbol} on {interval}")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol} on {interval}: {e}")
        return pd.DataFrame()

def calculate_pivot_points(df, period='daily'):
    """Calculate pivot points for scalping timeframe."""
    try:
        if len(df) < 2:
            return {}
            
        # Use shorter timeframe for scalping
        if period == 'scalping':
            pivot_df = df.resample('15T').agg({
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
            
            if len(pivot_df) < 2:
                return {}
                
            previous = pivot_df.iloc[-2]
        else:  # daily
            pivot_df = df.resample('D').agg({
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
            
            if len(pivot_df) < 2:
                return {}
                
            previous = pivot_df.iloc[-2]
        
        h, l, c = previous['high'], previous['low'], previous['close']
        
        # Calculate pivot points
        pp = (h + l + c) / 3
        r1 = (2 * pp) - l
        s1 = (2 * pp) - h
        
        return {
            'pivot_point': pp,
            'support_1': s1,
            'resistance_1': r1
        }
    except Exception as e:
        logger.error(f"Error calculating pivot points: {e}")
        return {}

def calculate_support_resistance(df, num_levels=5):
    """Calculates support and resistance levels for scalping."""
    try:
        # Use recent price action for scalping
        recent_df = df.tail(20)
        highs = recent_df['high'].values
        lows = recent_df['low'].values
        
        # Find local maxima and minima
        resistance_indices = argrelextrema(highs, np.greater, order=SR_LOOKBACK)[0]
        resistance_levels = highs[resistance_indices]
        
        support_indices = argrelextrema(lows, np.less, order=SR_LOOKBACK)[0]
        support_levels = lows[support_indices]
        
        current_price = df['close'].iloc[-1]
        
        # Get nearest levels
        resistance_above = sorted([r for r in resistance_levels if r > current_price])[:num_levels]
        support_below = sorted([s for s in support_levels if s < current_price], reverse=True)[:num_levels]
        
        return {
            'support_below': support_below,
            'resistance_above': resistance_above,
            'current_price': current_price
        }
        
    except Exception as e:
        logger.error(f"Error calculating support/resistance: {e}")
        return {
            'support_below': [],
            'resistance_above': [],
            'current_price': df['close'].iloc[-1] if not df.empty else 0
        }

def calculate_leading_indicators(df):
    """Calculates leading indicators for scalping in volatile markets."""
    if df.empty or len(df) < 20:
        logger.warning("Not enough data to calculate leading indicators")
        return df

    # Remove lagging indicators (EMAs, MACD)
    # Add leading indicators instead
    
    # Momentum Indicators (Leading)
    df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_PERIOD)
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=CCI_PERIOD)
    df['williams'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=WILLIAMS_PERIOD)
    
    # Rate of Change (Leading)
    df['roc'] = ta.momentum.roc(df['close'], window=ROC_PERIOD)
    
    # Ultimate Oscillator (Leading)
    df['uo'] = ta.momentum.ultimate_oscillator(
        df['high'], df['low'], df['close'], 
        window1=ULTIMATE_OSCILLATOR_PERIODS[0],
        window2=ULTIMATE_OSCILLATOR_PERIODS[1],
        window3=ULTIMATE_OSCILLATOR_PERIODS[2]
    )
    
    # Awesome Oscillator (Leading)
    df['ao'] = ta.momentum.awesome_oscillator(df['high'], df['low'], window1=5, window2=34)
    
    # Vortex Indicator (Leading for trend changes)
    df['vortex_pos'] = ta.trend.vortex_indicator_pos(df['high'], df['low'], df['close'], window=14)
    df['vortex_neg'] = ta.trend.vortex_indicator_neg(df['high'], df['low'], df['close'], window=14)
    
    # Volatility Indicators (Leading for breakouts)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=ATR_PERIOD)
    
    # Bollinger Bands with tighter settings
    bollinger = ta.volatility.BollingerBands(df['close'], window=BOLLINGER_PERIOD, window_dev=BOLLINGER_STD)
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_lower'] = bollinger.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Keltner Channels (Volatility-based)
    keltner = ta.volatility.KeltnerChannel(
        df['high'], df['low'], df['close'], 
        window=KELTNER_PERIOD, 
        window_atr=KELTNER_PERIOD,
        multiplier=KELTNER_ATR_MULTIPLIER
    )
    df['kc_upper'] = keltner.keltner_channel_hband()
    df['kc_middle'] = keltner.keltner_channel_mband()
    df['kc_lower'] = keltner.keltner_channel_lband()
    
    # Donchian Channel (Breakout indicator)
    df['donchian_upper'] = df['high'].rolling(window=20).max()
    df['donchian_lower'] = df['low'].rolling(window=20).min()
    
    # Volume-based Leading Indicators
    df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], window=20)
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    # Volume Price Trend (Leading)
    df['vpt'] = ta.volume.volume_price_trend(df['close'], df['volume'])
    
    # Force Index (Leading momentum)
    df['force_index'] = ta.volume.force_index(df['close'], df['volume'], window=13)
    
    # Price Rate of Change (Leading)
    df['momentum'] = ta.momentum.roc(df['close'], window=MOMENTUM_PERIOD)
    
    # Price position relative to volatility bands
    df['price_vs_bb'] = (df['close'] - df['bb_middle']) / df['bb_middle'] * 100
    df['price_vs_kc'] = (df['close'] - df['kc_middle']) / df['kc_middle'] * 100
    df['squeeze'] = (df['bb_upper'] - df['bb_lower']) / (df['kc_upper'] - df['kc_lower'])
    
    return df

def analyze_pair(symbol, df, timeframe='5m'):
    """Analyzes a single pair using leading indicators for scalping."""
    if df.empty or len(df) < 20:
        return None

    df = calculate_leading_indicators(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    current_price = df['close'].iloc[-1]
    
    # Calculate support/resistance and pivot points
    sr_levels = calculate_support_resistance(df, num_levels=3)
    scalping_pivots = calculate_pivot_points(df, 'scalping')
    
    # Price position relative to key levels
    near_support = any(abs(current_price - level) / current_price < 0.003 for level in sr_levels['support_below'])
    near_resistance = any(abs(current_price - level) / current_price < 0.003 for level in sr_levels['resistance_above'])
    
    # Check if price is near pivot levels
    near_pivot = False
    if scalping_pivots:
        pivot_levels = [scalping_pivots['pivot_point'], scalping_pivots['support_1'], scalping_pivots['resistance_1']]
        near_pivot = any(abs(current_price - level) / current_price < 0.005 for level in pivot_levels)

    # Long Signal Conditions (Leading indicators only)
    long_conditions = [
        # Momentum conditions (Leading)
        latest['rsi'] > 45 and latest['rsi'] < 70,
        latest['stoch_k'] > latest['stoch_d'] and latest['stoch_k'] < 75,
        latest['cci'] > -50 and latest['cci'] < 100,
        latest['williams'] > -80 and latest['williams'] < -20,
        
        # Oscillator conditions
        latest['uo'] > 50,
        latest['ao'] > 0,
        
        # Vortex trend
        latest['vortex_pos'] > latest['vortex_neg'],
        
        # Volume confirmation
        latest['force_index'] > 0,
        df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1],
        
        # Volatility breakout conditions
        latest['close'] > latest['bb_middle'],
        latest['close'] > latest['kc_middle'],
        latest['squeeze'] < 1.0,  # Not in squeeze
        
        # Price action
        near_support or near_pivot,
        df['close'].iloc[-1] > df['open'].iloc[-1],
        df['close'].iloc[-1] > df['close'].iloc[-2],
    ]

    # Short Signal Conditions (Leading indicators only)
    short_conditions = [
        # Momentum conditions (Leading)
        latest['rsi'] < 55 and latest['rsi'] > 30,
        latest['stoch_k'] < latest['stoch_d'] and latest['stoch_k'] > 25,
        latest['cci'] < 50 and latest['cci'] > -100,
        latest['williams'] < -20 and latest['williams'] > -80,
        
        # Oscillator conditions
        latest['uo'] < 50,
        latest['ao'] < 0,
        
        # Vortex trend
        latest['vortex_neg'] > latest['vortex_pos'],
        
        # Volume confirmation
        latest['force_index'] < 0,
        df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1],
        
        # Volatility breakdown conditions
        latest['close'] < latest['bb_middle'],
        latest['close'] < latest['kc_middle'],
        latest['squeeze'] < 1.0,  # Not in squeeze
        
        # Price action
        near_resistance or near_pivot,
        df['close'].iloc[-1] < df['open'].iloc[-1],
        df['close'].iloc[-1] < df['close'].iloc[-2],
    ]

    long_confirmations = sum(long_conditions)
    short_confirmations = sum(short_conditions)
    total_conditions = len(long_conditions)
    
    long_confidence = (long_confirmations / total_conditions) * 100
    short_confidence = (short_confirmations / total_conditions) * 100

    # Calculate entry and exit points based on ATR
    atr = latest['atr']
    entry_price = current_price
    
    if long_confidence >= MIN_CONFIDENCE:
        stop_loss = entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER)
        take_profit = entry_price + (atr * TAKE_PROFIT_ATR_MULTIPLIER)
        signal = f"LONG ({long_confidence:.0f}% confidence)"
        direction = 'LONG'
        
    elif short_confidence >= MIN_CONFIDENCE:
        stop_loss = entry_price + (atr * STOP_LOSS_ATR_MULTIPLIER)
        take_profit = entry_price - (atr * TAKE_PROFIT_ATR_MULTIPLIER)
        signal = f"SHORT ({short_confidence:.0f}% confidence)"
        direction = 'SHORT'
    else:
        signal = "HOLD"
        direction = 'HOLD'
        entry_price = 0
        stop_loss = 0
        take_profit = 0

    # Calculate risk-reward ratio
    risk_reward = 0
    if entry_price > 0 and stop_loss > 0 and take_profit > 0:
        if direction == 'LONG':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
            
        if risk > 0:
            risk_reward = reward / risk

    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'signal': signal,
        'direction': direction,
        'long_confirmations': long_confirmations,
        'short_confirmations': short_confirmations,
        'total_conditions': total_conditions,
        'long_confidence': long_confidence,
        'short_confidence': short_confidence,
        'current_price': current_price,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward': risk_reward,
        'atr': atr,
        'sr_levels': sr_levels,
        'scalping_pivots': scalping_pivots,
        'volume': df['volume'].iloc[-1]
    }

def scan_all_pairs():
    """Scans all available trading pairs using leading indicators."""
    try:
        # Get all available trading pairs
        exchange_info = client.futures_exchange_info()
        all_symbols = [symbol['symbol'] for symbol in exchange_info['symbols'] 
                      if symbol['symbol'].endswith('USDT') and symbol['status'] == 'TRADING']
        
        # Filter out low volume pairs and limit the number of pairs to scan
        all_symbols = all_symbols[:SCAN_LIMIT]
        
        logger.info(f"Scanning {len(all_symbols)} trading pairs with leading indicators...")
        
        results = []
        
        for i, symbol in enumerate(all_symbols):
            try:
                if i % 10 == 0:
                    logger.info(f"Analyzing {symbol} with leading indicators ({i+1}/{len(all_symbols)})")
                
                df = get_latest_historical_data(symbol, '5m')
                if not df.empty:
                    result = analyze_pair(symbol, df.copy(), '5m')
                    if result and result['signal'] != "HOLD" and result['risk_reward'] >= MIN_RISK_REWARD:
                        confidence = result['long_confidence'] if 'LONG' in result['signal'] else result['short_confidence']
                        
                        results.append({
                            'symbol': symbol,
                            'signal': result['signal'],
                            'confidence': confidence,
                            'direction': result['direction'],
                            'entry_price': result['entry_price'],
                            'stop_loss': result['stop_loss'],
                            'take_profit': result['take_profit'],
                            'risk_reward': result['risk_reward'],
                            'atr': result['atr'],
                            'current_price': result['current_price'],
                            'volume': result['volume']
                        })
                        
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by confidence and volume (high volume pairs preferred for scalping)
        results.sort(key=lambda x: (x['confidence'], x['volume']), reverse=True)
        
        return results[:15], results  # Return top 15 pairs
        
    except Exception as e:
        logger.error(f"Error scanning all pairs: {e}")
        return [], []

def analyze_top_pairs_multitimeframe(top_pairs):
    """Performs multi-timeframe analysis using leading indicators."""
    if not top_pairs:
        return []
    
    detailed_results = []
    
    for pair in top_pairs:
        symbol = pair['symbol']
        timeframe_results = {}
        
        # Analyze across all timeframes
        for timeframe in TIME_FRAMES:
            try:
                df = get_latest_historical_data(symbol, timeframe)
                if not df.empty:
                    result = analyze_pair(symbol, df.copy(), timeframe)
                    if result:
                        timeframe_results[timeframe] = result
            except Exception as e:
                logger.error(f"Error analyzing {symbol} on {timeframe}: {e}")
                continue
        
        if not timeframe_results:
            continue
            
        # Calculate consensus across timeframes
        total_long = sum(1 for res in timeframe_results.values() if res['direction'] == 'LONG')
        total_short = sum(1 for res in timeframe_results.values() if res['direction'] == 'SHORT')
        total_timeframes = len(timeframe_results)
        
        # Calculate weighted confidence (shorter timeframes have more weight)
        timeframe_weights = {'1m': 1.5, '3m': 1.2, '5m': 1.0, '15m': 0.8}
        weighted_long = sum(timeframe_weights.get(tf, 1.0) * res['long_confidence'] 
                          for tf, res in timeframe_results.items() if res['direction'] == 'LONG')
        weighted_short = sum(timeframe_weights.get(tf, 1.0) * res['short_confidence'] 
                           for tf, res in timeframe_results.items() if res['direction'] == 'SHORT')
        
        total_weight_long = sum(timeframe_weights.get(tf, 1.0) for tf in timeframe_results 
                              if timeframe_results[tf]['direction'] == 'LONG')
        total_weight_short = sum(timeframe_weights.get(tf, 1.0) for tf in timeframe_results 
                               if timeframe_results[tf]['direction'] == 'SHORT')
        
        avg_long_confidence = weighted_long / max(1, total_weight_long)
        avg_short_confidence = weighted_short / max(1, total_weight_short)
        
        # Determine overall signal
        if total_long >= total_short and avg_long_confidence >= MIN_CONFIDENCE:
            overall_signal = f"LONG ({total_long}/{total_timeframes} timeframes)"
            signal_strength = avg_long_confidence
            direction = 'LONG'
        elif total_short > total_long and avg_short_confidence >= MIN_CONFIDENCE:
            overall_signal = f"SHORT ({total_short}/{total_timeframes} timeframes)"
            signal_strength = avg_short_confidence
            direction = 'SHORT'
        else:
            continue
        
        # Use 5m timeframe for entry/exit levels
        latest_result = timeframe_results['5m'] if '5m' in timeframe_results else list(timeframe_results.values())[-1]
        
        detailed_results.append({
            'symbol': symbol,
            'overall_signal': overall_signal,
            'signal_strength': signal_strength,
            'timeframe_results': timeframe_results,
            'total_long_timeframes': total_long,
            'total_short_timeframes': total_short,
            'total_timeframes': total_timeframes,
            'entry_price': latest_result['entry_price'],
            'stop_loss': latest_result['stop_loss'],
            'take_profit': latest_result['take_profit'],
            'risk_reward': latest_result['risk_reward'],
            'atr': latest_result['atr'],
            'current_price': latest_result['current_price'],
            'direction': direction,
            'volume': latest_result.get('volume', 0)
        })
    
    return detailed_results

def display_sure_shot_signals(signals):
    """Displays sure-shot signals with detailed leading indicator analysis."""
    if not signals:
        return "No sure-shot signals found."
    
    output_buffer = io.StringIO()
    
    print("\n" + "="*120, file=output_buffer)
    print("🎯 LEADING INDICATOR SCALPING SIGNALS", file=output_buffer)
    print("="*120, file=output_buffer)
    
    for i, signal in enumerate(signals, 1):
        symbol = signal['symbol']
        
        print(f"\n{i}. {symbol} - {signal['overall_signal']} (Strength: {signal['signal_strength']:.1f}%)", file=output_buffer)
        print(f"   Entry: {signal['entry_price']:.6f} | SL: {signal['stop_loss']:.6f} | TP: {signal['take_profit']:.6f}", file=output_buffer)
        print(f"   Risk/Reward: {signal['risk_reward']:.2f}:1 | ATR: {signal['atr']:.6f} | Volume: {signal['volume']:,.0f}", file=output_buffer)
        
        # Show leading indicator values from 5m timeframe
        if '5m' in signal['timeframe_results']:
            tf_result = signal['timeframe_results']['5m']
            print(f"   RSI: {tf_result.get('rsi', 0):.1f} | Stoch: {tf_result.get('stoch_k', 0):.1f}/{tf_result.get('stoch_d', 0):.1f}", file=output_buffer)
            print(f"   CCI: {tf_result.get('cci', 0):.1f} | Williams: {tf_result.get('williams', 0):.1f}", file=output_buffer)
            print(f"   UO: {tf_result.get('uo', 0):.1f} | AO: {tf_result.get('ao', 0):.6f}", file=output_buffer)
            
            if tf_result.get('sr_levels'):
                print(f"   Support: {[f'{x:.6f}' for x in tf_result['sr_levels']['support_below'][:2]]}", file=output_buffer)
                print(f"   Resistance: {[f'{x:.6f}' for x in tf_result['sr_levels']['resistance_above'][:2]]}", file=output_buffer)
    
    print("\n" + "="*120, file=output_buffer)
    
    output_text = output_buffer.getvalue()
    output_buffer.close()
    
    return output_text

def periodic_analysis():
    """Runs analysis using leading indicators."""
    global best_pairs, sure_shot_signals
    
    while True:
        try:
            # Calculate next minute boundary
            now = datetime.utcnow()
            next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
            sleep_seconds = (next_minute - now).total_seconds()
            
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"SCANNING WITH LEADING INDICATORS AT {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*60}")
            
            # Scan all pairs every 3 minutes for faster reaction
            if now.minute % 3 == 0:
                logger.info("Scanning all pairs with leading indicators...")
                top_pairs_5m, all_results = scan_all_pairs()
                
                # Multi-timeframe analysis on top pairs
                best_pairs = analyze_top_pairs_multitimeframe(top_pairs_5m)
                
                # Filter for sure-shot signals
                sure_shot_signals = [pair for pair in best_pairs 
                                   if pair['signal_strength'] >= MIN_CONFIDENCE 
                                   and pair['risk_reward'] >= MIN_RISK_REWARD
                                   and pair['volume'] > 100000]  # Minimum volume filter
                
                output_buffer = io.StringIO()
                print("\n" + "="*100, file=output_buffer)
                print("LEADING INDICATOR SCAN RESULTS", file=output_buffer)
                print("="*100, file=output_buffer)
                
                print(f"\nTotal pairs analyzed: {len(all_results)}", file=output_buffer)
                print(f"Potential signals found: {len(best_pairs)}", file=output_buffer)
                print(f"High-confidence signals: {len(sure_shot_signals)}", file=output_buffer)
                
                # Display sure-shot signals
                if sure_shot_signals:
                    sure_shot_text = display_sure_shot_signals(sure_shot_signals)
                    print(sure_shot_text, file=output_buffer)
                else:
                    print("\nNo high-confidence signals found.", file=output_buffer)
                    if best_pairs:
                        print("Top 3 potential signals:", file=output_buffer)
                        for i, pair in enumerate(best_pairs[:3], 1):
                            print(f"{i}. {pair['symbol']}: {pair['overall_signal']} ({pair['signal_strength']:.1f}%)", file=output_buffer)
                
                print("\n" + "="*100, file=output_buffer)
                
                output_text = output_buffer.getvalue()
                output_buffer.close()
                
                if gui_instance:
                    gui_instance.update_display(output_text)
                else:
                    print(output_text)
            
            # Monitor sure-shot pairs every minute
            if sure_shot_signals:
                for signal in sure_shot_signals:
                    symbol = signal['symbol']
                    df = get_latest_historical_data(symbol, '5m')
                    if not df.empty:
                        updated_analysis = analyze_pair(symbol, df.copy(), '5m')
                        
                        if updated_analysis and updated_analysis['signal'] != "HOLD":
                            output_buffer = io.StringIO()
                            print(f"\n⚡ UPDATED: {symbol} - {updated_analysis['signal']}", file=output_buffer)
                            print(f"Price: {updated_analysis['current_price']:.6f} | RSI: {updated_analysis.get('rsi', 0):.1f}", file=output_buffer)
                            print(f"SL: {updated_analysis['stop_loss']:.6f} | TP: {updated_analysis['take_profit']:.6f}", file=output_buffer)
                            
                            output_text = output_buffer.getvalue()
                            output_buffer.close()
                            
                            if gui_instance:
                                gui_instance.update_display(output_text)
                
        except Exception as e:
            logger.error(f"Error in periodic analysis: {e}")
            time.sleep(30)  # Shorter wait time for faster recovery

# --- Main Execution ---
if __name__ == "__main__":
    # Create GUI
    root = tk.Tk()
    gui_instance = TradingBotGUI(root)
    
    # Redirect stdout to our thread-safe print
    sys.stdout = ThreadSafePrint(gui_instance)
    
    # Run initial scan
    logger.info("Running initial market scan with leading indicators...")
    top_pairs_5m, all_results = scan_all_pairs()
    
    # Multi-timeframe analysis
    best_pairs = analyze_top_pairs_multitimeframe(top_pairs_5m)
    
    # Filter for sure-shot signals
    sure_shot_signals = []
    if best_pairs:
        sure_shot_signals = [pair for pair in best_pairs 
                           if pair.get('signal_strength', 0) >= MIN_CONFIDENCE 
                           and pair.get('risk_reward', 0) >= MIN_RISK_REWARD
                           and pair.get('volume', 0) > 100000]
    
    # Display initial results
    output_buffer = io.StringIO()
    print("\n" + "="*100, file=output_buffer)
    print("INITIAL LEADING INDICATOR SCAN RESULTS", file=output_buffer)
    print("="*100, file=output_buffer)
    
    print(f"\nTotal pairs analyzed: {len(all_results)}", file=output_buffer)
    print(f"Potential signals found: {len(best_pairs) if best_pairs else 0}", file=output_buffer)
    print(f"High-confidence signals: {len(sure_shot_signals)}", file=output_buffer)
    
    if sure_shot_signals:
        sure_shot_text = display_sure_shot_signals(sure_shot_signals)
        print(sure_shot_text, file=output_buffer)
    else:
        print("\nNo high-confidence signals found in initial scan.", file=output_buffer)
        if best_pairs:
            print("Top 5 potential signals:", file=output_buffer)
            for i, pair in enumerate(best_pairs[:5], 1):
                print(f"{i}. {pair['symbol']}: {pair['overall_signal']} ({pair.get('signal_strength', 0):.1f}%)", file=output_buffer)
    
    print("\n" + "="*100, file=output_buffer)
    
    output_text = output_buffer.getvalue()
    output_buffer.close()
    
    if gui_instance:
        gui_instance.update_display(output_text)
    
    # Start the periodic analysis thread
    analysis_thread = Thread(target=periodic_analysis, daemon=True)
    analysis_thread.start()
    
    logger.info(f"\nLeading indicator analysis running every minute")
    logger.info(f"Full market scan every 3 minutes")
    logger.info(f"Minimum confidence: {MIN_CONFIDENCE}%")
    logger.info(f"Minimum risk/reward: {MIN_RISK_REWARD}")
    logger.info(f"Volume filter: > 100,000 USDT")
    logger.info("\nPress Ctrl+C to exit.")

    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Script finished.")