import datetime as _dt
import hashlib as _hashlib
import hmac as _hmac
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Binance endpoint for public klines.  No authentication is required.
BINANCE_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

# Roostoo mock trading API configuration.  Replace API_KEY and SECRET_KEY
# with credentials provided by the hackathon organizers.  These keys are
# deliberately left blank.  Do not commit real keys to version control.
BASE_URL = "https://mock-api.roostoo.com"
API_KEY = os.getenv("ROOSTOO_API_KEY", "fl17xHP5vt0BSEjfPktZvaEST91jxGv9Dt7bg1viL9sgSjqJCiCx0BkqzMMNQnL5")
SECRET_KEY = os.getenv("ROOSTOO_SECRET_KEY", "iaZOFpbKpsIurl6YMN3dGKTU9AM1lbe1Z7nqRZoRVvdBbvqASl0OZIBf1EHRaOki")

# Transaction costs.  According to the hackathon rules, a market order
# incurs a 0.1% fee and a maker (limit) order incurs a 0.05% fee【858205458375192†screenshot】.
MARKET_FEE_RATE = 0.001
LIMIT_FEE_RATE = 0.0005


# ---------------------------------------------------------------------------
# Data Retrieval
# ---------------------------------------------------------------------------
def get_binance_klines(symbol: str, interval: str, start_time: Optional[int] = None,
                       end_time: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
    """
    Download historical klines (candlesticks) from Binance.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g. 'BTCUSDT').
    interval : str
        Kline interval (e.g. '1h', '15m').  Binance supports intervals from 1m
        up to 1 month.
    start_time : int, optional
        Millisecond timestamp for the start of the data.  If omitted, Binance
        will return the most recent ``limit`` klines.
    end_time : int, optional
        Millisecond timestamp for the end of the data.  Default is ``None``.
    limit : int
        Maximum number of klines per request (max 1000 for Binance).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [open_time, open, high, low, close, volume, close_time,
        quote_asset_volume, number_of_trades, taker_buy_base_volume,
        taker_buy_quote_volume, ignore].  Timestamps are converted to
        Python datetime in UTC.
    """
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time

    response = requests.get(BINANCE_BASE_URL + KLINES_ENDPOINT, params=params)
    response.raise_for_status()
    data = response.json()
    if not data:
        raise ValueError(f"No kline data returned for {symbol} {interval}")

    # Construct DataFrame
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume",
                    "taker_buy_base_volume", "taker_buy_quote_volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)
    # Convert timestamps to datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def download_symbol_data(symbol: str, interval: str, lookback_days: int = 90) -> pd.DataFrame:
    """
    Download up to ``lookback_days`` worth of historical data.  Binance limits
    requests to 1000 candles per call, so this function loops until all data
    is retrieved.  The function uses UTC times.  Set lookback_days high enough
    to cover both training and test periods.
    """
    end_time = int(_dt.datetime.utcnow().timestamp() * 1000)
    ms_per_interval = {
        '1m': 60_000, '3m': 3 * 60_000, '5m': 5 * 60_000, '15m': 15 * 60_000,
        '30m': 30 * 60_000, '1h': 60 * 60_000, '2h': 2 * 60 * 60_000,
        '4h': 4 * 60 * 60_000, '6h': 6 * 60 * 60_000, '8h': 8 * 60 * 60_000,
        '12h': 12 * 60 * 60_000, '1d': 24 * 60 * 60_000
    }
    interval_ms = ms_per_interval.get(interval)
    if interval_ms is None:
        raise ValueError(f"Unsupported interval: {interval}")
    start_time = end_time - lookback_days * 24 * 60 * 60 * 1000

    all_dfs: List[pd.DataFrame] = []
    while True:
        df = get_binance_klines(symbol, interval, start_time, end_time, limit=1000)
        if df.empty:
            break
        all_dfs.append(df)
        earliest = int(df["open_time"].iloc[0].timestamp() * 1000)
        # Stop if we have reached the start_time
        if earliest <= start_time + interval_ms:
            break
        # Move the end_time pointer backwards
        end_time = earliest - interval_ms
    full_df = pd.concat(all_dfs, ignore_index=True)
    # Sort chronologically
    full_df.sort_values("open_time", inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    return full_df


# ---------------------------------------------------------------------------
# Local CSV Data Loader
# ---------------------------------------------------------------------------
def load_local_csv(file_path: str) -> pd.DataFrame:
    """
    Load cryptocurrency price data from a locally stored CSV file.

    This helper enables backtesting without making external API calls.  The
    expected CSV format corresponds to the output from CryptoDataDownload
    (https://www.cryptodatadownload.com), which includes columns ``Unix``,
    ``Date``, ``Symbol``, ``Open``, ``High``, ``Low``, ``Close``, ``Volume BTC``,
    ``Volume USDT`` and ``tradecount``.  The function will parse the
    timestamps, rename columns to match the Binance API field names, and
    return a DataFrame sorted in chronological order.

    Parameters
    ----------
    file_path : str
        Path to the CSV file on disk.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``open_time``, ``open``, ``high``, ``low``,
        ``close``, ``volume`` and ``open_time`` is a timezone aware datetime in UTC.

    Notes
    -----
    The first row of CryptoDataDownload files often contains a URL comment,
    which this loader skips automatically.  The loader will also reverse
    the DataFrame so that the earliest timestamp appears first.
    """
    # Read the CSV, skipping the first row if it contains a URL comment
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        # Detect a URL comment at the top
        skiprows = 1 if first_line.startswith('http') else 0
    df = pd.read_csv(file_path, skiprows=skiprows)
    # Rename columns to Binance convention
    rename_map = {
        'Date': 'open_time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume BTC': 'volume'
    }
    df = df.rename(columns=rename_map)
    # Convert timestamps to datetime and ensure UTC
    # Parse timestamps; use errors='coerce' to handle occasional fractional seconds
    df['open_time'] = pd.to_datetime(df['open_time'], utc=True, errors='coerce')
    # Sort chronologically (CryptoDataDownload files are often in reverse order)
    df = df.sort_values('open_time').reset_index(drop=True)
    # Convert numeric columns from strings to floats
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop rows with any missing numeric values
    df.dropna(subset=numeric_cols + ['open_time'], inplace=True)
    # Retain only the columns needed for analysis
    return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for use as features.  The indicators include:

    * log returns: log(close / close.shift(1))
    * simple moving averages (SMA) over 5, 10 and 20 periods
    * exponential moving averages (EMA) over 5 and 10 periods
    * rolling volatility (standard deviation) over 10 periods of returns
    * Relative Strength Index (RSI) over 14 periods

    The resulting DataFrame is aligned with the original input and includes the
    computed features and a binary target column ``target`` indicating whether
    the next period's return is positive.
    """
    df = df.copy()
    # Log returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    # Simple moving averages
    for window in (5, 10, 20):
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
    # Exponential moving averages
    for span in (5, 10):
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    # Rolling volatility of returns
    df['volatility_10'] = df['log_ret'].rolling(10).std()
    # RSI implementation
    rsi_window = 14
    delta = df['close'].diff()
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up).rolling(rsi_window).mean()
    roll_down = pd.Series(down).rolling(rsi_window).mean()
    rs = roll_up / roll_down
    df['rsi'] = 100 - 100 / (1 + rs)
    # Target: 1 if next log return > 0 else 0
    df['target'] = (df['log_ret'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Model Training and Backtesting
# ---------------------------------------------------------------------------
def train_classifier(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[LogisticRegression, StandardScaler, pd.DataFrame, pd.DataFrame]:
    """
    Train a logistic regression model to predict the direction of the next return.
    Splits the data chronologically into train and test sets.

    Returns
    -------
    model : LogisticRegression
        Fitted classifier.
    scaler : StandardScaler
        Fitted scaler used to normalize features.
    train_df : pd.DataFrame
        Training subset with features and target.
    test_df : pd.DataFrame
        Test subset with features and target.
    """
    # Use a chronological split instead of random train_test_split because time
    # ordering matters in financial data.  70% training, 30% testing.
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    y_train = train_df['target']
    y_test = test_df['target']
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Classifier accuracy on test set: {acc:.3f}")
    print(classification_report(y_test, y_pred))
    # Store predictions for backtesting
    test_df['pred_prob'] = model.predict_proba(X_test)[:, 1]
    test_df['pred'] = y_pred
    return model, scaler, train_df, test_df


def backtest_strategy(df: pd.DataFrame, feature_cols: List[str], model: LogisticRegression,
                      scaler: StandardScaler, trade_fee_rate: float = MARKET_FEE_RATE) -> pd.DataFrame:
    """
    Backtest a simple long‑only strategy using the trained model.

    The strategy invests the entire portfolio in the asset when the predicted
    probability of a positive return exceeds 0.55.  Otherwise, it stays in cash.
    The threshold can be adjusted for different risk appetites.  When entering
    or exiting a position, a transaction cost (fee) is deducted from the
    portfolio.  The backtester computes the portfolio value after each
    prediction step.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the columns in ``feature_cols``,
        'close' and 'pred_prob'.
    feature_cols : list of str
        Feature column names used by the model.
    model : LogisticRegression
        Trained classifier.
    scaler : StandardScaler
        Fitted scaler.
    trade_fee_rate : float
        Fee rate applied to position changes (default uses market fee).

    Returns
    -------
    results : pd.DataFrame
        DataFrame with columns ['timestamp', 'price', 'position', 'portfolio']
        representing the equity curve of the strategy.
    """
    # Copy test data to avoid modifying original
    test_df = df.copy()
    # Normalize features and compute predicted probabilities if not provided
    if 'pred_prob' not in test_df.columns:
        X = scaler.transform(test_df[feature_cols])
        test_df['pred_prob'] = model.predict_proba(X)[:, 1]
    # Initialize variables
    portfolio = 1.0  # start with 1 unit of cash; actual dollar amount scales linearly
    position = 0.0   # 0 means fully in cash; 1 means fully invested in asset
    results = []
    prev_price = None
    for idx, row in test_df.iterrows():
        price = row['close']
        prob = row['pred_prob']
        # Determine desired position: invest if prob > threshold
        desired_position = 1.0 if prob > 0.55 else 0.0
        # If position changes, apply transaction fee
        if desired_position != position:
            # Compute fee on traded amount
            traded_amount = abs(desired_position - position) * portfolio
            fee = traded_amount * trade_fee_rate
            portfolio -= fee
            position = desired_position
        # Update portfolio based on price change
        if prev_price is not None:
            # When invested, portfolio value moves with price; when in cash it stays constant
            portfolio *= (1 + position * (price - prev_price) / prev_price)
        results.append({
            'timestamp': row['open_time'],
            'price': price,
            'position': position,
            'portfolio': portfolio
        })
        prev_price = price
    results_df = pd.DataFrame(results)
    return results_df


# ---------------------------------------------------------------------------
# Rule‑Based Strategy (Moving Average Cross with RSI Filter)
# ---------------------------------------------------------------------------
def backtest_moving_average_strategy(df: pd.DataFrame, short_window: int = 50, long_window: int = 200,
                                     rsi_threshold: float = 70.0) -> pd.DataFrame:
    """
    Backtest a simple trend‑following strategy using moving average crossovers and
    an RSI filter.  The strategy invests in the asset when the short‑term
    moving average is above the long‑term moving average (indicating a
    bullish trend) *and* the RSI is below a threshold to avoid overbought
    conditions.  Otherwise it stays in cash.  No leverage is used.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the 'close', 'open_time' columns.  The
        function computes the SMA and RSI internally if not already present.
    short_window : int, optional
        Length of the short moving average window (default 50 periods).
    long_window : int, optional
        Length of the long moving average window (default 200 periods).
    rsi_threshold : float, optional
        RSI level above which the market is considered overbought.  Positions
        are only opened when RSI < threshold (default 70).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['timestamp', 'price', 'position', 'portfolio']
        representing the equity curve of the strategy.
    """
    df = df.copy()
    # Compute moving averages if not present
    if f'sma_{short_window}' not in df.columns:
        df[f'sma_{short_window}'] = df['close'].rolling(short_window).mean()
    if f'sma_{long_window}' not in df.columns:
        df[f'sma_{long_window}'] = df['close'].rolling(long_window).mean()
    # Compute RSI if not present
    if 'rsi' not in df.columns:
        # Use 14‑period RSI
        rsi_window = 14
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)
        roll_up = up.rolling(rsi_window).mean()
        roll_down = down.rolling(rsi_window).mean()
        rs = roll_up / roll_down
        df['rsi'] = 100 - 100 / (1 + rs)
    # Determine positions
    df['position'] = ((df[f'sma_{short_window}'] > df[f'sma_{long_window}']) & (df['rsi'] < rsi_threshold)).astype(float)
    # Backtest equity curve
    portfolio = 1.0
    position = 0.0
    results = []
    prev_price = None
    for idx, row in df.iterrows():
        price = row['close']
        # Update portfolio based on prior position
        if prev_price is not None:
            portfolio *= (1 + position * (price - prev_price) / prev_price)
        # Update position for next period
        position = row['position']
        results.append({
            'timestamp': row['open_time'],
            'price': price,
            'position': position,
            'portfolio': portfolio
        })
        prev_price = price
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------
def compute_performance_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Compute cumulative return, annualized Sharpe ratio, Sortino ratio and Calmar
    ratio from a backtest equity curve.  The functions use definitions from
    reputable sources: the Sharpe ratio is the excess return divided by total
    volatility【803676202467100†L222-L244】, the Sortino ratio is the excess return divided
    by downside deviation【59786064410786†L215-L264】, and the Calmar ratio is the
    annualized return divided by maximum drawdown【654916195359731†L218-L266】.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame with a 'portfolio' column representing the equity curve over time.

    Returns
    -------
    dict
        Dictionary containing the computed metrics.
    """
    # Compute periodic returns of the equity curve
    portfolio = results['portfolio'].values
    returns = np.diff(portfolio) / portfolio[:-1]
    # Cumulative return
    cumulative_return = portfolio[-1] / portfolio[0] - 1.0
    # Annualize based on the frequency of observations.  Approximate number of
    # periods per year using the time delta between the first two timestamps.
    timestamps = results['timestamp']
    if len(timestamps) > 1:
        delta = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds()
        periods_per_year = 365.0 * 24 * 60 * 60 / delta
    else:
        periods_per_year = 1.0
    # Annualized return (CAGR)
    # Use geometric mean to avoid numerical overflow when compounding large
    # returns over many periods.  The CAGR is computed as the growth factor
    # raised to the ratio of periods per year to number of observations.
    # When the period count is zero (no trades), default to cumulative return.
    if len(returns) > 0:
        cagr = (portfolio[-1] / portfolio[0]) ** (periods_per_year / len(returns)) - 1
    else:
        cagr = cumulative_return
    # Risk-free rate assumed zero for crypto; subtract risk-free (rf) from returns
    excess_returns = returns  # rf = 0
    mean_return = excess_returns.mean()
    std_return = excess_returns.std(ddof=1)
    sharpe = mean_return / std_return * np.sqrt(periods_per_year) if std_return != 0 else np.nan
    # Sortino ratio: use downside deviation
    downside_returns = np.where(excess_returns < 0, excess_returns, 0)
    downside_std = np.sqrt((downside_returns ** 2).mean())
    sortino = mean_return / downside_std * np.sqrt(periods_per_year) if downside_std != 0 else np.nan
    # Calmar ratio: annualized return divided by max drawdown
    # Compute running max and drawdowns
    running_max = np.maximum.accumulate(portfolio)
    drawdowns = (running_max - portfolio) / running_max
    max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0
    calmar = cagr / max_drawdown if max_drawdown != 0 else np.nan
    return {
        'cumulative_return': cumulative_return,
        'annualized_return': cagr,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
    }


# ---------------------------------------------------------------------------
# Roostoo API Utilities
# ---------------------------------------------------------------------------
def _get_timestamp() -> str:
    """Return a 13‑digit millisecond timestamp as a string."""
    return str(int(time.time() * 1000))


def _sign_payload(payload: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str], str]:
    """
    Generate signed headers and total parameter string for Roostoo signed
    endpoints.  Follows the sample code in the Roostoo API documentation【56524916154858†L810-L947】.

    Parameters
    ----------
    payload : dict
        Dictionary of request parameters to be included in the signature.

    Returns
    -------
    headers : dict
        Headers containing the API key and message signature.
    payload : dict
        Payload including the timestamp.
    total_params : str
        Concatenated parameter string sorted by keys.
    """
    payload = payload.copy()
    payload['timestamp'] = _get_timestamp()
    sorted_keys = sorted(payload.keys())
    total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)
    signature = _hmac.new(
        SECRET_KEY.encode('utf-8'),
        total_params.encode('utf-8'),
        _hashlib.sha256
    ).hexdigest()
    headers = {
        'RST-API-KEY': API_KEY,
        'MSG-SIGNATURE': signature
    }
    return headers, payload, total_params


def roostoo_public_get(endpoint: str, params: Optional[Dict[str, str]] = None) -> Dict:
    """
    Send a GET request to a public Roostoo API endpoint.  No signature is
    required.
    """
    url = f"{BASE_URL}{endpoint}"
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling {endpoint}: {e}")
        return {}


def roostoo_signed_get(endpoint: str, payload: Dict[str, str]) -> Dict:
    """Send a signed GET request to a Roostoo API endpoint."""
    headers, payload, _ = _sign_payload(payload)
    url = f"{BASE_URL}{endpoint}"
    try:
        res = requests.get(url, headers=headers, params=payload)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling {endpoint}: {e}")
        return {}


def roostoo_signed_post(endpoint: str, payload: Dict[str, str]) -> Dict:
    """Send a signed POST request to a Roostoo API endpoint."""
    headers, payload, total_params = _sign_payload(payload)
    headers['Content-Type'] = 'application/x-www-form-urlencoded'
    url = f"{BASE_URL}{endpoint}"
    try:
        res = requests.post(url, headers=headers, data=total_params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling {endpoint}: {e}")
        return {}


def check_server_time() -> Dict:
    """Check the server time (public endpoint)."""
    return roostoo_public_get("/v3/serverTime")


def get_exchange_info() -> Dict:
    """Get available trading pairs and info (public endpoint)."""
    return roostoo_public_get("/v3/exchangeInfo")


def get_balance() -> Dict:
    """Retrieve wallet balances (signed endpoint)."""
    return roostoo_signed_get("/v3/balance", {})


def place_order(pair: str, side: str, quantity: float, price: Optional[float] = None,
                order_type: Optional[str] = None) -> Dict:
    """
    Place a LIMIT or MARKET order.  By default, uses a MARKET order unless a
    price is specified.  The pair should be formatted as 'COIN/USD'.  See the
    sample code in the Roostoo API docs for additional details【56524916154858†L920-L954】.
    """
    if order_type is None:
        order_type = 'LIMIT' if price is not None else 'MARKET'
    if order_type == 'LIMIT' and price is None:
        raise ValueError("LIMIT orders require a price")
    payload = {
        'pair': pair,
        'side': side.upper(),
        'type': order_type.upper(),
        'quantity': str(quantity)
    }
    if order_type == 'LIMIT':
        payload['price'] = str(price)
    return roostoo_signed_post("/v3/place_order", payload)


def query_order(order_id: Optional[int] = None, pair: Optional[str] = None,
                pending_only: Optional[bool] = None) -> Dict:
    """Query order history or pending orders (signed endpoint)."""
    payload: Dict[str, str] = {}
    if order_id is not None:
        payload['order_id'] = str(order_id)
    elif pair is not None:
        payload['pair'] = pair
        if pending_only is not None:
            payload['pending_only'] = 'TRUE' if pending_only else 'FALSE'
    return roostoo_signed_post("/v3/query_order", payload)


def cancel_order(order_id: Optional[int] = None, pair: Optional[str] = None) -> Dict:
    """Cancel specific or all pending orders (signed endpoint)."""
    payload: Dict[str, str] = {}
    if order_id is not None:
        payload['order_id'] = str(order_id)
    elif pair is not None:
        payload['pair'] = pair
    return roostoo_signed_post("/v3/cancel_order", payload)


# ---------------------------------------------------------------------------
# Live Trading Loop (Skeleton)
# ---------------------------------------------------------------------------
def run_live_bot(symbol: str = 'BTCUSDT', interval: str = '1h',
                 threshold: float = 0.55, lookback_intervals: int = 100) -> None:
    """
    Skeleton for running the trading bot in live mode.  This function
    demonstrates how to continuously fetch recent data, generate a trading
    signal with the trained model and place orders on the Roostoo exchange.

    NOTE:  This function does not provide a complete implementation.  Running
    an automated trading bot in production requires robust error handling,
    logging, persistence and concurrency management.  You should expand this
    skeleton according to your own requirements.
    """
    # Fetch exchange info to determine the correct pair format for Roostoo
    # E.g. Binance uses BTCUSDT but Roostoo may use 'BTC/USD'.
    exchange_info = get_exchange_info()
    print("Exchange info:", exchange_info)
    # Download historical data to train the model (initially)
    data = download_symbol_data(symbol, interval, lookback_days=30)
    data = compute_indicators(data)
    feature_cols = [col for col in data.columns if col not in {'open_time','close_time','target','pred','pred_prob'}]
    model, scaler, train_df, test_df = train_classifier(data, feature_cols)
    # Run an infinite loop to fetch the latest interval data and trade
    while True:
        try:
            # Fetch the latest lookback_intervals data points
            end = int(_dt.datetime.utcnow().timestamp() * 1000)
            start = end - lookback_intervals * {
                '1m': 60_000, '3m': 3 * 60_000, '5m': 5 * 60_000,
                '15m': 15 * 60_000, '30m': 30 * 60_000, '1h': 60 * 60_000,
                '2h': 2 * 60 * 60_000, '4h': 4 * 60 * 60_000, '6h': 6 * 60 * 60_000,
                '8h': 8 * 60 * 60_000, '12h': 12 * 60 * 60_000, '1d': 24 * 60 * 60_000
            }[interval]
            df_live = get_binance_klines(symbol, interval, start, end, limit=lookback_intervals)
            df_live = compute_indicators(df_live)
            # Use the most recent row for prediction
            latest_row = df_live.iloc[-1]
            features = latest_row[feature_cols].values.reshape(1, -1)
            features_scaled = scaler.transform(features)
            prob = model.predict_proba(features_scaled)[0, 1]
            print(f"Predicted probability of next positive return: {prob:.3f}")
            # Determine desired position
            desired_position = 1.0 if prob > threshold else 0.0
            # Query current balance and existing orders to decide whether to trade
            balance = get_balance()
            # ... implement logic to place or cancel orders based on desired_position ...
            # Sleep until next interval
            interval_seconds = {
                '1m': 60, '3m': 3 * 60, '5m': 5 * 60, '15m': 15 * 60,
                '30m': 30 * 60, '1h': 60 * 60, '2h': 2 * 60 * 60,
                '4h': 4 * 60 * 60, '6h': 6 * 60 * 60, '8h': 8 * 60 * 60,
                '12h': 12 * 60 * 60, '1d': 24 * 60 * 60
            }[interval]
            time.sleep(interval_seconds)
        except Exception as e:
            print(f"Error in live bot loop: {e}")
            time.sleep(10)


# ---------------------------------------------------------------------------
# Charting Utility
# ---------------------------------------------------------------------------
def save_equity_curve_chart(results: pd.DataFrame, filename: str) -> None:
    """
    Save a chart of the portfolio equity curve to the specified filename.
    The function will create the directory if it doesn't exist.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(results['timestamp'], results['portfolio'], label='Portfolio value')
    plt.title('Portfolio Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Portfolio (relative to initial)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def main(symbol: str = 'BTCUSDT', interval: str = '1h', lookback_days: int = 90,
         data_file: Optional[str] = None, strategy: str = 'ml') -> None:
    """
    High‑level routine to orchestrate the backtest.  This function will either
    load data from a local CSV file (if ``data_file`` is provided) or download
    data from Binance's public API.  The subsequent steps compute indicators,
    train the model, backtest the strategy, print performance metrics and
    save an equity curve chart.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (ignored when ``data_file`` is provided).
    interval : str
        Kline interval for Binance API (ignored when ``data_file`` is provided).
    lookback_days : int
        Number of days of data to download when using Binance API.
    data_file : str, optional
        Path to a local CSV file containing historical OHLCV data.  If
        supplied, the downloader is skipped and this file is used instead.
    """
    # Ensure charts directory exists
    os.makedirs('charts', exist_ok=True)
    if data_file:
        print(f"Loading local data from {data_file}...")
        df = load_local_csv(data_file)
        # Restrict to the most recent ``lookback_days`` of data from the CSV
        if lookback_days is not None and lookback_days > 0:
            max_time = df['open_time'].max()
            start_time = max_time - pd.Timedelta(days=lookback_days)
            df = df[df['open_time'] >= start_time].reset_index(drop=True)
    else:
        print(f"Downloading {lookback_days} days of {interval} data for {symbol}...")
        df = download_symbol_data(symbol, interval, lookback_days)
    if strategy == 'ml':
        # Machine learning strategy
        df = compute_indicators(df)
        feature_cols = [col for col in df.columns if col not in {'open_time','close_time','target','pred','pred_prob'}]
        model, scaler, train_df, test_df = train_classifier(df, feature_cols)
        backtest_results = backtest_strategy(test_df, feature_cols, model, scaler)
        metrics = compute_performance_metrics(backtest_results)
        print("Performance metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        # Save equity curve chart
        chart_filename = f'equity_curve_{symbol}_{interval}_ml.png'
        chart_path = os.path.join('charts', chart_filename)
        save_equity_curve_chart(backtest_results, chart_path)
        print(f"Equity curve chart saved to {chart_path}")
    elif strategy == 'sma':
        # Rule‑based moving average strategy
        df = compute_indicators(df)
        backtest_results = backtest_moving_average_strategy(df)
        metrics = compute_performance_metrics(backtest_results)
        print("Performance metrics (moving average strategy):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        # Save equity curve chart
        chart_filename = f'equity_curve_{symbol}_{interval}_sma.png'
        chart_path = os.path.join('charts', chart_filename)
        save_equity_curve_chart(backtest_results, chart_path)
        print(f"Equity curve chart saved to {chart_path}")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Backtest AI-driven trading strategy')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol (e.g. BTCUSDT)')
    parser.add_argument('--interval', type=str, default='1h', help='Kline interval (e.g. 1h, 15m, 1d)')
    parser.add_argument('--lookback_days', type=int, default=90, help='Number of days of historical data to download')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to local CSV file for backtesting (overrides API download)')
    parser.add_argument('--strategy', type=str, default='ml', choices=['ml','sma'],
                        help='Trading strategy to use: ml (machine learning) or sma (moving average)')
    args = parser.parse_args()
    main(args.symbol, args.interval, args.lookback_days, args.data_file, args.strategy)