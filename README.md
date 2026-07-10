# Systematic Crypto Strategy — SG×HK Quant Hackathon

Built for the SG vs HK University Web3 Quant Hackathon. Implements and compares two systematic strategies (logistic regression classifier and SMA momentum with RSI filter) on BTC/USDT hourly data, with a full backtesting engine including Sharpe, Sortino, Calmar, and max drawdown. Deployed live on AWS via Roostoo mock trading API.

## Overview

This trading bot is designed for the **SG vs HK University Web3 Quant Hackathon**, where participants deploy an automatic trading bot in AWS to trade in crypto markets via the Roostoo mock trading API.

The bot implements **two trading strategies**:
1. **Machine Learning (ML) Strategy**: Uses Logistic Regression to predict price direction based on technical indicators
2. **Simple Moving Average (SMA) Strategy**: A rule-based trend-following strategy using moving average crossovers with RSI filter

## Features

### Data Management
- **Binance API Integration**: Fetches historical OHLCV (candlestick) data from Binance public API
- **Local CSV Support**: Load historical data from CSV files for backtesting without API calls
- **Automatic Data Download**: Handles Binance's 1000-candle limit by looping through multiple requests

### Technical Indicators
The bot computes the following technical indicators for feature engineering:
- **Log Returns**: Logarithmic price returns
- **Simple Moving Averages (SMA)**: 5, 10, and 20 period SMAs
- **Exponential Moving Averages (EMA)**: 5 and 10 period EMAs
- **Rolling Volatility**: 10-period standard deviation of returns
- **Relative Strength Index (RSI)**: 14-period RSI for momentum measurement

### Trading Strategies

#### ML Strategy
- Trains a Logistic Regression classifier on 70% of historical data
- Predicts the probability of the next period's return being positive
- Invests in the asset when predicted probability exceeds 55%, otherwise stays in cash
- Features are scaled using StandardScaler for optimal model performance

#### SMA Strategy
- Uses a 50-period short SMA and 200-period long SMA for trend detection
- Additional RSI filter (threshold: 70) to avoid overbought conditions
- Opens positions only when short SMA > long SMA AND RSI < threshold

### Risk Management
- **Transaction Fees**: Properly applies 0.1% market order fee and 0.05% limit order fee
- **Portfolio Tracking**: Real-time portfolio value calculation considering price changes and fees
- **Position Sizing**: Binary position (100% invested or 100% cash) - no partial positions

### Performance Metrics
The backtester computes comprehensive performance metrics:
- **Cumulative Return**: Total return over the backtest period
- **Annualized Return (CAGR)**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return relative to maximum drawdown
- **Maximum Drawdown**: Largest peak-to-trough decline

### API Integration
- **Roostoo Mock API**: Connects to the hackathon's mock trading platform
- **HMAC-SHA256 Signature**: Secure request signing for authenticated endpoints
- **Order Management**: Place, query, and cancel orders
- **Balance Tracking**: Retrieve wallet balances

## Installation

### Prerequisites
- Python 3.8+
- Required packages:
  ```
  pip install numpy pandas requests scikit-learn matplotlib
  ```

### Configuration

1. **API Keys**: The bot uses environment variables for Roostoo API credentials:
   ```bash
   export ROOSTOO_API_KEY="your_api_key"
   export ROOSTOO_SECRET_KEY="your_secret_key"
   ```

2. **Binance Data**: No authentication required for Binance public klines API

## Usage

### Basic Backtest (ML Strategy)
```bash
python trading_bot.py --symbol BTCUSDT --interval 1h --lookback_days 90
```

### Rule-Based SMA Strategy
```bash
python trading_bot.py --symbol BTCUSDT --interval 1h --lookback_days 90 --strategy sma
```

### Using Local CSV Data
```bash
python trading_bot.py --data_file ./data/btc_data.csv --lookback_days 90 --strategy ml
```

### Command Line Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--symbol` | str | BTCUSDT | Trading pair symbol |
| `--interval` | str | 1h | Kline interval (1m, 5m, 15m, 1h, 4h, 1d, etc.) |
| `--lookback_days` | int | 90 | Days of historical data to download |
| `--data_file` | str | None | Path to local CSV file (overrides API) |
| `--strategy` | str | ml | Strategy: 'ml' or 'sma' |

## Output

After running a backtest, the bot will:
1. Display classifier accuracy and classification report (ML strategy)
2. Print performance metrics for the strategy
3. Save an equity curve chart to `charts/equity_curve_{symbol}_{interval}_{strategy}.png`

## Architecture

```
trading_bot.py
├── Configuration
│   ├── Binance API endpoints
│   ├── Roostoo API credentials
│   └── Fee rates
├── Data Retrieval
│   ├── get_binance_klines()     # Single API call
│   ├── download_symbol_data()   # Full history download
│   └── load_local_csv()         # CSV file loader
├── Feature Engineering
│   └── compute_indicators()     # Technical indicators
├── Model Training & Backtesting
│   ├── train_classifier()       # ML model training
│   ├── backtest_strategy()       # ML strategy backtest
│   └── backtest_moving_average_strategy()  # SMA strategy backtest
├── Performance Metrics
│   └── compute_performance_metrics()  # Risk metrics
├── Roostoo API Utilities
│   ├── _sign_payload()          # HMAC signature
│   ├── place_order()            # Order placement
│   ├── query_order()            # Order查询
│   └── cancel_order()           # Order cancellation
└── Live Trading
    └── run_live_bot()           # Production trading skeleton
```

## Bug Fixes Applied

The following issues were identified and fixed in the original code:

1. **Transaction Fee Application in SMA Strategy**: The original `backtest_moving_average_strategy()` did not apply transaction fees when changing positions. This has been corrected to match the ML strategy's fee handling.

2. **Order of Operations**: Fixed the order of portfolio updates and position changes to ensure fees are applied correctly and portfolio value reflects the current position's gains/losses.

3. **Feature Column Selection**: Added `_get_feature_cols()` helper function to properly exclude non-feature columns (like 'ignore', 'log_ret', 'desired_position') from model training.

## Notes

- **No Live Trading Implementation**: The `run_live_bot()` function is a skeleton demonstrating the architecture. A production implementation would require additional error handling, logging, and persistence.
- **Past Performance Disclaimer**: Historical backtest results do not guarantee future performance.
- **Risk Warning**: Cryptocurrency trading involves substantial risk of loss. This code is for educational purposes only.

## Hackathon Information

- **Problem Statement**: [Notion Link](https://roostoo.notion.site/Problem-Statement-SG-vs-HK-University-Web3-Quant-Hackathon-309ba22fed7980a79da6d8a08b5216c9)
- **Info Session**: [Pitch.com Link](https://pitch.com/v/sg-vs-hk-quant-hackathon-info-session-5e6wkq)
