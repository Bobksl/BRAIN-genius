# Systematic Crypto Strategies — SG×HK Quant Hackathon

This repository contains the trading system I built for the **SG vs HK University Web3 Quant Hackathon**, deploying an automatic trading bot on AWS to trade crypto via the Roostoo mock trading API.

The project implements and compares **two systematic strategies** on Binance BTCUSDT hourly data:

1. **Logistic Regression Strategy**  
   - Uses technical features (moving averages, volatility, RSI, log returns)  
   - Trains a classifier to predict next-period positive return  
   - Takes long/cash decisions when predicted probability exceeds a threshold

2. **SMA + RSI Momentum Strategy**  
   - 50/200 SMA crossover for trend detection  
   - RSI filter to avoid extended overbought regimes  
   - Simple binary position model (fully invested vs cash)

---

## What I Learned

- **End-to-end data pipeline**:  
  Fetch OHLCV from Binance, handle API pagination limits, and optionally load historical data from local CSVs.

- **Feature engineering for signals**:  
  Built a feature set of SMAs/EMAs, rolling volatility, RSI, and log returns for supervised learning.

- **Backtesting with risk metrics**:  
  Implemented a small backtest engine that tracks portfolio equity, fees (market vs limit), and computes performance statistics (CAGR, Sharpe, Sortino, Calmar, max drawdown).

- **API-based execution**:  
  Integrated the Roostoo mock trading API with HMAC-SHA256 signing to place/cancel/query orders and monitor balances.

This project is intentionally educational: it focuses on clarity of implementation and risk reporting rather than production-grade infrastructure.

---

## Project Structure

- `trading_bot.py`  
  Single entry point that orchestrates:
  - Data download and CSV loading
  - Indicator computation and feature generation
  - Model training (Logistic Regression) and backtesting
  - Rule-based SMA+RSI strategy backtest
  - Performance metric calculation and chart saving
  - Roostoo API interaction utilities
  - A skeleton `run_live_bot()` for future extension

---

## Key Components

- **Data Management**
  - Binance public OHLCV API integration
  - Local CSV data support for offline experiments
  - Automatic multi-request download to bypass 1000-candle limit

- **Strategies**
  - `ml` mode: Logistic Regression classifier + probability threshold
  - `sma` mode: SMA crossover + RSI filter

- **Risk & Performance**
  - Explicit transaction fee modelling (0.1% market, 0.05% limit)
  - Portfolio tracking with position state and equity curve
  - Metrics: cumulative return, CAGR, Sharpe, Sortino, Calmar, max drawdown

- **Hackathon Integration**
  - Roostoo API: authenticated requests, order management, balance queries
  - Command-line interface for quick backtests on BTCUSDT and other symbols

---

## Usage Examples

Basic ML strategy backtest:

```bash
python trading_bot.py --symbol BTCUSDT --interval 1h --lookback_days 90 --strategy ml
```

Rule-based SMA strategy:

```bash
python trading_bot.py --symbol BTCUSDT --interval 1h --lookback_days 90 --strategy sma
```

Using local CSV data:

```bash
python trading_bot.py --data_file ./data/btc_data.csv --lookback_days 90 --strategy ml
```

---

## Notes

- Crypto trading carries significant risk; this project is for **educational and hackathon** purposes only.
- The live trading function is a skeleton and would need robust error handling, logging, and persistence for production use.
  
## Hackathon Information

- **Problem Statement**: [Notion Link](https://roostoo.notion.site/Problem-Statement-SG-vs-HK-University-Web3-Quant-Hackathon-309ba22fed7980a79da6d8a08b5216c9)
- **Info Session**: [Pitch.com Link](https://pitch.com/v/sg-vs-hk-quant-hackathon-info-session-5e6wkq)
