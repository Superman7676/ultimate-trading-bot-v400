# 🚀 Ultimate Trading System v400

**Automated Stock Analysis & Trading Bot with 10 ML Models, 43-Column Excel Reports, and 462-Stock Watchlist**

## ✨ Features

### 🤖 Machine Learning (10 Models)
- LSTM (Deep Learning)
- XGBoost
- Random Forest
- Gradient Boosting
- SVR (Support Vector Regression)
- LightGBM
- ARIMA (Time Series)
- Facebook Prophet
- Linear Regression
- Exponential Smoothing

### 📊 Technical Analysis
- **61+ Candlestick Patterns** (TA-Lib)
- **15+ Chart Patterns** (Head & Shoulders, Cup & Handle, Triangles)
- **TD Sequential D9** (Tom DeMark Algorithm)
- **100+ Technical Indicators** (RSI, MACD, ADX, Stochastic, Bollinger Bands)
- **Bull/Bear Trap Detection**
- **Support/Resistance Levels** (Pivot Points, Fibonacci)

### 📈 Reports
- **43-Column Excel × 6 Sheets:**
  - ALL stocks analysis
  - BUY_STRONG (Bullish)
  - SHORT_STRONG (Bearish)
  - KEY_LEVELS (Support/Resistance)
  - MARKET_NEWS (Index summary)
  - PATTERNS (Chart patterns)

### 🗂️ Watchlist
- **462 Stocks** (Tech, Finance, Healthcare, etc.)
- **Customizable** (Add/Remove stocks)
- **Auto-saved** (watchlist.json)

### 📰 News Integration
- Finnhub
- Alpha Vantage
- Polygon
- TwelveData
- NewsAPI
- MarketAux
- RapidAPI

### 🔄 Automation
- **Auto Reports** every 30 minutes
- **Background Tasks** (asyncio)
- **Database** (SQLite)
- **Logging** (Rotating)

### 💬 Telegram Bot
- Real-time commands
- Interactive analysis
- Report generation
- Watchlist management

## 🔧 Installation

### 1. Clone Repository
```bash
git clone https://github.com/Superman7676/ultimate-trading-bot-v400.git
cd ultimate-trading-bot-v400
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup .env File
```bash
cp .env.example .env
```

**Edit `.env` with your API keys and Telegram token:**
```
BOT_TOKEN=your_telegram_bot_token
AUTHORIZED_USER_ID=your_telegram_id
FINNHUB_API_KEY=your_key
# ... add other API keys
```

### 4. Run Bot
```bash
python bot.py
```

## 📱 Telegram Commands

| Command | Description |
|---------|-------------|
| `/a SYMBOL` | Full Analysis (100+ indicators) |
| `/pn SYMBOL` | Pattern + News Detection |
| `/scan` | Full Market Scan |
| `/predict SYMBOL` | 10 ML Models Prediction |
| `/backtest SYMBOL` | 4-Year Backtest |
| `/report` | Excel Report (43×6) |
| `/price SYMBOL` | Current Price |
| `/add SYMBOL` | Add to Watchlist |
| `/remove SYMBOL` | Remove from Watchlist |
| `/list` | Show Watchlist |

## 📁 Project Structure

```
ultimate-trading-bot-v400/
├── bot.py                 # Main bot (all features)
├── watchlist.json         # 462 stocks (auto-created)
├── trading_system.db      # SQLite database (auto-created)
├── requirements.txt       # Dependencies
├── .env.example          # API template
├── .env                  # Your credentials (NOT committed)
├── .gitignore           # Ignore sensitive files
├── README.md            # Documentation
└── reports/             # Excel reports (auto-created)
```

## 🔒 Security

**⚠️ IMPORTANT:** Your `.env` file is in `.gitignore` for safety. API keys are loaded from environment variables only.

## 🚀 Features Breakdown

### 1. Full Technical Analysis
- 100+ indicators calculated automatically
- Real-time data from yFinance
- Pattern detection (61+ candlestick patterns)
- Chart pattern recognition
- Bull/Bear trap alerts

### 2. Machine Learning Predictions
- Each stock gets 10 predictions
- Ensemble average
- Confidence levels for each model
- Validation to prevent unrealistic predictions

### 3. Excel Reports (Production-Ready)
- **43 Columns** with all indicators, levels, and factors
- **6 Sheets** for different analysis views
- Bullish/Bearish filtering
- Support/Resistance levels
- Trading entry/exit levels
- Hebrew factors explanation

### 4. Auto Reports
- Generates every 30 minutes
- Analyzes 462 stocks
- Saves to `/reports` folder
- Takes 2-5 minutes per run

### 5. Backtesting
- 4 years of historical data
- RSI + MA Crossover strategy
- Win rate, Sharpe ratio, max drawdown
- Trade-by-trade history

## 🔗 API Keys

Get free API keys from:
- [Finnhub](https://finnhub.io) - Stock data
- [Alpha Vantage](https://alphavantage.co) - Financial data
- [Polygon](https://polygon.io) - Market data
- [TwelveData](https://twelvedata.com) - Time series
- [NewsAPI](https://newsapi.org) - News
- [MarketAux](https://www.marketaux.com) - Market data
- [RapidAPI](https://rapidapi.com) - Various APIs

## 📈 Performance

- **Speed:** 400+ stocks scanned in 2-5 minutes
- **Accuracy:** Multiple models ensure reliable predictions
- **Reliability:** Fallback logic for API failures
- **Scalability:** No stock limit (currently 462)

## 🐛 Troubleshooting

### Bot doesn't start
```bash
python bot.py
# Check: Is BOT_TOKEN in .env correct?
```

### No data for symbol
```bash
# Try: /a AAPL (use valid US stock symbol)
```

### Excel report fails
```bash
pip install openpyxl
```

### TA-Lib import fails
```bash
# Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Linux: sudo apt-get install build-essential && pip install TA-Lib
# Mac: brew install ta-lib && pip install TA-Lib
```

## 📝 License

MIT License - Use, modify, and distribute freely!

## 👨‍💻 Author

Built for automated trading analysis and stock pattern recognition.

---

🚀 **Happy Trading!**
