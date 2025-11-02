# Crypto Trading Bot - Technical Analysis & Machine Learning

## 📊 Overview

Automated cryptocurrency trading system comparing two strategies:
- **Standard Bot** - Traditional technical indicator-based trading
- **XGBoost Bot** - Intelligent trading with Machine Learning
- **Automated backtesting** and performance evaluation
- **Visualization** with detailed charts

## 🎯 Experimental Results

| Bot | Profit | Win Rate | Performance |
|-----|--------|----------|-------------|
| **Standard Bot** | 34.85% | 52.27% | ✅ Good |
| **XGBoost Bot** | 228.64% | 100.00% | ✅ Excellent |

> **Technology Stack**: XGBoost + TA-Lib + 26 Technical Features + Scikit-learn

## 🏗️ System Architecture

```
CSV Data → Technical Indicators → Trading Strategy
                                         ↓
                                   Backtesting
                                         ↓
                              Performance Analysis
                                         ↓
                              Visualization (Charts)
```

### Standard Bot - Traditional Strategy

**Technical Indicators:**
- **EMA (Exponential Moving Average)**: Short (9) & Long (20)
- **RSI (Relative Strength Index)**: 15 periods
- **Thresholds**: Oversold < 30, Overbought > 70

**Trading Signals:**
- **Buy**: RSI < 30 OR Golden Cross (Short EMA crosses above Long EMA)
- **Sell**: RSI > 70 OR Death Cross (Short EMA crosses below Long EMA)

### XGBoost Bot - Machine Learning

**AI Technology:**
- **XGBoost**: Extreme Gradient Boosting
- **26 Features**: Combined technical indicators
- **3-Class Classification**: Down (0), Neutral (1), Up (2)

**Advanced Indicator Set:**

**Momentum:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)

**Volatility:**
- Bollinger Bands (upper, middle, lower)
- ATR (Average True Range)
- Standard Deviation (14-day rolling)

**Trend:**
- EMA (9 & 21 periods)
- Price Change & EMA Difference Percentage

**Multi-Timeframe:**
- Returns: 3-day, 7-day, 14-day
- Volume Changes: 3-day, 7-day, 14-day

**ML Pipeline:**
```python
# Hyperparameters
{
    'objective': 'multi:softprob',
    'num_class': 3,
    'max_depth': 5,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0.1
}
```

**Trading Logic:**
- **Buy**: Class = 2 with confidence > 60%
- **Sell**: Class = 0 with confidence > 60%
- **Hold**: All other cases

## 📁 System Requirements

| Component | Version/Requirement |
|-----------|-------------------|
| **Python** | 3.8+ |
| **TA-Lib** | Separate installation (see guide) |
| **RAM** | Minimum 4GB |
| **OS** | Windows / macOS / Linux |
| **Data** | CSV with OHLCV format |

## 🔧 Installation & Setup

### Step 1: Install TA-Lib

**Windows:**
```bash
# Download wheel file from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.xx‑cpxx‑cpxx‑win_amd64.whl
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

### Step 2: Install Python Dependencies

```bash
pip install pandas numpy talib xgboost scikit-learn matplotlib
```

### Step 3: Prepare Data

Create CSV file with this format (UTF-8 encoding):

```csv
time,open,high,low,close,volume
2024-01-01 00:00:00,42000.5,42500.0,41800.0,42300.0,1250000
2024-01-01 01:00:00,42300.0,42800.0,42100.0,42600.0,1180000
```

### Step 4: Configure Bot

**Standard Bot:**
```python
bot = CryptoTradingBot(
    csv_file_path='data.csv',
    short_ema=9,
    long_ema=20,
    rsi_period=15,
    rsi_oversold=30,
    rsi_overbought=70,
    initial_balance=10000
)
```

**XGBoost Bot:**
```python
xgbot = XboostCryptoTradingBot(
    csv_file_path='data.csv',
    short_ema=9,
    long_ema=21,
    rsi_period=14,
    rsi_oversold=30,
    rsi_overbought=70,
    initial_balance=10000
)
```

### Step 5: Run Bot

```bash
python trading_bot.py
```

**First Run (XGBoost Bot):**
- Training model (~2-5 minutes depending on dataset)
- Creating features from 26 indicators
- Validating model performance

**Subsequent Runs:**
- Load pre-trained model
- Fast backtesting execution

## 📈 Console Interface

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                    CRYPTO TRADING BOT - BACKTEST SYSTEM                              ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Loading data from data.csv...
✓ Loaded 8760 rows from 2024-01-01 to 2024-12-31

[STANDARD BOT]
Calculating indicators...
✓ RSI, EMA calculated
Running backtest...
✓ Completed 44 trades

[XGBOOST BOT]
Training ML model...
✓ Split into 8000 training samples
✓ Validation accuracy: 85.3%
Running backtest...
✓ Completed 15 trades

══════════════════════════════════════════════════════════════════════════════════════════
                            BACKTEST RESULTS
══════════════════════════════════════════════════════════════════════════════════════════
Standard Bot:
├─ Total Trades: 44
├─ Win Rate: 52.27%
├─ Profit: 34.85%
├─ Max Drawdown: 18.32%
├─ Initial Balance: $10,000.00
└─ Final Balance: $13,485.00

XGBoost Bot:
├─ Total Trades: 15
├─ Win Rate: 100.00%
├─ Profit: 228.64%
├─ Initial Balance: $10,000.00
└─ Final Balance: $32,864.00

Generating charts...
✓ Charts saved to results/
══════════════════════════════════════════════════════════════════════════════════════════
```

## 🔄 Updating Data

When you need to add new data:

1. **Add** new data to CSV file
2. **Re-run** bot → automatically processes new data
3. **XGBoost Bot**: Automatically retrains model if needed

```bash
# Backup old data
cp data.csv data_backup.csv

# Append new data
cat new_data.csv >> data.csv
```

## ⚙️ Configuration Options

### Performance Tuning

| Setting | Default | Purpose | Recommendation |
|---------|---------|---------|----------------|
| `short_ema` | 9 | Short EMA period | 5-15 |
| `long_ema` | 20-21 | Long EMA period | 15-50 |
| `rsi_period` | 14-15 | RSI calculation period | 10-20 |
| `rsi_oversold` | 30 | Oversold threshold | 20-35 |
| `rsi_overbought` | 70 | Overbought threshold | 65-80 |

### ML Model Tuning

**Increase Accuracy:**
```python
params = {
    'max_depth': 7,           # More complex model
    'eta': 0.03,              # Lower learning rate
    'num_boost_round': 200    # More iterations
}
```

**Increase Speed:**
```python
params = {
    'max_depth': 3,           # Simpler model
    'eta': 0.1,               # Higher learning rate
    'num_boost_round': 50     # Fewer iterations
}
```

### Speed vs Accuracy Trade-offs

**Prioritize Speed:**
```python
# Reduce features
features = ['rsi', 'ema_diff', 'macd', 'bb_width']

# Reduce training data
train_size = 0.6  # Use only 60% of data
```

**Prioritize Accuracy:**
```python
# Use all 26 features
# Increase training set size
train_size = 0.8

# Cross-validation
cv_folds = 5
```

## 🚀 Advanced Features

### Compare Multiple Strategies

```python
# Run both bots and compare
strategies = {
    'Standard': CryptoTradingBot(...),
    'XGBoost': XboostCryptoTradingBot(...),
}

results = {}
for name, bot in strategies.items():
    df, trades, performance = bot.run()
    results[name] = performance

# Print comparison table
compare_strategies(results)
```

### Export Results

```python
# After running bot
import pandas as pd

# Save trades
trades_df = pd.DataFrame(trades)
trades_df.to_csv('trades_history.csv', index=False)

# Save performance metrics
performance_df = pd.DataFrame([performance])
performance_df.to_csv('performance.csv', index=False)
```

### Hyperparameter Optimization

```python
from itertools import product

# Grid search
ema_short = [5, 9, 12]
ema_long = [20, 26, 50]
rsi_period = [10, 14, 20]

best_profit = 0
best_params = {}

for s, l, r in product(ema_short, ema_long, rsi_period):
    bot = CryptoTradingBot(
        csv_file_path='data.csv',
        short_ema=s,
        long_ema=l,
        rsi_period=r
    )
    _, _, perf = bot.run()
    
    if perf['profit_pct'] > best_profit:
        best_profit = perf['profit_pct']
        best_params = {'short_ema': s, 'long_ema': l, 'rsi_period': r}

print(f"Best params: {best_params}, Profit: {best_profit}%")
```

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| **TA-Lib import error** | Reinstall TA-Lib following Step 1 instructions |
| **Out of memory** | Reduce dataset size or increase RAM |
| **Model not converging** | Adjust learning rate and max_depth |
| **Too many trades** | Increase oversold/overbought thresholds |
| **Too few trades** | Lower thresholds or add trading conditions |
| **Poor results** | Try adjusting parameters or add more data |

## 📊 Understanding Results

### Key Metrics

**Total Trades**: Number of executed trades
- Too few (<10): Strategy too conservative
- Too many (>100): Possible overtrading

**Win Rate**: Percentage of profitable trades
- <50%: Strategy needs improvement
- 50-70%: Good for standard bot
- >80%: Possible overfitting (with ML bot)

**Profit %**: Total return
- Positive: Profitable strategy
- Negative: Loss-making strategy
- Compare with Buy & Hold strategy

**Max Drawdown**: Largest peak-to-trough decline
- <20%: Low risk
- 20-40%: Medium risk
- >40%: High risk

## 📝 Important Notes

> **DISCLAIMER**: This is a research and educational project. DO NOT use directly for live trading without thorough testing.

⚠️ **Overfitting Risk:**
- XGBoost Bot's 100% win rate may indicate overfitting
- Needs verification on out-of-sample data
- Real performance typically lower than backtest

⚠️ **Trading Risks:**
- Cryptocurrency is extremely volatile
- Backtest doesn't include trading fees, slippage
- No guarantee of future profits
- Can result in total capital loss

⚠️ **Recommendations:**
- Paper trade before using real money
- Start with small capital
- Always use stop-loss
- Diversify portfolio
- Monitor and update model regularly

## 🎓 How It Works

### Standard Bot - Rule-Based Trading

1. **Calculate indicators**: EMA and RSI from closing prices
2. **Detect signals**: Compare with thresholds and crossovers
3. **Execute trades**: Buy/sell based on signals
4. **Capital management**: All-in when buying, all-out when selling

### XGBoost Bot - Machine Learning Trading

1. **Feature Engineering**: Create 26 features from OHLCV
2. **Labeling**: Classify future trend (24 periods ahead)
3. **Training**: XGBoost learns patterns from historical data
4. **Prediction**: Forecast trend with confidence score
5. **Trading**: Only trade when confidence > 60%

---

**Author**: 
Trading Bot Research - Contact: mduc11011@gmail.com
