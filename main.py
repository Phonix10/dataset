import os
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------
# CONFIG
# ---------------------------------
START = "2002-07-01"
END = "2025-01-09"

INPUT_SENTI_FILE = r"C:\Users\uditr\Allproject\newssenttimentDataset\daily_news_sentiment_only.csv"
OUTPUT_DIR = r"C:\Users\uditr\Allproject\newssenttimentDataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_PATH = os.path.join(OUTPUT_DIR, "NIFTY50_Master_with_Sentiment.csv")

# ---------------------------------
# LOAD & AVERAGE PRE-PROCESSED SENTIMENT
# ---------------------------------
print("Loading pre-processed sentiment file...")
senti_df = pd.read_csv(INPUT_SENTI_FILE)

# Ensure Date is properly formatted
senti_df["Date"] = pd.to_datetime(senti_df["Date"], errors="coerce")
senti_df = senti_df.dropna(subset=["Date"])

print("Averaging sentiment for days with multiple articles...")
# Group by date and average the sentiment score
daily_senti = senti_df.groupby(senti_df["Date"].dt.date)["NewsSentiment"].mean().reset_index()
daily_senti.columns = ["Date", "newsenti"]
daily_senti["Date"] = pd.to_datetime(daily_senti["Date"])

# ---------------------------------
# DOWNLOAD NIFTY 50 & MACROS
# ---------------------------------
print("Downloading NIFTY 50 data...")
nifty = yf.download("^NSEI", start=START, end=END, auto_adjust=True)

if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.get_level_values(0)

nifty.reset_index(inplace=True)
nifty["Date"] = pd.to_datetime(nifty["Date"])

print("Downloading Macro Data (Oil, USDINR, S&P 500)...")
macro_tickers = {
    "CL=F": "OilPrice",
    "INR=X": "USDINR",
    "^GSPC": "SP500"
}

nifty.set_index("Date", inplace=True)

for ticker, col_name in macro_tickers.items():
    macro_data = yf.download(ticker, start=START, end=END, auto_adjust=True)
    if isinstance(macro_data.columns, pd.MultiIndex):
        macro_data.columns = macro_data.columns.get_level_values(0)

    macro_close = macro_data[["Close"]].rename(columns={"Close": col_name})
    nifty = nifty.join(macro_close, how="left")

# Forward-fill missing macro data (handles differing US/India holidays)
nifty[["OilPrice", "USDINR", "SP500"]] = nifty[["OilPrice", "USDINR", "SP500"]].ffill()
nifty.reset_index(inplace=True)

# Merge Sentiment Data
print("Merging sentiment data...")
nifty = nifty.merge(daily_senti, on="Date", how="left")
nifty["newsenti"] = nifty["newsenti"].fillna(0.0)  # Fill missing news days with Neutral (0)

# ---------------------------------
# TECHNICAL INDICATORS
# ---------------------------------
print("Calculating technical indicators...")

close = nifty["Close"]
high = nifty["High"]
low = nifty["Low"]
volume = nifty["Volume"]

nifty["MA20"] = close.rolling(20).mean()
nifty["MA50"] = close.rolling(50).mean()

nifty["EMA20"] = close.ewm(span=20, adjust=False).mean()
nifty["EMA50"] = close.ewm(span=50, adjust=False).mean()

# RSI
delta = close.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
nifty["RSI"] = 100 - (100 / (1 + rs))

# Vectorized OBV
direction = np.sign(delta).fillna(0)
nifty["OBV"] = (direction * volume).cumsum()

# Rolling Fibonacci (252 Trading Days ~ 1 Year)
rolling_high = high.rolling(window=252, min_periods=1).max()
rolling_low = low.rolling(window=252, min_periods=1).min()
diff = rolling_high - rolling_low

nifty["Fib23"] = rolling_high - (diff * 0.236)
nifty["Fib38"] = rolling_high - (diff * 0.382)
nifty["Fib50"] = rolling_high - (diff * 0.500)
nifty["Fib61"] = rolling_high - (diff * 0.618)

# ---------------------------------
# FINAL CLEANUP & SAVE
# ---------------------------------
# Drop rows with NaN values created by moving averages
nifty.dropna(inplace=True)

# Reorder exactly to requested columns
req_columns = [
    "Date", "Close", "High", "Low", "Open", "Volume",
    "MA20", "MA50", "EMA20", "EMA50", "RSI", "OBV",
    "Fib23", "Fib38", "Fib50", "Fib61",
    "OilPrice", "USDINR", "SP500", "newsenti"
]

nifty = nifty[req_columns]

nifty.to_csv(FILE_PATH, index=False)

print("\n✅ DATASET READY")
print(f"File saved successfully at: {FILE_PATH}")
print(f"Total trading days captured: {len(nifty)}")