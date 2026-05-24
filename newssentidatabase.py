import os
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ==================================
# 1. CONFIGURATION
# ==================================
START = "2002-07-01"
END = "2025-01-09"

# Ensure this path is where the file actually is on your computer
NEWS_FILE = r"C:\Users\uditr\Downloads\archive\NifSent\final_news_sentiment_analysis.csv"
OUTPUT_FILE = r"C:\Users\uditr\Allproject\newssenttimentDataset\daily_news_sentiment_only.csv"

# ==================================
# 2. INITIALIZE SENTIMENT MODEL
# ==================================
print("Initializing NLP Sentiment Model...")
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

# Financial vocabulary weightings
financial_lexicon = {
    'surge': 2.0, 'soar': 2.0, 'skyrocket': 3.0, 'profit': 2.0, 'beat': 2.0,
    'upgrade': 2.0, 'outperform': 2.0, 'rally': 2.0, 'dividend': 1.5,
    'fraud': -3.0, 'crash': -3.0, 'bankrupt': -3.0, 'scam': -3.0,
    'indict': -3.0, 'loss': -2.0, 'lawsuit': -2.0, 'plummet': -2.5,
    'plunge': -2.5, 'downgrade': -2.0, 'drop': -1.5, 'fall': -1.5, 'default': -3.0
}
sia.lexicon.update(financial_lexicon)


def calculate_ml_sentiment(text):
    if pd.isna(text):
        return 0.0
    scores = sia.polarity_scores(str(text))
    # Scale from -1.0 to 1.0 out to -5.0 to 5.0
    return round(scores['compound'] * 5.0, 1)


# ==================================
# 3. LOAD & DEDUPLICATE NEWS
# ==================================
print("Loading and processing news...")
news = pd.read_csv(NEWS_FILE)

# Standardize column names based on your requirements
if "Symbol" in news.columns:
    news = news.rename(columns={"Publish Date": "Date", "Headline": "Article", "Symbol": "Stock"})
else:
    news = news.rename(columns={"Publish Date": "Date", "Headline": "Article"})

# Clean the dates to purely YYYY-MM-DD
news["Date"] = pd.to_datetime(news["Date"], errors="coerce").dt.date
news = news.dropna(subset=["Date"])

# DEDUPLICATE: Drop any repeated articles on the exact same date
news = news.drop_duplicates(subset=["Date", "Article"], keep="first")

# Apply sentiment scoring (-5 to 5)
news["NewsSentiment"] = news["Article"].apply(calculate_ml_sentiment)

# Keep only the requested columns
if "Stock" not in news.columns:
    news["Stock"] = ""

news = news[["Date", "Stock", "Article", "NewsSentiment"]]

# ==================================
# 4. PAD MISSING DATES
# ==================================
print("Padding missing dates with blanks and zeros...")

# Create a master dataframe with every single calendar day from START to END
all_dates = pd.date_range(start=START, end=END, freq='D').date
date_df = pd.DataFrame({"Date": all_dates})

# Merge our news into the master calendar
final = pd.merge(date_df, news, on="Date", how="left")

# Rule: If no news exists for that date, leave Stock/Article blank, and Sentiment as 0
final["Stock"] = final["Stock"].fillna("")
final["Article"] = final["Article"].fillna("")
final["NewsSentiment"] = final["NewsSentiment"].fillna(0.0)

# ==================================
# 5. SAVE
# ==================================
print("Saving file...")
final.to_csv(OUTPUT_FILE, index=False)

print(f"\nDone! File successfully saved to:\n{OUTPUT_FILE}")