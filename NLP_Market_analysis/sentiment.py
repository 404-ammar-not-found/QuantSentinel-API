import feedparser
import numpy as np
import time
from urllib.parse import quote
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

from config import COMPANY, ARTICLES_PER_DAY, SLEEP, START_DATE, END_DATE

# Load FinBERT model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Maps for labels and scores
label_map = {0: "neutral", 1: "positive", 2: "negative"}
score_map = {"positive": 1, "neutral": 0, "negative": -1}

def finbert_score(text):
    """
    Docstring for finbert_score
    
    :param text: string containing the text to be analyzed

    :return: sentiment score as integer (1 for positive, 0 for neutral, -1 for negative)
    """

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
    label_idx = np.argmax(probs)
    label = label_map[label_idx]
    return score_map[label]

def fetch_daily_sentiment(date):
    """
    Docstring for fetch_daily_sentiment
    
    :param date: datetime object representing the date for which to fetch sentiment
    """

    query = f"{COMPANY} stock"
    rss = f"https://news.google.com/rss/search?q={quote(query)}+when:{date.strftime('%Y-%m-%d')}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss)
    scores = [finbert_score(entry.title) for entry in feed.entries[:ARTICLES_PER_DAY]]
    return np.mean(scores) if scores else 0

def build_sentiment_series():
    """
    Docstring for build_sentiment_series
    """

    dates = pd.date_range(START_DATE, END_DATE, freq="B")
    data = []
    for d in dates:
        print(f"Fetching sentiment for {d.date()}")
        score = fetch_daily_sentiment(d)
        data.append({"Date": d, "Sentiment": score})
        time.sleep(SLEEP)
    df = pd.DataFrame(data)
    df.set_index("Date", inplace=True)
    return df
