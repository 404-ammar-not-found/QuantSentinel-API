import pandas as pd
from config import PRICE_FILE

def load_prices():
    """
    Docstring for load_prices

    :return: DataFrame containing price data with Date as index
    """
    df = pd.read_csv(PRICE_FILE, parse_dates=["Price"])
    df = df.rename(columns={"Price": "Date"})
    df.set_index("Date", inplace=True)
    df = df[["Close", "Return_t+1"]]
    return df

def aggregate_weekly(sentiment, prices, freq='W-FRI'):
    """
    Docstring for aggregate_weekly
    
    :param sentiment: Dataframe or Series containing daily sentiment data
    :param prices: pandas DataFrame or Series containing daily price data
    :param freq: frequency for resampling (default is 'W-FRI' for weekly on Fridays)
    
    :return: DataFrame containing weekly aggregated sentiment and returns data
    """
    if isinstance(sentiment, pd.DataFrame):
        sentiment = sentiment.iloc[:, 0]
    if isinstance(prices, pd.DataFrame):
        prices_series = prices['Close']
    else:
        prices_series = prices
    sentiment_weekly = sentiment.resample(freq).mean()
    prices_weekly = prices_series.resample(freq).last()
    returns_weekly = prices_weekly.pct_change().shift(-1)
    df = pd.DataFrame({
        'Sentiment': sentiment_weekly,
        'Return_t+1': returns_weekly
    }).dropna()
    return df
