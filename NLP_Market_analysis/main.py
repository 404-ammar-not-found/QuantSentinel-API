import matplotlib.pyplot as plt
from sentiment import build_sentiment_series
from data import load_prices, aggregate_weekly
from backtest import tune_parameters_weekly
from config import TRADE_FREQ, TICKER

def main():
    sentiment = build_sentiment_series()
    prices = load_prices()
    weekly_df = aggregate_weekly(sentiment, prices, freq=TRADE_FREQ)
    results, best_params, best_sharpe = tune_parameters_weekly(weekly_df)

    corr = results["Sentiment"].corr(results["Return_t+1"])
    total_return = results["Cumulative"].iloc[-1]

    print("\nBACKTEST RESULTS (AAPL, Weekly, Tuned with FinBERT)")
    print("----------------------------------------")
    print(f"Best Parameters: {best_params}")
    print(f"Sharpe Ratio: {best_sharpe:.2f}")
    print(f"Total Return: {total_return:.2f}x")
    print(f"Correlation (Sentiment → Return): {corr:.4f}")

    plt.figure(figsize=(10,5))
    plt.plot(results.index, results["Cumulative"], label="Sentiment Strategy")
    plt.plot(results.index, (1 + results["Return_t+1"]).cumprod(), label="Buy & Hold")
    plt.legend()
    plt.title(f"{TICKER} Sentiment Backtest ({TRADE_FREQ}, Tuned with FinBERT)")
    plt.show()

if __name__ == "__main__":
    main()
