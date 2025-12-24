import numpy as np

def backtest_weekly_tuned(df, window=1, pos_thresh=0.05, neg_thresh=-0.05): 
    """
    Docstring for backtest_weekly_tuned
    
    :param df: DataFrame containing weekly sentiment and returns data
    :param window: number of weeks for rolling sentiment average
    :param pos_thresh: positive threshold for sentiment
    :param neg_thresh: negative threshold for sentiment

    :return: DataFrame containing backtest results
    """

    df = df.copy()
    df['SentimentRoll'] = df['Sentiment'].rolling(window).mean()
    df['Position'] = 0
    df.loc[df['SentimentRoll'] > pos_thresh, 'Position'] = 1
    df.loc[df['SentimentRoll'] < neg_thresh, 'Position'] = -1
    df['StrategyReturn'] = df['Position'] * df['Return_t+1']
    df['Cumulative'] = (1 + df['StrategyReturn']).cumprod()
    return df

def tune_parameters_weekly(weekly_df):
    """
    Docstring for tune_parameters_weekly
    
    :param weekly_df: DataFrame containing weekly sentiment and returns data
    
    :return: Tuple containing the best DataFrame, best parameters, and best Sharpe ratio
    """

    best_sharpe = -np.inf
    best_params = None
    best_df = None

    windows = list(range(1, 11)) 
    pos_threshs = [round(x * 0.1, 2) for x in range(0, 11)]  
    neg_threshs = [round(-x * 0.1, 2) for x in range(10, -1, -1)]  

    for w in windows:
        for pt in pos_threshs:
            for nt in neg_threshs:
                df = backtest_weekly_tuned(weekly_df, window=w, pos_thresh=pt, neg_thresh=nt)
                if df['StrategyReturn'].std() == 0:
                    continue
                sharpe = (df['StrategyReturn'].mean() / df['StrategyReturn'].std()) * np.sqrt(52)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = (w, pt, nt)
                    best_df = df
    return best_df, best_params, best_sharpe
