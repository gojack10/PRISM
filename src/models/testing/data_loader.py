import pandas as pd
import os
from sqlalchemy import create_engine
import numpy as np
import yaml

def get_db_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    return os.path.join(project_root, 'data', 'processed', 'market_data.db')

def load_tickers():
    with open('PRISM/config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config['tickers']

def load_data():
    db_path = get_db_path()
    engine = create_engine(f'sqlite:///{db_path}')

    tickers = load_tickers()
    
    query = f"""
    SELECT * FROM intraday_{tickers[0]}
    JOIN indicator_sma_{tickers[0]} ON intraday_{tickers[0]}.date = indicator_sma_{tickers[0]}.date
    JOIN indicator_macd_{tickers[0]} ON intraday_{tickers[0]}.date = indicator_macd_{tickers[0]}.date
    JOIN indicator_rsi_{tickers[0]} ON intraday_{tickers[0]}.date = indicator_rsi_{tickers[0]}.date
    JOIN indicator_bbands_{tickers[0]} ON intraday_{tickers[0]}.date = indicator_bbands_{tickers[0]}.date
    JOIN indicator_obv_{tickers[0]} ON intraday_{tickers[0]}.date = indicator_obv_{tickers[0]}.date
    JOIN indicator_cci_{tickers[0]} ON intraday_{tickers[0]}.date = indicator_cci_{tickers[0]}.date
    JOIN sentiment_{tickers[0]} ON intraday_{tickers[0]}.date = sentiment_{tickers[0]}.date
    """

    data = pd.read_sql(query, engine)
    data = data.loc[:,~data.columns.duplicated()]

    print("Columns in the DataFrame:")
    print(data.columns)

    y = data['close']
    X = data.drop(columns=['ticker', 'close'])

    X['date'] = pd.to_datetime(X['date']).astype(int) / 10**9
    X = X.select_dtypes(include=[np.number])

    # Ensure sentiment columns are included
    sentiment_columns = ['sentiment_score', 'sentiment_confidence']
    for col in sentiment_columns:
        if col not in X.columns:
            print(f"Warning: {col} not found in the data. Please ensure it's in the sentiment table.")

    print("\nFeatures used for prediction:")
    print(X.columns)

    return X, y

def main():
    print("Starting data loading process...")
    
    print("\nLoading tickers:")
    tickers = load_tickers()
    print(f"Tickers loaded: {tickers}")
    
    print("\nLoading data...")
    X, y = load_data()
    
    print("\nData loading complete.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    print("\nFirst few rows of X:")
    print(X.head())
    
    print("\nFirst few values of y:")
    print(y.head())
    
    print("\nBasic statistics of y:")
    print(y.describe())
    
    print("\nSentiment data summary:")
    sentiment_columns = ['sentiment_score', 'sentiment_confidence']
    for col in sentiment_columns:
        if col in X.columns:
            print(f"\n{col} statistics:")
            print(X[col].describe())
        else:
            print(f"\n{col} not found in the data.")

if __name__ == "__main__":
    main()
