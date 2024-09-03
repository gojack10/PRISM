import pandas as pd
import os
from sqlalchemy import create_engine
import numpy as np

def get_db_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    return os.path.join(project_root, 'data', 'processed', 'market_data.db')

def load_data():
    db_path = get_db_path()
    engine = create_engine(f'sqlite:///{db_path}')

    query = """
    SELECT * FROM intraday_aapl
    JOIN indicator_sma_aapl ON intraday_aapl.date = indicator_sma_aapl.date
    JOIN indicator_macd_aapl ON intraday_aapl.date = indicator_macd_aapl.date
    JOIN indicator_rsi_aapl ON intraday_aapl.date = indicator_rsi_aapl.date
    JOIN indicator_bbands_aapl ON intraday_aapl.date = indicator_bbands_aapl.date
    JOIN indicator_obv_aapl ON intraday_aapl.date = indicator_obv_aapl.date
    JOIN indicator_cci_aapl ON intraday_aapl.date = indicator_cci_aapl.date
    """

    data = pd.read_sql(query, engine)
    data = data.loc[:,~data.columns.duplicated()]

    print("Columns in the DataFrame:")
    print(data.columns)

    y = data['close']
    X = data.drop(columns=['ticker', 'close'])

    X['date'] = pd.to_datetime(X['date']).astype(int) / 10**9
    X = X.select_dtypes(include=[np.number])

    print("\nFeatures used for prediction:")
    print(X.columns)

    return X, y
