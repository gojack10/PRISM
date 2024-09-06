import pandas as pd
import os
from sqlalchemy import create_engine, text
import numpy as np
import yaml

def get_db_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    db_path = os.path.join(project_root, 'data', 'processed', 'market_data.db')
    print(f"Attempting to access database at: {db_path}")
    return db_path

def load_tickers():
    with open('PRISM/config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config['tickers']

def load_data():
    db_path = get_db_path()
    print(f"Database path: {db_path}")
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return None, None

    try:
        engine = create_engine(f'sqlite:///{db_path}')
        print("Engine created successfully")
        
        # Get list of tables in the database
        with engine.connect() as connection:
            print("Connected to database")
            result = connection.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
            tables = [row[0] for row in result]
        
        print("Tables in the database:")
        for table in tables:
            print(f"- {table}")
        
        tickers = load_tickers()
        print(f"Tickers loaded: {tickers}")
        
        # query data for all tickers
        data_frames = []
        for ticker in tickers:
            query = f"SELECT * FROM intraday_{ticker}"
            print(f"Executing query: {query}")
            df = pd.read_sql(query, engine)
            data_frames.append(df)
        
        data = pd.concat(data_frames, ignore_index=True)

        print("Query executed successfully")
        print("Columns in the DataFrame:")
        print(data.columns)

        # preprocess the data
        data['date'] = pd.to_datetime(data['date'])
        data['date'] = data['date'].astype(int) // 10**9  # convert to unix timestamp

        # ensure all columns are numeric
        for col in data.columns:
            if col not in ['ticker', 'date'] and data[col].dtype == 'object':
                try:
                    data[col] = pd.to_numeric(data[col])
                except ValueError:
                    print(f"warning: could not convert column '{col}' to numeric. dropping this column.")
                    data = data.drop(col, axis=1)

        # set multi-index and sort
        data = data.set_index(['ticker', 'date']).sort_index()

        print("\nProcessed DataFrame info:")
        print(data.info())

        print("\nFirst few rows of processed data:")
        print(data.head())

        # Reset the index to make 'date' and 'ticker' regular columns
        data = data.reset_index()

        # Convert date to numeric if it's not already
        if pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = data['date'].astype(int) / 10**9
        elif not pd.api.types.is_numeric_dtype(data['date']):
            # If date is neither datetime nor numeric, try to convert to datetime first
            data['date'] = pd.to_datetime(data['date'], unit='s')
            data['date'] = data['date'].astype(int) / 10**9

        print("\nDate column type after conversion:", data['date'].dtype)
        print("First few values of date column:", data['date'].head())

        # Function to calculate RSI
        def calculate_rsi(data, window=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        # Function to calculate MACD
        def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
            short_ema = data.ewm(span=short_window, adjust=False).mean()
            long_ema = data.ewm(span=long_window, adjust=False).mean()
            macd = short_ema - long_ema
            signal = macd.ewm(span=signal_window, adjust=False).mean()
            return macd, signal

        # Function to calculate ATR
        def calculate_atr(high, low, close, window=14):
            tr = pd.concat([high - low, 
                            abs(high - close.shift()), 
                            abs(low - close.shift())], axis=1).max(axis=1)
            return tr.rolling(window=window).mean()

        # Function to calculate Bollinger Bands
        def calculate_bbands(data, window=20, num_std=2):
            rolling_mean = data.rolling(window=window).mean()
            rolling_std = data.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            return upper_band, rolling_mean, lower_band

        # Function to calculate On-Balance Volume (OBV)
        def calculate_obv(close, volume):
            return (np.sign(close.diff()) * volume).cumsum()

        # Function to calculate Commodity Channel Index (CCI)
        def calculate_cci(high, low, close, window=20):
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(window=window).mean()
            mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
            cci = (tp - sma_tp) / (0.015 * mad)
            return cci

        # Group by ticker and apply feature engineering
        data = data.groupby('ticker').apply(lambda x: x.assign(
            sma_20=x['close'].rolling(window=20).mean(),
            sma_50=x['close'].rolling(window=50).mean(),
            rsi_14=calculate_rsi(x['close']),
            pct_change=x['close'].pct_change(),
            volume_pct_change=x['volume'].pct_change(),
            day_of_week=pd.to_datetime(x['date'], unit='s').dt.dayofweek,
            month=pd.to_datetime(x['date'], unit='s').dt.month,
            obv=calculate_obv(x['close'], x['volume']),
            cci=calculate_cci(x['high'], x['low'], x['close'])
        )).reset_index(drop=True)

        # Calculate MACD for each ticker
        for ticker in data['ticker'].unique():
            macd, signal = calculate_macd(data.loc[data['ticker'] == ticker, 'close'])
            data.loc[data['ticker'] == ticker, 'macd'] = macd
            data.loc[data['ticker'] == ticker, 'macd_signal'] = signal

        # Calculate ATR for each ticker
        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker]
            data.loc[data['ticker'] == ticker, 'atr'] = calculate_atr(ticker_data['high'], ticker_data['low'], ticker_data['close'])

        # Calculate Bollinger Bands for each ticker
        for ticker in data['ticker'].unique():
            upper, middle, lower = calculate_bbands(data.loc[data['ticker'] == ticker, 'close'])
            data.loc[data['ticker'] == ticker, 'bb_upper'] = upper
            data.loc[data['ticker'] == ticker, 'bb_middle'] = middle
            data.loc[data['ticker'] == ticker, 'bb_lower'] = lower

        # Handle missing data
        data = data.dropna().reset_index(drop=True)

        print("\nDataFrame structure after feature engineering:")
        print(data.info())
        print("\nFirst few rows of processed data:")
        print(data.head())

        return data, None

    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error args: {e.args}")
        return None, None

def main():
    data, _ = load_data()
    if data is not None:
        print("Data loaded successfully.")
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()
