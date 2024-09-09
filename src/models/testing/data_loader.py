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
            # Join intraday data with sentiment data
            query = f"""
            SELECT i.*, s.sentiment_score, s.sentiment_confidence
            FROM intraday_{ticker} i
            LEFT JOIN stock_sentiment s ON i.date = s.date AND s.ticker = '{ticker}'
            """
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
