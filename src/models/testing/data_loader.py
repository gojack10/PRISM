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
        
        # Query only the intraday table
        query = f"SELECT * FROM intraday_{tickers[0]}"
        
        print(f"Executing query: {query}")
        
        data = pd.read_sql(query, engine)

        print("Query executed successfully")
        print("Columns in the DataFrame:")
        print(data.columns)

        # Preprocess the data
        data['date'] = pd.to_datetime(data['date'])
        data['date'] = data['date'].astype(int) / 10**9  # Convert to Unix timestamp
        data = data.drop('ticker', axis=1)  # Drop the ticker column as it's constant

        # Ensure all columns are numeric
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_numeric(data[col])
                except ValueError:
                    print(f"Warning: Could not convert column '{col}' to numeric. Dropping this column.")
                    data = data.drop(col, axis=1)

        print("\nProcessed DataFrame info:")
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
