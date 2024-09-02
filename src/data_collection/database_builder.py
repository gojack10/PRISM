import os
import sqlite3
import pandas as pd
import json
from datetime import datetime
import yaml
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open('PRISM/config/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

STOCK_TICKERS = config['tickers']

# paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
DB_PATH = os.path.join(project_root, 'data', 'processed', 'market_data.db')
RAW_DATA_DIR = os.path.join(project_root, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')

def connect_to_database() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def get_historical_table_name(ticker: str) -> str:
    return f"historical_data_daily_{ticker.lower().replace('.', '_')}"

def create_tables(conn: sqlite3.Connection):
    cursor = conn.cursor()

    # create tickers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tickers (
        ticker TEXT PRIMARY KEY,
        name TEXT,
        sector TEXT,
        industry TEXT
    )
    ''')

    # create market_drivers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_drivers (
        date TEXT,
        driver TEXT,
        impact_score REAL,
        PRIMARY KEY (date, driver)
    )
    ''')

    # create stock_comparison table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_comparison (
        date TEXT PRIMARY KEY,
        sentiment_difference REAL,
        topic_overlap_percentage REAL
    )
    ''')

    # create stock_sentiment table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_sentiment (
        date TEXT,
        ticker TEXT,
        sentiment_score REAL,
        key_topics TEXT,
        sentiment_change REAL,
        financial_metrics TEXT,
        PRIMARY KEY (date, ticker)
    )
    ''')

    conn.commit()

def create_historical_table(conn: sqlite3.Connection, ticker: str):
    cursor = conn.cursor()
    table_name = get_historical_table_name(ticker)
    
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        date TEXT,
        ticker TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        PRIMARY KEY (date, ticker)
    )
    ''')
    
    conn.commit()

def insert_historical_data(conn: sqlite3.Connection, ticker: str, data: List[Dict]):
    cursor = conn.cursor()
    table_name = get_historical_table_name(ticker)
    
    for row in data:
        cursor.execute(f'''
        INSERT OR REPLACE INTO {table_name}
        (date, ticker, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (row['date'], ticker, row['open'], row['high'], row['low'], row['close'], row['volume']))
    
    conn.commit()

def update_historical_data(conn: sqlite3.Connection):
    cursor = conn.cursor()
    intraday_dir = os.path.join(RAW_DATA_DIR, 'intraday')
    past_dir = os.path.join(intraday_dir, 'past')
    daily_update_dir = os.path.join(intraday_dir, 'daily_update')
    
    logging.info(f"Starting update_historical_data function")
    logging.info(f"intraday_dir: {intraday_dir}")
    logging.info(f"past_dir: {past_dir}")
    logging.info(f"daily_update_dir: {daily_update_dir}")

    # load data from 'past' directory
    logging.info(f"Loading historical data from {past_dir}")
    for file in os.listdir(past_dir):
        if file.endswith('.csv'):
            ticker = file.split('_')[0]
            table_name = get_historical_table_name(ticker)
            file_path = os.path.join(past_dir, file)
            logging.info(f"Processing file: {file_path}")
            try:
                df = pd.read_csv(file_path)
                
                # rename columns to match our database structure
                df = df.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # drop 'Adj Close' column if it exists
                if 'Adj Close' in df.columns:
                    df = df.drop('Adj Close', axis=1)
                
                # convert date column to datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # log sample data
                logging.info(f"sample data from {file}:\n{df.head()}")
                
                create_historical_table(conn, ticker)  # Ensure table exists
                
                for _, row in df.iterrows():
                    cursor.execute(f'''
                    INSERT OR IGNORE INTO {table_name}
                    (date, ticker, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (row['date'].strftime('%Y-%m-%d'), ticker, row['open'], row['high'], row['low'], row['close'], row['volume']))
                logging.info(f"Inserted data for {ticker} from {file}")
            except Exception as e:
                logging.error(f"Error processing {file}: {str(e)}")
    
    conn.commit()
    logging.info("Historical data from 'past' directory loaded successfully.")
    
    # load updates from 'daily_update' directory
    logging.info(f"Loading daily updates from {daily_update_dir}")
    for file in os.listdir(daily_update_dir):
        if file.endswith('.csv'):
            ticker = file.split('_')[0]
            table_name = get_historical_table_name(ticker)
            file_path = os.path.join(daily_update_dir, file)
            logging.info(f"Processing file: {file_path}")
            try:
                df = pd.read_csv(file_path)
                logging.info(f"DataFrame shape after reading CSV: {df.shape}")
                logging.info(f"DataFrame columns: {df.columns}")
                logging.info(f"DataFrame dtypes:\n{df.dtypes}")
                logging.info(f"Sample data from {file}:\n{df.to_string()}")
                
                # Extract date from filename
                date_str = file.split('_')[-1].split('.')[0]
                
                create_historical_table(conn, ticker)  # Ensure table exists
                
                for index, row in df.iterrows():
                    logging.info(f"Processing row {index}: {row.to_dict()}")
                    try:
                        cursor.execute(f'''
                        INSERT OR REPLACE INTO {table_name}
                        (date, ticker, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (date_str, ticker, row['open'], row['high'], row['low'], row['close'], row['volume']))
                        logging.info(f"Executed SQL for row {index}")
                    except Exception as e:
                        logging.error(f"Error inserting row {index}: {str(e)}")
                logging.info(f"Finished processing all rows for {ticker} from {file}")
            except Exception as e:
                logging.error(f"Error processing file {file}: {str(e)}")
    
    conn.commit()
    logging.info("Committed changes to database")

    # Update these checks to work with multiple tables
    total_rows = 0
    for ticker in STOCK_TICKERS:
        table_name = get_historical_table_name(ticker)
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        total_rows += count
        logging.info(f"Number of rows in {table_name}: {count}")

    logging.info(f"Total number of rows across all historical data tables: {total_rows}")

    # Check the most recent data
    for ticker in STOCK_TICKERS:
        table_name = get_historical_table_name(ticker)
        cursor.execute(f"SELECT MAX(date) FROM {table_name}")
        latest_date = cursor.fetchone()[0]
        logging.info(f"Latest date for {ticker}: {latest_date}")

        cursor.execute(f"SELECT * FROM {table_name} ORDER BY date DESC LIMIT 5")
        recent_rows = cursor.fetchall()
        logging.info(f"5 most recent rows for {ticker}:")
        for row in recent_rows:
            logging.info(row)

    logging.info("Finished update_historical_data function")

def update_sentiment_data(conn: sqlite3.Connection):
    cursor = conn.cursor()
    sentiment_dir = os.path.join(RAW_DATA_DIR, 'sentiment', 'claude')
    
    for file in os.listdir(sentiment_dir):
        if file.endswith('.json'):
            with open(os.path.join(sentiment_dir, file), 'r') as f:
                data = json.load(f)
            
            # extract date from filename and format it
            date_str = file.split('_')[2].split('.')[0]  # extract date from filename
            date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            
            # update market_drivers
            for driver in data['market_drivers']:
                cursor.execute('''
                INSERT OR REPLACE INTO market_drivers (date, driver, impact_score)
                VALUES (?, ?, ?)
                ''', (date, driver['driver'], driver['impact_score']))
            
            # update stock_comparison
            cursor.execute('''
            INSERT OR REPLACE INTO stock_comparison 
            (date, sentiment_difference, topic_overlap_percentage)
            VALUES (?, ?, ?)
            ''', (date, data['stock_comparison']['sentiment_difference'], 
                  data['stock_comparison']['topic_overlap_percentage']))
            
            # update stock_sentiment
            for ticker, stock_data in data['stocks'].items():
                cursor.execute('''
                INSERT OR REPLACE INTO stock_sentiment
                (date, ticker, sentiment_score, key_topics, sentiment_change, financial_metrics)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (date, ticker, stock_data['sentiment_score'], 
                      json.dumps(stock_data['key_topics']),
                      stock_data['sentiment_change']['change_percentage'],
                      json.dumps(stock_data['financial_metrics'])))
    
    conn.commit()
    logging.info("Sentiment data updated successfully.")

def main():
    today = datetime.now().strftime('%Y-%m-%d')
    
    conn = connect_to_database()
    create_tables(conn)
    
    try:
        update_historical_data(conn)
        update_sentiment_data(conn)
        logging.info("Database updated successfully.")
    except Exception as e:
        logging.error(f"An error occurred while updating the database: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
