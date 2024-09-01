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

# commented out for debugging
# def is_market_inactive(date: str) -> bool:
#     dt = datetime.strptime(date, '%Y-%m-%d')
#     return dt.weekday() >= 5  # 5 and 6 are saturday and sunday

def connect_to_database() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def create_tables(conn: sqlite3.Connection):
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS historical_data_daily (
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
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_drivers (
        date TEXT,
        driver TEXT,
        impact_score REAL,
        PRIMARY KEY (date, driver)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_comparison (
        date TEXT,
        sentiment_difference REAL,
        topic_overlap_percentage REAL,
        PRIMARY KEY (date)
    )
    ''')
    
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

def update_historical_data(conn: sqlite3.Connection):
    cursor = conn.cursor()
    historical_dir = os.path.join(RAW_DATA_DIR, 'historical')
    past_dir = os.path.join(historical_dir, 'past')
    daily_update_dir = os.path.join(historical_dir, 'daily_update')
    
    logging.info("Starting update_historical_data function")

    cursor.execute("SELECT COUNT(*) FROM historical_data_daily")
    count = cursor.fetchone()[0]
    logging.info(f"Current number of rows in historical_data_daily: {count}")

    if count == 0:
        logging.info(f"Database is empty. Loading historical data from {past_dir}")
        for file in os.listdir(past_dir):
            if file.endswith('.csv'):
                ticker = file.split('_')[0]
                file_path = os.path.join(past_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    df = df.rename(columns={
                        'Date': 'date',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    if 'Adj Close' in df.columns:
                        df = df.drop('Adj Close', axis=1)
                    df['date'] = pd.to_datetime(df['date'])
                    
                    for _, row in df.iterrows():
                        cursor.execute('''
                        INSERT OR IGNORE INTO historical_data_daily
                        (date, ticker, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (row['date'].strftime('%Y-%m-%d'), ticker, row['open'], row['high'], row['low'], row['close'], row['volume']))
                    logging.info(f"Inserted historical data for {ticker}")
                except Exception as e:
                    logging.error(f"Error processing historical file {file}: {str(e)}")
        conn.commit()
        logging.info("Historical data loaded successfully")

    logging.info(f"Loading daily updates from {daily_update_dir}")
    for file in os.listdir(daily_update_dir):
        if file.endswith('.csv'):
            ticker = file.split('_')[0]
            file_path = os.path.join(daily_update_dir, file)
            try:
                df = pd.read_csv(file_path)
                date_str = file.split('_')[-1].split('.')[0]
                
                for _, row in df.iterrows():
                    cursor.execute('''
                    INSERT OR REPLACE INTO historical_data_daily
                    (date, ticker, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (date_str, ticker, row['open'], row['high'], row['low'], row['close'], row['volume']))
                logging.info(f"Inserted/updated data for {ticker} on {date_str}")
            except Exception as e:
                logging.error(f"Error processing daily update file {file}: {str(e)}")
    
    conn.commit()
    logging.info("Daily updates loaded successfully")

    cursor.execute("SELECT COUNT(*) FROM historical_data_daily")
    final_count = cursor.fetchone()[0]
    logging.info(f"Final number of rows in historical_data_daily: {final_count}")

    cursor.execute("SELECT MAX(date) FROM historical_data_daily")
    latest_date = cursor.fetchone()[0]
    logging.info(f"Latest date in the database: {latest_date}")

    logging.info("Finished update_historical_data function")

def update_market_data(conn: sqlite3.Connection):
    cursor = conn.cursor()
    historical_dir = os.path.join(RAW_DATA_DIR, 'historical')
    daily_dir = os.path.join(historical_dir, 'daily')
    sentiment_dir = os.path.join(RAW_DATA_DIR, 'sentiment', 'claude')
    
    for file in os.listdir(daily_dir):
        if file.endswith('.csv'):
            ticker = file.split('_')[0]
            df = pd.read_csv(os.path.join(daily_dir, file))
            
            for _, row in df.iterrows():
                date = row['date']
                
                # get sentiment data
                sentiment_file = f"sentiment_{ticker}_{date}.json"
                sentiment_path = os.path.join(sentiment_dir, sentiment_file)
                
                if os.path.exists(sentiment_path):
                    with open(sentiment_path, 'r') as f:
                        sentiment_data = json.load(f)
                    
                    stock_data = sentiment_data['stocks'][ticker]
                    
                    cursor.execute('''
                    INSERT OR REPLACE INTO market_data
                    (date, ticker, open, high, low, close, volume, 
                    sentiment_score, key_topics, sentiment_change, financial_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (date, ticker, row['open'], row['high'], row['low'], row['close'], row['volume'],
                          stock_data['sentiment_score'], json.dumps(stock_data['key_topics']),
                          stock_data['sentiment_change']['change_percentage'],
                          json.dumps(stock_data['financial_metrics'])))
                else:
                    # if sentiment data is not available, insert only historical data
                    cursor.execute('''
                    INSERT OR REPLACE INTO market_data
                    (date, ticker, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (date, ticker, row['open'], row['high'], row['low'], row['close'], row['volume']))
    
    conn.commit()

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
                insert or replace into market_drivers (date, driver, impact_score)
                values (?, ?, ?)
                ''', (date, driver['driver'], driver['impact_score']))
            
            # update stock_comparison
            cursor.execute('''
            insert or replace into stock_comparison 
            (date, sentiment_difference, topic_overlap_percentage)
            values (?, ?, ?)
            ''', (date, data['stock_comparison']['sentiment_difference'], 
                  data['stock_comparison']['topic_overlap_percentage']))
            
            # update stock_sentiment
            for ticker, stock_data in data['stocks'].items():
                cursor.execute('''
                insert or replace into stock_sentiment
                (date, ticker, sentiment_score, key_topics, sentiment_change, financial_metrics)
                values (?, ?, ?, ?, ?, ?)
                ''', (date, ticker, stock_data['sentiment_score'], 
                      json.dumps(stock_data['key_topics']),
                      stock_data['sentiment_change']['change_percentage'],
                      json.dumps(stock_data['financial_metrics'])))
    
    conn.commit()
    logging.info("sentiment data updated successfully.")

def main():
    today = datetime.now().strftime('%Y-%m-%d')
    
    # commented out for debugging
    # if is_market_inactive(today):
    #     logging.info(f"Market is inactive on {today}. Skipping database update.")
    #     return
    
    conn = connect_to_database()
    create_tables(conn)
    
    try:
        update_historical_data(conn)
        update_sentiment_data(conn)
        logging.info("database updated successfully.")
    except Exception as e:
        logging.error(f"an error occurred while updating the database: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
