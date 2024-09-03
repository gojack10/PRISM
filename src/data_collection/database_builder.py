import os
import sqlite3
import pandas as pd
import json
from datetime import datetime
import yaml
import logging
from typing import List, Dict, Any

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

# define indicators
INDICATORS = ['SMA', 'MACD', 'RSI', 'BBANDS', 'OBV', 'CCI']

def connect_to_database() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def get_historical_table_name(ticker: str) -> str:
    return f"intraday_{ticker.lower().replace('.', '_')}"

def get_indicator_table_name(ticker: str, indicator: str) -> str:
    return f"indicator_{indicator.lower()}_{ticker.lower().replace('.', '_')}"

def create_tables(conn: sqlite3.Connection):
    cursor = conn.cursor()

    # create market_drivers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sentiment_market_drivers (
        date TEXT,
        driver TEXT,
        impact_score REAL,
        PRIMARY KEY (date, driver)
    )
    ''')

    # create stock_comparison table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sentiment_stock_comparison (
        date TEXT PRIMARY KEY,
        sentiment_difference REAL,
        topic_overlap_percentage REAL,
        relationship_analysis TEXT
    )
    ''')

    # create stock_sentiment table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sentiment_stock_sentiment (
        date TEXT,
        ticker TEXT,
        sentiment_score REAL,
        sentiment_confidence REAL,
        key_topics TEXT,
        sentiment_change TEXT,
        financial_metrics TEXT,
        short_term_outlook TEXT,
        long_term_outlook TEXT,
        PRIMARY KEY (date, ticker)
    )
    ''')

    # create company_overview table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS company_overview (
        ticker TEXT,
        date TEXT,
        PERatio REAL,
        PEGRatio REAL,
        PriceToBookRatio REAL,
        OperatingMarginTTM REAL,
        ReturnOnEquityTTM REAL,
        QuarterlyEarningsGrowthYOY REAL,
        QuarterlyRevenueGrowthYOY REAL,
        DividendYield REAL,
        AnalystTargetPrice REAL,
        "50DayMovingAverage" REAL,
        "200DayMovingAverage" REAL,
        Beta REAL,
        MarketCapitalization REAL,
        PRIMARY KEY (ticker, date)
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

def create_indicator_table(conn: sqlite3.Connection, ticker: str, indicator: str):
    cursor = conn.cursor()
    table_name = get_indicator_table_name(ticker, indicator)
    
    if indicator == 'BBANDS':
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            date TEXT,
            ticker TEXT,
            real_upper_band REAL,
            real_middle_band REAL,
            real_lower_band REAL,
            PRIMARY KEY (date, ticker)
        )
        ''')
    elif indicator == 'MACD':
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            date TEXT,
            ticker TEXT,
            macd REAL,
            macd_hist REAL,
            macd_signal REAL,
            PRIMARY KEY (date, ticker)
        )
        ''')
    else:
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            date TEXT,
            ticker TEXT,
            value REAL,
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

def insert_indicator_data(conn: sqlite3.Connection, ticker: str, indicator: str, data: Dict):
    cursor = conn.cursor()
    table_name = get_indicator_table_name(ticker, indicator)
    
    for date, values in data['Technical Analysis'].items():
        if indicator == 'BBANDS':
            cursor.execute(f'''
            INSERT OR REPLACE INTO {table_name}
            (date, ticker, real_upper_band, real_middle_band, real_lower_band)
            VALUES (?, ?, ?, ?, ?)
            ''', (date, ticker, values['Real Upper Band'], values['Real Middle Band'], values['Real Lower Band']))
        elif indicator == 'MACD':
            cursor.execute(f'''
            INSERT OR REPLACE INTO {table_name}
            (date, ticker, macd, macd_hist, macd_signal)
            VALUES (?, ?, ?, ?, ?)
            ''', (date, ticker, values['MACD'], values['MACD_Hist'], values['MACD_Signal']))
        else:
            cursor.execute(f'''
            INSERT OR REPLACE INTO {table_name}
            (date, ticker, value)
            VALUES (?, ?, ?)
            ''', (date, ticker, list(values.values())[0]))
    
    conn.commit()

def insert_sentiment_data(conn: sqlite3.Connection, data: Dict):
    cursor = conn.cursor()
    
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    for ticker, stock_data in data['stocks'].items():
        cursor.execute('''
        INSERT OR REPLACE INTO sentiment_stock_sentiment
        (date, ticker, sentiment_score, sentiment_confidence, key_topics, sentiment_change, 
        financial_metrics, short_term_outlook, long_term_outlook)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date_str, ticker, stock_data['sentiment_score'], 
              stock_data['sentiment_confidence'],
              json.dumps(stock_data['key_topics']),
              json.dumps(stock_data['sentiment_change']),
              json.dumps(stock_data['financial_metrics']),
              json.dumps(stock_data['short_term_outlook']),
              json.dumps(stock_data['long_term_outlook'])))
    
    # insert market drivers
    for driver in data['market_drivers']:
        cursor.execute('''
        INSERT OR REPLACE INTO sentiment_market_drivers (date, driver, impact_score)
        VALUES (?, ?, ?)
        ''', (date_str, driver['driver'], driver['impact_score']))
    
    # insert stock comparison
    cursor.execute('''
    INSERT OR REPLACE INTO sentiment_stock_comparison 
    (date, sentiment_difference, topic_overlap_percentage, relationship_analysis)
    VALUES (?, ?, ?, ?)
    ''', (date_str, data['stock_comparison']['sentiment_difference'], 
          data['stock_comparison']['topic_overlap_percentage'],
          data['stock_comparison']['relationship_analysis']))
    
    conn.commit()

def insert_company_overview_data(conn: sqlite3.Connection, ticker: str, data: Dict):
    cursor = conn.cursor()
    table_name = "company_overview"
    
    cursor.execute(f'''
    INSERT OR REPLACE INTO {table_name}
    (ticker, date, PERatio, PEGRatio, PriceToBookRatio, OperatingMarginTTM, ReturnOnEquityTTM, 
    QuarterlyEarningsGrowthYOY, QuarterlyRevenueGrowthYOY, DividendYield, AnalystTargetPrice, 
    "50DayMovingAverage", "200DayMovingAverage", Beta, MarketCapitalization)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (ticker, datetime.now().strftime('%Y-%m-%d'), data.get('PERatio'), data.get('PEGRatio'), 
          data.get('PriceToBookRatio'), data.get('OperatingMarginTTM'), data.get('ReturnOnEquityTTM'), 
          data.get('QuarterlyEarningsGrowthYOY'), data.get('QuarterlyRevenueGrowthYOY'), data.get('DividendYield'), 
          data.get('AnalystTargetPrice'), data.get('50DayMovingAverage'), data.get('200DayMovingAverage'), 
          data.get('Beta'), data.get('MarketCapitalization')))
    
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

    logging.info(f"Total number of rows across all intraday data tables: {total_rows}")

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

def update_indicator_data(conn: sqlite3.Connection):
    cursor = conn.cursor()
    indicators_dir = os.path.join(RAW_DATA_DIR, 'indicators')
    
    for indicator in INDICATORS:
        historical_dir = os.path.join(indicators_dir, 'historical', indicator)
        daily_update_dir = os.path.join(indicators_dir, 'daily_update', indicator)
        
        for ticker in STOCK_TICKERS:
            create_indicator_table(conn, ticker, indicator)
            table_name = get_indicator_table_name(ticker, indicator)
            
            # check if historical data exists in the database
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            
            if count == 0:
                # load historical data
                historical_file = os.path.join(historical_dir, f"{ticker}_{indicator}_*.json")
                historical_files = [f for f in os.listdir(historical_dir) if f.startswith(f"{ticker}_{indicator}_")]
                
                if historical_files:
                    with open(os.path.join(historical_dir, historical_files[0]), 'r') as f:
                        data = json.load(f)
                    insert_indicator_data(conn, ticker, indicator, data)
                    logging.info(f"Inserted historical {indicator} data for {ticker}")
            
            # load daily update
            daily_update_file = os.path.join(daily_update_dir, f"{ticker}_{indicator}_*.json")
            daily_update_files = [f for f in os.listdir(daily_update_dir) if f.startswith(f"{ticker}_{indicator}_")]
            
            if daily_update_files:
                latest_file = max(daily_update_files)
                date_str = latest_file.split('_')[-1].split('.')[0]
                
                # check if the latest daily update is already in the database
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE date = ?", (date_str,))
                count = cursor.fetchone()[0]
                
                if count == 0:
                    with open(os.path.join(daily_update_dir, latest_file), 'r') as f:
                        data = json.load(f)
                    insert_indicator_data(conn, ticker, indicator, data)
                    logging.info(f"Inserted daily update {indicator} data for {ticker} on {date_str}")
                else:
                    logging.info(f"Daily update {indicator} data for {ticker} on {date_str} already exists in the database")
    
    conn.commit()
    logging.info("Indicator data updated successfully.")

def update_sentiment_data(conn: sqlite3.Connection):
    cursor = conn.cursor()
    sentiment_dir = os.path.join(RAW_DATA_DIR, 'sentiment', 'claude')
    
    logging.info(f"Starting update_sentiment_data function")
    logging.info(f"Sentiment directory: {sentiment_dir}")

    if not os.path.exists(sentiment_dir):
        logging.warning(f"Sentiment directory not found: {sentiment_dir}")
        return

    for file in os.listdir(sentiment_dir):
        if file.endswith('.json'):
            file_path = os.path.join(sentiment_dir, file)
            logging.info(f"Processing sentiment file: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                insert_sentiment_data(conn, data)
                logging.info(f"Processed sentiment data from file: {file}")
            except Exception as e:
                logging.error(f"Error processing sentiment file {file}: {str(e)}")
    
    conn.commit()
    logging.info("Sentiment data updated successfully.")

def update_company_overview_data(conn: sqlite3.Connection):
    overview_dir = os.path.join(RAW_DATA_DIR, 'overview')
    
    logging.info(f"Loading company overview data from {overview_dir}")
    for file in os.listdir(overview_dir):
        if file.endswith('.json'):
            ticker = file.split('_')[0]
            file_path = os.path.join(overview_dir, file)
            logging.info(f"Processing file: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                insert_company_overview_data(conn, ticker, data)
                logging.info(f"Inserted company overview data for {ticker} from {file}")
            except Exception as e:
                logging.error(f"Error processing {file}: {str(e)}")

def rename_existing_tables(conn: sqlite3.Connection):
    cursor = conn.cursor()
    
    # get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for (table_name,) in tables:
        if table_name.startswith('historical_data_daily_'):
            new_name = 'intraday_' + table_name.split('_')[-1]
            cursor.execute(f"ALTER TABLE {table_name} RENAME TO {new_name}")
            logging.info(f"renamed table {table_name} to {new_name}")
    
    conn.commit()
    logging.info("finished renaming existing tables")

def main():
    today = datetime.now().strftime('%Y-%m-%d')
    
    conn = connect_to_database()
    create_tables(conn)
    rename_existing_tables(conn)
    
    try:
        update_historical_data(conn)
        update_indicator_data(conn)
        update_sentiment_data(conn)
        update_company_overview_data(conn)
        logging.info("Database updated successfully.")
    except Exception as e:
        logging.error(f"An error occurred while updating the database: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
