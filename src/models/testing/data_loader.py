import os
import logging
import pandas as pd
import random
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def load_environment_variables(dotenv_path):
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        logging.info(f"Loaded .env file from {dotenv_path}")
    else:
        logging.error(f".env file not found at {dotenv_path}")
        raise FileNotFoundError(f".env file not found at {dotenv_path}")

def create_db_engine():
    """
    Create a SQLAlchemy engine using environment variables.
    """
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT', '5432')  # Default to 5432 if not set
    db_name = os.getenv('DB_NAME')

    if not all([db_user, db_password, db_host, db_name]):
        missing = [var for var in ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_NAME'] if not os.getenv(var)]
        logging.error(f"Missing environment variables for: {', '.join(missing)}")
        raise EnvironmentError(f"Missing environment variables for: {', '.join(missing)}")

    try:
        db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(
            db_url,
            pool_size=20,            # Adjust pool size as needed
            max_overflow=0,          # No overflow beyond pool_size
            pool_timeout=30,         # Seconds to wait before giving up on getting a connection
            pool_recycle=1800        # Recycle connections after 30 minutes
        )
        logging.info("Database engine created successfully.")
        logging.info(f"Engine type: {type(engine)}")  # Added logging
        return engine
    except Exception as e:
        logging.error(f"Failed to create database engine: {e}")
        raise

def get_session(engine):
    """
    Create a new database session.
    """
    Session = scoped_session(sessionmaker(bind=engine))
    logging.info("Session factory created.")
    return Session()

def fetch_data(table_name, engine):
    logging.info(f"Fetching data from table: {table_name}")
    logging.info(f"Engine type in fetch_data: {type(engine)}")  # Added logging

    try:
        with engine.connect() as conn:
            df = pd.read_sql_table(table_name, conn)
        logging.info(f"Successfully fetched data from {table_name}")
        return df
    except AttributeError as e:
        logging.error(f"AttributeError while fetching data from {table_name}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error fetching data from {table_name}: {e}")
        return None

def rename_columns(df):
    """
    Remove numerical prefixes from column names.
    E.g., '4. close' becomes 'close'.
    """
    original_columns = df.columns.tolist()
    df.rename(columns=lambda x: x.split('. ', 1)[1] if '. ' in x else x, inplace=True)
    new_columns = df.columns.tolist()
    logging.info(f"Renamed columns from {original_columns} to {new_columns}")
    return df

def parse_timestamp(ts):
    """
    Attempts to parse a timestamp string into a pandas Timestamp object.
    Supports multiple datetime formats:
    - "%Y-%m-%d %H:%M:%S"
    - "%Y-%m-%d %H:%M"
    - "%Y-%m-%d"

    Parameters:
        ts (str): The timestamp string to parse.

    Returns:
        pd.Timestamp or pd.NaT: The parsed Timestamp or Not-a-Time if parsing fails.
    """
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return pd.to_datetime(ts, format=fmt)
        except ValueError:
            continue
    # Attempt to parse without specifying format
    try:
        return pd.to_datetime(ts)
    except ValueError:
        return pd.NaT

def process_intraday_data(df):
    """
    Preprocess intraday data:
    - Rename columns
    - Parse timestamps
    - Handle missing data
    - Ensure consistent frequency
    """
    logging.info("Processing intraday data.")
    df = rename_columns(df)

    # Parse timestamps
    df['timestamp'] = df['timestamp'].apply(parse_timestamp)
    unparsed = df['timestamp'].isnull().sum()
    if unparsed > 0:
        logging.warning(f"{unparsed} timestamps couldn't be parsed and are set to NaT.")

    # Remove rows with unparseable dates
    df = df.dropna(subset=['timestamp'])
    if df.empty:
        logging.error("All intraday timestamps are invalid after parsing.")
        return None

    # Sort data by timestamp
    df = df.sort_values('timestamp')

    # Set timestamp as index
    df = df.set_index('timestamp')

    # Ensure there is a 'close' column
    if 'close' not in df.columns:
        logging.error("The 'close' column is missing from the intraday data.")
        return None

    # Handle missing timestamps by setting consistent frequency (Assuming hourly frequency 'H')
    df = df.asfreq('H')

    # Forward fill missing values
    df = df.fillna(method='ffill')

    logging.info("Intraday data processing completed.")
    return df

def process_hourly_data(df):
    """
    Preprocess hourly data:
    - Parse timestamps
    - Handle missing data
    - Ensure consistent frequency
    """
    logging.info("Processing hourly data.")
    # Parse timestamps
    df['timestamp'] = df['timestamp'].apply(parse_timestamp)
    unparsed = df['timestamp'].isnull().sum()
    if unparsed > 0:
        logging.warning(f"{unparsed} timestamps couldn't be parsed and are set to NaT.")

    # Remove rows with unparseable dates
    df = df.dropna(subset=['timestamp'])
    if df.empty:
        logging.error("All hourly timestamps are invalid after parsing.")
        return None

    # Sort data by timestamp
    df = df.sort_values('timestamp')

    # Set timestamp as index
    df = df.set_index('timestamp')

    # Ensure there is an 'ATR' column
    if 'ATR' not in df.columns:
        logging.error("The 'ATR' column is missing from the hourly ATR data.")
        return None

    # Handle missing timestamps by setting consistent frequency (Assuming hourly frequency 'H')
    df = df.asfreq('H')

    # Forward fill missing values
    df = df.fillna(method='ffill')

    logging.info("Hourly data processing completed.")
    return df

def process_monthly_data(df):
    """
    Preprocess monthly data:
    - Parse dates
    - Handle missing data
    - Ensure consistent frequency
    """
    logging.info("Processing monthly data.")
    # Rename columns if necessary (assuming no numerical prefixes)
    # Parse dates
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.set_index('date')

    # Ensure there is a 'value' column
    if 'value' not in df.columns:
        logging.error("The 'value' column is missing from the monthly data.")
        return None

    # Handle missing timestamps by setting consistent frequency
    df = df.asfreq('MS')  # Month Start frequency

    # Forward fill missing values
    df = df.fillna(method='ffill')

    logging.info("Monthly data processing completed.")
    return df

def process_quarterly_data(df):
    """
    Preprocess quarterly data:
    - Rename columns if necessary
    - Parse dates
    - Handle missing data
    - Ensure consistent frequency
    """
    logging.info("Processing quarterly data.")
    # Rename columns if necessary (assuming no numerical prefixes)
    # Parse dates
    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
    df = df.sort_values('fiscalDateEnding')
    df = df.set_index('fiscalDateEnding')

    # Ensure required columns exist
    required_columns = ['netIncome']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns in quarterly data: {missing_columns}")
        return None

    # Handle missing timestamps by setting consistent frequency
    df = df.asfreq('QS')  # Quarter Start frequency

    # Forward fill missing values
    df = df.fillna(method='ffill')

    logging.info("Quarterly data processing completed.")
    return df

def fetch_and_process_data(engine):
    """
    Fetch data from all tables and process them based on their frequency.
    
    Returns:
        dict: A dictionary containing processed DataFrames categorized by frequency.
    """
    logging.info("Fetching and processing all data.")

    table_frequencies = {
        'combined_intraday_data': 'intraday',
        'hourly_atr_data': 'hourly',
        'hourly_bbands_data': 'hourly',  
        'hourly_macd_data': 'hourly',    
        'hourly_rsi_data': 'hourly',     
        'hourly_vwap_data': 'hourly',    
        'monthly_cpi_data': 'monthly',
        'monthly_federal_funds_rate_data': 'monthly',
        'monthly_unemployment_data': 'monthly',
        'annual_income_statement_data': 'quarterly',
        'quarterly_earnings_data': 'quarterly',
    }

    processed_data = {
        'intraday': [],
        'hourly': [],
        'monthly': [],
        'quarterly': [],
        'other': []
    }

    for table_name, freq in table_frequencies.items():
        df = fetch_data(table_name, engine)
        if df is None:
            logging.error(f"DataFrame for table '{table_name}' is None. Skipping.")
            continue

        if freq == 'intraday':
            processed_df = process_intraday_data(df)
            if processed_df is not None:
                processed_data['intraday'].append(processed_df)
        elif freq == 'hourly':
            processed_df = process_hourly_data(df)
            if processed_df is not None:
                processed_data['hourly'].append(processed_df)
        elif freq == 'monthly':
            processed_df = process_monthly_data(df)
            if processed_df is not None:
                processed_data['monthly'].append(processed_df)
        elif freq == 'quarterly':
            processed_df = process_quarterly_data(df)
            if processed_df is not None:
                processed_data['quarterly'].append(processed_df)
        else:
            logging.warning(f"Unknown frequency '{freq}' for table '{table_name}'. Adding to 'other'.")
            processed_data['other'].append(df)

    # Concatenate dataframes within each frequency
    for freq in ['intraday', 'hourly', 'monthly', 'quarterly']:
        if processed_data[freq]:
            processed_data[freq] = pd.concat(processed_data[freq], axis=0)
            logging.info(f"Concatenated {len(processed_data[freq])} DataFrames for frequency '{freq}'.")
        else:
            processed_data[freq] = None
            logging.warning(f"No DataFrames found for frequency '{freq}'.")

    return processed_data

def debug_print_random_table(table_names, engine):
    """
    Print the first 10 lines of a random table.
    """
    random_table = random.choice(table_names)
    df = fetch_data(random_table, engine)
    if df is not None:
        print(f"\nDebug: First 10 lines of random table '{random_table}':")
        print(df.head(10))
    else:
        print(f"\nDebug: Failed to fetch data from random table '{random_table}'")

def main():
    # Calculate the current directory and project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    dotenv_path = os.path.join(project_root, 'config', '.env')

    # Load environment variables
    load_environment_variables(dotenv_path)

    try:
        engine = create_db_engine()
        logging.info(f"Engine type in main: {type(engine)}")  # Added logging

        # Fetch and process all data
        processed_data = fetch_and_process_data(engine)

        # Debug: Print first 10 lines of a random table
        table_names = [
            "annual_income_statement_data",
            "combined_intraday_data",
            "hourly_atr_data",
            "hourly_bbands_data",
            "hourly_macd_data",
            "hourly_rsi_data",
            "hourly_vwap_data",
            "monthly_cpi_data",
            "monthly_federal_funds_rate_data",
            "monthly_unemployment_data",
            "quarterly_earnings_data"
        ]
        debug_print_random_table(table_names, engine)

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
