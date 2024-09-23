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

def convert_timestamps_to_unix(df):
    if df is None:
        logging.error("DataFrame is None. Cannot convert timestamps.")
        return None
    timestamp_cols = df.select_dtypes(include=['datetime64[ns]', 'object']).columns
    for col in timestamp_cols:
        if pd.api.types.is_datetime64_ns_dtype(df[col]):
            df[col] = df[col].astype(int) / 10**9  # Convert to UNIX timestamp
    return df

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

        for table in table_names:
            df = fetch_data(table, engine)
            if df is not None:
                df = convert_timestamps_to_unix(df)
                # Further processing as needed
            else:
                logging.error(f"Failed to process table: {table}")

        # Debug: Print first 10 lines of a random table
        debug_print_random_table(table_names, engine)

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
