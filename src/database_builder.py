import os
import sys
import yaml
import logging
import requests
import datetime
import time
import json
import csv
from io import StringIO
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def load_environment_variables(dotenv_path):
    """
    Load environment variables from the .env file.
    """
    if not os.path.exists(dotenv_path):
        logging.error(f".env file not found at {dotenv_path}")
        raise FileNotFoundError(f".env file not found at {dotenv_path}")

    load_dotenv(dotenv_path, override=True)
    logging.info(f"Successfully loaded .env file from {dotenv_path}")

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

def main():
    # Calculate current directory and project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) 
    dotenv_path = os.path.join(project_root, 'config', '.env')  
    load_environment_variables(dotenv_path)
    engine = create_db_engine()

    # Test the database connection
    try:
        with engine.connect() as conn:
            logging.info("Successfully connected to the database.")
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        raise

    # Dynamically find the latest run directory
    run_dirs = glob.glob(os.path.join(project_root, 'data', 'raw', 'RUN *'))
    if not run_dirs:
        logging.error("No run directories found in data/raw/")
        raise FileNotFoundError("No run directories found in data/raw/")
    latest_run_dir = max(run_dirs, key=os.path.getctime)
    data_dir = latest_run_dir
    logging.info(f"Using data directory: {data_dir}")

    # List of CSV files to load
    csv_files = [
        'annual_income_statement_data.csv',
        'combined_intraday_data.csv',
        'hourly_atr_data.csv',
        'hourly_bbands_data.csv',
        'hourly_macd_data.csv',
        'hourly_rsi_data.csv',
        'hourly_vwap_data.csv',
        'monthly_cpi_data.csv',
        'monthly_federal_funds_rate_data.csv',
        'monthly_unemployment_data.csv',
        'quarterly_earnings_data.csv'
    ]

    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        table_name = os.path.splitext(csv_file)[0]
        if not os.path.exists(file_path):
            logging.warning(f"CSV file not found: {file_path}. Skipping.")
            continue
        try:
            df = pd.read_csv(file_path)
            # Drop the 'ticker' column if it exists
            if 'ticker' in df.columns:
                df = df.drop('ticker', axis=1)
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            logging.info(f"Loaded {csv_file} into table {table_name}")
        except Exception as e:
            logging.error(f"Failed to load {csv_file}: {e}")

    logging.info("Database population completed.")

if __name__ == "__main__":
    main()
