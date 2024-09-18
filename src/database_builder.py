import os
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError, OperationalError, SQLAlchemyError
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if not os.path.basename(project_root) == 'PRISM':
        raise ValueError("Unexpected directory structure. Make sure this script is in the PRISM/src directory.")
    
    dotenv_path = os.path.join(project_root, 'config', '.env')
    
    if not os.path.exists(dotenv_path):
        logging.error(f".env file not found at {dotenv_path}")
        raise FileNotFoundError(f".env file not found at {dotenv_path}")
    
    # Load environment variables from the specified .env file
    loaded = load_dotenv(dotenv_path, override=True)
    if loaded:
        logging.info(f"Successfully loaded .env file from {dotenv_path}")
    else:
        logging.error(f"Failed to load .env file from {dotenv_path}")
        raise FileNotFoundError(f"Failed to load .env file from {dotenv_path}")
    
    # Database connection details
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT', '5432')  # Default to 5432 if not set
    db_name = os.getenv('DB_NAME')
    
    # Debugging: Print environment variables to verify they're loaded correctly
    logging.info("=== Database Connection Details ===")
    logging.info(f"DB_USER: {db_user}")
    logging.info(f"DB_PASSWORD: {'***' if db_password else None}")
    logging.info(f"DB_HOST: {db_host}")
    logging.info(f"DB_PORT: {db_port}")
    logging.info(f"DB_NAME: {db_name}")
    logging.info(f"Env file path: {dotenv_path}")
    logging.info("==================================\n")
    
    # Check if all necessary environment variables are set
    missing_vars = []
    for var_name, var_value in [('DB_USER', db_user), ('DB_PASSWORD', db_password), 
                                ('DB_HOST', db_host), ('DB_NAME', db_name)]:
        if not var_value:
            missing_vars.append(var_name)
    
    if missing_vars:
        raise ValueError(f"Missing database connection details for: {', '.join(missing_vars)}. Please check your .env file.")
    
    # Construct the database URL
    try:
        db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        logging.info(f"Database URL: postgresql+psycopg2://{db_user}:***@{db_host}:{db_port}/{db_name}")
    except Exception as e:
        logging.error(f"Error constructing database URL: {e}")
        raise
    
    # Create the engine
    try:
        engine = create_engine(db_url)
        # Test the connection
        with engine.connect() as conn:
            logging.info("Successfully connected to the database.")
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        raise
    
    # Define data directories
    INTRADAY_DATA_DIR = os.path.join(project_root, 'data', 'raw', 'intraday')
    INDICATOR_DATA_DIR = os.path.join(project_root, 'data', 'raw', 'indicator')
    OVERVIEW_DATA_DIR = os.path.join(project_root, 'data', 'raw', 'overview')
    SENTIMENT_DATA_DIR = os.path.join(project_root, 'data', 'raw', 'sentiment')

    # Process each data directory
    for data_dir, table_prefix, file_extension in [
        (INTRADAY_DATA_DIR, 'intraday_', '.csv'),
        (INDICATOR_DATA_DIR, 'indicator_', '.csv'),
        (OVERVIEW_DATA_DIR, 'overview_', '.json')
    ]:
        if os.path.exists(data_dir):
            logging.info(f"Processing directory: {data_dir} with prefix: {table_prefix}")
            if file_extension == '.csv':
                process_csv_directory(data_dir, table_prefix, engine)
            elif file_extension == '.json':
                process_overview_directory(data_dir, table_prefix, engine)
        else:
            logging.warning(f"Directory not found: {data_dir}")

    # Process sentiment data
    if os.path.exists(SENTIMENT_DATA_DIR):
        logging.info(f"Processing sentiment directory: {SENTIMENT_DATA_DIR}")
        process_sentiment_data(SENTIMENT_DATA_DIR, engine)
    else:
        logging.warning(f"Sentiment data directory not found: {SENTIMENT_DATA_DIR}")

    logging.info("Database builder completed successfully.")

def convert_to_seconds(timestamp):
    """
    Converts various timestamp formats to Unix timestamp in seconds.
    Floors the timestamp to the nearest hour.
    """
    if pd.isna(timestamp):
        return np.nan

    # If timestamp is a string of digits, convert to int first
    if isinstance(timestamp, str):
        if timestamp.isdigit():
            timestamp = int(timestamp)
        else:
            try:
                ts = pd.Timestamp(timestamp)
                return int(ts.timestamp())
            except ValueError:
                return np.nan

    if isinstance(timestamp, (int, float)):
        # Check if timestamp is in nanoseconds
        if timestamp > 1e12:  # Unix time in nanoseconds
            timestamp = int(timestamp) // 1_000_000_000  # Convert to seconds
        else:
            timestamp = int(timestamp)

    elif isinstance(timestamp, pd.Timestamp):
        timestamp = int(timestamp.timestamp())

    else:
        return np.nan

    # Floor the timestamp to the nearest hour
    try:
        dt = pd.to_datetime(timestamp, unit='s')
        floored_dt = dt.floor('H')
        return int(floored_dt.timestamp())
    except Exception:
        return np.nan

def process_csv_directory(data_dir, table_prefix, engine):
    """
    Processes CSV files in the given directory, converts date/time columns,
    inserts data into the database, and creates hypertables.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        logging.warning(f"No CSV files found in {data_dir}.")
        return

    for filename in files:
        file_path = os.path.join(data_dir, filename)
        logging.info(f"Processing file: {file_path}")

        try:
            data = pd.read_csv(file_path)
            logging.info(f"Read {len(data)} records from {filename}.")
        except Exception as e:
            logging.error(f"Error reading {filename}: {e}")
            continue

        # Identify date/time columns
        date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
        logging.info(f"Identified date/time columns: {date_columns}")

        # Convert date/time columns to Unix timestamp in seconds
        for col in date_columns:
            data[col] = data[col].apply(convert_to_seconds)
            logging.info(f"Converted column '{col}' to Unix timestamp in seconds.")

        # Define table name and convert to lowercase
        table_name = f"{table_prefix}{os.path.splitext(filename)[0]}".lower()
        logging.info(f"Table name set to: {table_name}")

        # Insert data into PostgreSQL
        try:
            data.to_sql(table_name, engine, if_exists='replace', index=False)
            logging.info(f"Data inserted into table: {table_name}")
        except Exception as e:
            logging.error(f"Error inserting data into table {table_name}: {e}")
            continue

        # Create hypertable in TimescaleDB
        with engine.connect() as conn:
            try:
                # Ensure 'timestamp' column exists
                if 'timestamp' in data.columns:
                    time_column = 'timestamp'
                else:
                    # Use the first date/time column found
                    time_column = date_columns[0]
                    # Rename the time column to 'timestamp' if it's different
                    if time_column != 'timestamp':
                        conn.execute(text(f'ALTER TABLE "{table_name}" RENAME COLUMN "{time_column}" TO timestamp'))
                        data.rename(columns={time_column: 'timestamp'}, inplace=True)
                        time_column = 'timestamp'
                        logging.info(f"Renamed column '{time_column}' to 'timestamp'.")

                # Verify the time_column data type
                time_dtype = data['timestamp'].dtype
                logging.info(f"Data type of '{time_column}': {time_dtype}")

                # Create hypertable
                hypertable_query = f"""
                SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE, migrate_data => TRUE)
                """
                conn.execute(text(hypertable_query))
                logging.info(f"Hypertable created for table: {table_name}")
            except (ProgrammingError, OperationalError, SQLAlchemyError) as e:
                logging.error(f"Hypertable creation error for table '{table_name}': {e}")
            except Exception as e:
                logging.error(f"Unexpected error during hypertable creation for table '{table_name}': {e}")

def process_overview_directory(data_dir, table_prefix, engine):
    """
    Processes JSON files in the overview directory, extracts necessary fields,
    converts date fields to Unix timestamps, inserts data into the database,
    and creates hypertables.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    if not files:
        logging.warning(f"No JSON files found in {data_dir}.")
        return

    overview_records = []
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        logging.info(f"Processing overview file: {file_path}")

        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                logging.info(f"Loaded data from {filename}.")
        except Exception as e:
            logging.error(f"Error reading {filename}: {e}")
            continue

        # Select only the fields needed for the PyCaret model
        # Modify the list below based on your model's requirements
        required_fields = [
            "Symbol",
            "MarketCapitalization",
            "EBITDA",
            "PERatio",
            "EPS",
            "RevenueTTM",
            "GrossProfitTTM",
            "DilutedEPSTTM",
            "QuarterlyEarningsGrowthYOY",
            "QuarterlyRevenueGrowthYOY",
            "TrailingPE",
            "ForwardPE",
            "PriceToSalesRatioTTM",
            "PriceToBookRatio",
            "EVToRevenue",
            "EVToEBITDA",
            "Beta",
            "52WeekHigh",
            "52WeekLow",
            "50DayMovingAverage",
            "200DayMovingAverage",
            "SharesOutstanding",
            "DividendPerShare",
            "DividendYield",
            "DividendDate",
            "ExDividendDate",
            "LatestQuarter"
        ]

        record = {}
        for field in required_fields:
            record[field] = data.get(field, np.nan)

        # Convert date fields to Unix timestamps in seconds
        date_fields = ["LatestQuarter", "DividendDate", "ExDividendDate"]
        for date_field in date_fields:
            if pd.notna(record.get(date_field)):
                record[date_field] = convert_to_seconds(record[date_field])
            else:
                record[date_field] = np.nan

        overview_records.append(record)

    if not overview_records:
        logging.warning(f"No valid records found in {data_dir}.")
        return

    # Create DataFrame
    df = pd.DataFrame(overview_records)
    logging.info(f"Created DataFrame with {len(df)} records from overview data.")

    # Define table name and convert to lowercase
    table_name = f"{table_prefix}overview".lower()
    logging.info(f"Table name set to: {table_name}")

    # Insert data into PostgreSQL
    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        logging.info(f"Data inserted into table: {table_name}")
    except Exception as e:
        logging.error(f"Error inserting data into table {table_name}: {e}")
        return

    # Create hypertable in TimescaleDB
    with engine.connect() as conn:
        try:
            # Ensure 'timestamp' column exists
            if 'timestamp' in df.columns:
                time_column = 'timestamp'
            else:
                # Use the first date/time column found
                time_column = date_fields[0]
                # Rename the time column to 'timestamp' if it's different
                if time_column != 'timestamp':
                    conn.execute(text(f'ALTER TABLE "{table_name}" RENAME COLUMN "{time_column}" TO timestamp'))
                    df.rename(columns={time_column: 'timestamp'}, inplace=True)
                    time_column = 'timestamp'
                    logging.info(f"Renamed column '{time_column}' to 'timestamp'.")
            # Ensure 'latestquarter' is used as the timestamp
            time_column = 'LatestQuarter' if 'LatestQuarter' in df.columns else None
            if time_column is None:
                logging.warning("Column 'LatestQuarter' not found in overview data.")
                return

            # Rename the column to 'timestamp' if necessary
            if time_column != 'timestamp':
                conn.execute(text(f'ALTER TABLE "{table_name}" RENAME COLUMN "{time_column}" TO timestamp'))
                df.rename(columns={time_column: 'timestamp'}, inplace=True)
                time_column = 'timestamp'
                logging.info(f"Renamed column '{time_column}' to 'timestamp'.")

            # Verify the time_column data type
            time_dtype = df['timestamp'].dtype
            logging.info(f"Data type of '{time_column}': {time_dtype}")

            # Create hypertable
            hypertable_query = f"""
            SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE, migrate_data => TRUE)
            """
            conn.execute(text(hypertable_query))
            logging.info(f"Hypertable created for table: {table_name}")
        except (ProgrammingError, OperationalError, SQLAlchemyError) as e:
            logging.error(f"Hypertable creation error for table '{table_name}': {e}")
        except Exception as e:
            logging.error(f"Unexpected error during hypertable creation for table '{table_name}': {e}")

def process_sentiment_data(sentiment_dir, engine):
    """
    Processes JSON files in the sentiment directory, converts date fields,
    inserts data into the database, and creates hypertables.
    """
    files = [f for f in os.listdir(sentiment_dir) if f.endswith('.json')]
    if not files:
        logging.warning(f"No JSON files found in {sentiment_dir}.")
        return

    sentiment_data = []
    for filename in files:
        file_path = os.path.join(sentiment_dir, filename)
        logging.info(f"Processing sentiment file: {file_path}")

        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                logging.info(f"Loaded data from {filename}.")
        except Exception as e:
            logging.error(f"Error reading {filename}: {e}")
            continue

        for ticker, info in data.items():
            try:
                # Convert 'date_from' and 'date_to' to Unix timestamps in seconds
                date_from_s = convert_to_seconds(info.get('date_from', np.nan))
                date_to_s = convert_to_seconds(info.get('date_to', np.nan))

                sentiment_data.append({
                    'timestamp': date_to_s,
                    'ticker': ticker.lower(),
                    'sentiment_score': info.get('overall_sentiment_score', np.nan)
                })
                logging.info(f"Processed sentiment data for ticker: {ticker}")
            except Exception as e:
                logging.error(f"Error processing ticker {ticker} in {filename}: {e}")

    if not sentiment_data:
        logging.warning("No sentiment data to insert.")
        return

    df = pd.DataFrame(sentiment_data)
    logging.info(f"Prepared sentiment DataFrame with {len(df)} records.")

    # Define table name
    table_name = 'sentiment'.lower()
    logging.info(f"Table name set to: {table_name}")

    # Insert sentiment data into PostgreSQL
    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        logging.info(f"Sentiment data inserted into table: {table_name}")
    except Exception as e:
        logging.error(f"Error inserting sentiment data into table {table_name}: {e}")
        return

    # Create hypertable for sentiment data
    with engine.connect() as conn:
        try:
            # Use 'timestamp' column for hypertable
            time_column = 'timestamp'
            if time_column not in df.columns:
                logging.warning(f"'timestamp' column not found in sentiment data.")
                return

            # Verify the time_column data type
            time_dtype = df['timestamp'].dtype
            logging.info(f"Data type of '{time_column}': {time_dtype}")

            # Create hypertable
            hypertable_query = f"""
            SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE, migrate_data => TRUE)
            """
            conn.execute(text(hypertable_query))
            logging.info(f"Hypertable created for table: {table_name}")
        except (ProgrammingError, OperationalError, SQLAlchemyError) as e:
            logging.error(f"Hypertable creation error for table '{table_name}': {e}")
        except Exception as e:
            logging.error(f"Unexpected error during hypertable creation for table '{table_name}': {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
