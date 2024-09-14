import os
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError, OperationalError, SQLAlchemyError
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()

    # Database connection details
    db_url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)

    # Get project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

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
            print(f"Processing directory: {data_dir} with prefix: {table_prefix}")
            if file_extension == '.csv':
                process_csv_directory(data_dir, table_prefix, engine)
            elif file_extension == '.json':
                process_overview_directory(data_dir, table_prefix, engine)
        else:
            print(f"Directory not found: {data_dir}")

    # Process sentiment data
    if os.path.exists(SENTIMENT_DATA_DIR):
        print(f"Processing sentiment directory: {SENTIMENT_DATA_DIR}")
        process_sentiment_data(SENTIMENT_DATA_DIR, engine)
    else:
        print(f"Sentiment data directory not found: {SENTIMENT_DATA_DIR}")

def convert_to_seconds(timestamp):
    """
    Converts various timestamp formats to Unix timestamp in seconds.
    """
    if pd.isna(timestamp):
        return np.nan
    if isinstance(timestamp, (int, float)):
        # Check if timestamp is in nanoseconds
        if timestamp > 1e18:
            return int(timestamp) // 1_000_000_000
        else:
            return int(timestamp)
    elif isinstance(timestamp, pd.Timestamp):
        return int(timestamp.timestamp())
    elif isinstance(timestamp, str):
        try:
            ts = pd.Timestamp(timestamp)
            return int(ts.timestamp())
        except ValueError:
            return np.nan
    else:
        return np.nan

def process_csv_directory(data_dir, table_prefix, engine):
    """
    Processes CSV files in the given directory, converts date/time columns,
    inserts data into the database, and creates hypertables.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        print(f"No CSV files found in {data_dir}.")
        return

    for filename in files:
        file_path = os.path.join(data_dir, filename)
        print(f"Processing file: {file_path}")

        try:
            data = pd.read_csv(file_path)
            print(f"Read {len(data)} records from {filename}.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # Identify date/time columns
        date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
        print(f"Identified date/time columns: {date_columns}")

        # Convert date/time columns to Unix timestamp in seconds
        for col in date_columns:
            data[col] = data[col].apply(convert_to_seconds)
            print(f"Converted column '{col}' to Unix timestamp in seconds.")

        # Define table name and convert to lowercase
        table_name = f"{table_prefix}{os.path.splitext(filename)[0]}".lower()
        print(f"Table name set to: {table_name}")

        # Insert data into PostgreSQL
        try:
            data.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"Data inserted into table: {table_name}")
        except Exception as e:
            print(f"Error inserting data into table {table_name}: {e}")
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
                        print(f"Renamed column '{time_column}' to 'timestamp'.")

                # Verify the time_column data type
                time_dtype = data['timestamp'].dtype
                print(f"Data type of '{time_column}': {time_dtype}")

                # Create hypertable
                hypertable_query = f"""
                SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE, migrate_data => TRUE)
                """
                conn.execute(text(hypertable_query))
                print(f"Hypertable created for table: {table_name}")
            except (ProgrammingError, OperationalError, SQLAlchemyError) as e:
                print(f"Hypertable creation error for table '{table_name}': {e}")
            except Exception as e:
                print(f"Unexpected error during hypertable creation for table '{table_name}': {e}")

def process_overview_directory(data_dir, table_prefix, engine):
    """
    Processes JSON files in the overview directory, extracts necessary fields,
    converts date fields to Unix timestamps, inserts data into the database,
    and creates hypertables.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    if not files:
        print(f"No JSON files found in {data_dir}.")
        return

    overview_records = []
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        print(f"Processing overview file: {file_path}")

        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                print(f"Loaded data from {filename}.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
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
        print(f"Processed overview data for symbol: {record.get('Symbol')}")

    if not overview_records:
        print("No overview data to insert.")
        return

    df = pd.DataFrame(overview_records)
    print(f"Prepared overview DataFrame with {len(df)} records.")

    # Define table name
    table_name = 'overview'.lower()
    print(f"Table name set to: {table_name}")

    # Insert overview data into PostgreSQL
    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Overview data inserted into table: {table_name}")
    except Exception as e:
        print(f"Error inserting overview data into table {table_name}: {e}")
        return

    # Create hypertable for overview data
    with engine.connect() as conn:
        try:
            # Ensure 'latestquarter' is used as the timestamp
            time_column = 'LatestQuarter' if 'LatestQuarter' in df.columns else None
            if time_column is None:
                print("Column 'LatestQuarter' not found in overview data.")
                return

            # Rename the column to 'timestamp' if necessary
            if time_column != 'timestamp':
                conn.execute(text(f'ALTER TABLE "{table_name}" RENAME COLUMN "{time_column}" TO timestamp'))
                df.rename(columns={time_column: 'timestamp'}, inplace=True)
                time_column = 'timestamp'
                print(f"Renamed column '{time_column}' to 'timestamp'.")

            # Verify the time_column data type
            time_dtype = df['timestamp'].dtype
            print(f"Data type of '{time_column}': {time_dtype}")

            # Create hypertable
            hypertable_query = f"""
            SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE, migrate_data => TRUE)
            """
            conn.execute(text(hypertable_query))
            print(f"Hypertable created for table: {table_name}")
        except (ProgrammingError, OperationalError, SQLAlchemyError) as e:
            print(f"Hypertable creation error for table '{table_name}': {e}")
        except Exception as e:
            print(f"Unexpected error during hypertable creation for table '{table_name}': {e}")

def process_sentiment_data(sentiment_dir, engine):
    """
    Processes JSON files in the sentiment directory, converts date fields,
    inserts data into the database, and creates hypertables.
    """
    files = [f for f in os.listdir(sentiment_dir) if f.endswith('.json')]
    if not files:
        print(f"No JSON files found in {sentiment_dir}.")
        return

    sentiment_data = []
    for filename in files:
        file_path = os.path.join(sentiment_dir, filename)
        print(f"Processing sentiment file: {file_path}")

        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                print(f"Loaded data from {filename}.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        for ticker, info in data.items():
            try:
                # Convert 'date_from' and 'date_to' to Unix timestamps in seconds
                date_from_s = convert_to_seconds(info.get('date_from', np.nan))
                date_to_s = convert_to_seconds(info.get('date_to', np.nan))

                sentiment_data.append({
                    'timestamp_from': date_from_s,
                    'timestamp_to': date_to_s,
                    'ticker': ticker.lower(),
                    'sentiment_score': info.get('overall_sentiment_score', np.nan)
                })
                print(f"Processed sentiment data for ticker: {ticker}")
            except Exception as e:
                print(f"Error processing ticker {ticker} in {filename}: {e}")

    if not sentiment_data:
        print("No sentiment data to insert.")
        return

    df = pd.DataFrame(sentiment_data)
    print(f"Prepared sentiment DataFrame with {len(df)} records.")

    # Define table name
    table_name = 'sentiment'.lower()
    print(f"Table name set to: {table_name}")

    # Insert sentiment data into PostgreSQL
    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Sentiment data inserted into table: {table_name}")
    except Exception as e:
        print(f"Error inserting sentiment data into table {table_name}: {e}")
        return

    # Create hypertable for sentiment data
    with engine.connect() as conn:
        try:
            # Decide which timestamp to use for hypertable
            # Here, we'll use 'timestamp_to'
            time_column = 'timestamp_to' if 'timestamp_to' in df.columns else 'timestamp_from'
            if time_column not in df.columns:
                print(f"Neither 'timestamp_to' nor 'timestamp_from' found in sentiment data.")
                return

            # Rename the chosen timestamp column to 'timestamp' if necessary
            if time_column != 'timestamp':
                conn.execute(text(f'ALTER TABLE "{table_name}" RENAME COLUMN "{time_column}" TO timestamp'))
                df.rename(columns={time_column: 'timestamp'}, inplace=True)
                time_column = 'timestamp'
                print(f"Renamed column '{time_column}' to 'timestamp'.")

            # Verify the time_column data type
            time_dtype = df['timestamp'].dtype
            print(f"Data type of '{time_column}': {time_dtype}")

            # Create hypertable
            hypertable_query = f"""
            SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE, migrate_data => TRUE)
            """
            conn.execute(text(hypertable_query))
            print(f"Hypertable created for table: {table_name}")
        except (ProgrammingError, OperationalError, SQLAlchemyError) as e:
            print(f"Hypertable creation error for table '{table_name}': {e}")
        except Exception as e:
            print(f"Unexpected error during hypertable creation for table '{table_name}': {e}")

if __name__ == "__main__":
    main()
