import os
import yaml
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv
import logging
import psycopg2
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Determine the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

# Path to the .env file
dotenv_path = os.path.join(project_root, 'config', '.env')
load_dotenv(dotenv_path)

# Load config
config_path = os.path.join(project_root, 'config', 'config.yaml')

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

TICKERS = config.get('tickers', [])

# Define indicators with their specific columns and desired aliases
INDICATORS = {
    'BBANDS': {
        'columns': ['Real Upper Band', 'Real Middle Band', 'Real Lower Band'],
        'aliases': ['BBANDS_REAL_UPPER_BAND', 'BBANDS_REAL_MIDDLE_BAND', 'BBANDS_REAL_LOWER_BAND']
    },
    'ATR': {
        'columns': ['ATR'],
        'aliases': ['ATR_ATR']
    },
    'EMA': {
        'columns': ['EMA'],
        'aliases': ['EMA_EMA']
    },
    'MACD': {
        'columns': ['MACD', 'MACD_Signal', 'MACD_Hist'],
        'aliases': ['MACD_MACD', 'MACD_MACD_SIGNAL', 'MACD_MACD_HIST']
    },
    'OBV': {
        'columns': ['OBV'],
        'aliases': ['OBV_OBV']
    },
    'ROC': {
        'columns': ['ROC'],
        'aliases': ['ROC_ROC']
    },
    'RSI': {
        'columns': ['RSI'],
        'aliases': ['RSI_RSI']
    },
    'SMA': {
        'columns': ['SMA'],
        'aliases': ['SMA_SMA']
    }
}

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
logging.info(f"Indicators: {INDICATORS}")
logging.info("==================================\n")

# Check if all necessary environment variables are set
missing_vars = []
for var_name, var_value in [('DB_USER', db_user), ('DB_PASSWORD', db_password), 
                            ('DB_HOST', db_host), ('DB_NAME', db_name)]:
    if not var_value:
        missing_vars.append(var_name)

if missing_vars:
    raise ValueError(f"Missing database connection details for: {', '.join(missing_vars)}. Please check your .env file.")

def build_sql_query(ticker, indicators):
    select_fields = "i.*, "
    lateral_joins = ""

    # Define indicators and their respective columns
    indicator_columns = {
        'BBANDS': ['Real Upper Band', 'Real Middle Band', 'Real Lower Band'],
        'ATR': ['ATR'],
        'EMA': ['EMA'],
        'MACD': ['MACD', 'MACD_Signal', 'MACD_Hist'],
        'OBV': ['OBV'],
        'ROC': ['ROC'],
        'RSI': ['RSI'],
        'SMA': ['SMA']
    }

    for indicator, columns in indicator_columns.items():
        table_name = f"indicator_{ticker.lower()}_{indicator.lower()}_indicator"
        for column in columns:
            # Create a unique alias by combining indicator and column, replacing spaces with underscores
            alias = f"{indicator}_{column.replace(' ', '_')}".upper()

            # Append to SELECT clause with proper quoting
            select_fields += f'"{alias}"."{column}" AS "{alias}", '

            # Append to LATERAL JOIN clause with consistent quoting
            lateral_joins += f"""
    LEFT JOIN LATERAL (
        SELECT "{column}"
        FROM {table_name} "{alias}"
        WHERE "{alias}"."timestamp" <= i."timestamp"
        ORDER BY "{alias}"."timestamp" DESC
        LIMIT 1
    ) "{alias}" ON true
            """

    # Remove trailing comma and space from SELECT fields
    select_fields = select_fields.rstrip(', ')

    # Construct the final SQL query
    sql_query = f"""
SELECT 
    {select_fields}
FROM 
    intraday_{ticker.lower()}_intraday i
{lateral_joins}
ORDER BY i."timestamp"
LIMIT 80;
    """
    return sql_query

def load_data():
    # Create the SQLAlchemy engine
    engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
    all_data = []

    with engine.connect() as conn:
        for ticker in TICKERS:
            logging.info(f"Loading data for ticker: {ticker}")
            try:
                sql_query = build_sql_query(ticker, INDICATORS)
                logging.debug(f"Generated SQL Query for {ticker}:\n{sql_query[:500]}...")  # Log first 500 chars

                logging.info(f"Executing SQL query for ticker: {ticker}")
                query_start_time = pd.Timestamp.now()

                # Execute the query
                df = pd.read_sql_query(text(sql_query), conn)

                query_end_time = pd.Timestamp.now()
                duration = (query_end_time - query_start_time).total_seconds()
                logging.info(f"Query executed in {duration} seconds for ticker {ticker}")

                logging.info(f"Rows retrieved for ticker {ticker}: {len(df)}")

            except Exception as e:
                logging.error(f"Error loading data for ticker {ticker}: {e}")
                # Rollback the failed transaction
                conn.execute("ROLLBACK")
                continue

            # Data preprocessing steps...
            df.rename(columns={'timestamp': 'date'}, inplace=True)
            df.set_index('date', inplace=True)
            all_data.append(df)
            logging.info(f"Loaded and processed data for ticker {ticker}:")
            logging.info(f"Shape: {df.shape}")
            logging.info(f"Columns: {df.columns.tolist()}")
            logging.info("\n" + "="*50 + "\n")

    if not all_data:
        raise ValueError("No data was loaded. Please check your SQL queries and database connection.")

    # Combine all DataFrames
    logging.info("Combining all ticker data into a single DataFrame...")
    combined_df = pd.concat(all_data, ignore_index=False)

    logging.info("Combined DataFrame:")
    logging.info(f"Shape: {combined_df.shape}")
    logging.info(f"Columns: {combined_df.columns.tolist()}")
    logging.info(f"First few rows:")
    logging.info(f"\n{combined_df.head()}")

    # DEBUG: Export the combined DataFrame to a CSV file for inspection
    # Uncomment the following lines during debugging and comment them out after resolving issues
    # debug_csv_path = os.path.join(project_root, 'combined_data_debug.csv')
    # combined_df.to_csv(debug_csv_path)
    # logging.debug(f"Exported combined DataFrame to CSV at: {debug_csv_path}")

    return combined_df

if __name__ == "__main__":
    combined_data = load_data()