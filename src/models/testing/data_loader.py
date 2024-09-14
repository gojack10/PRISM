import os
import yaml
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv
import logging

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
logging.info(f"Database URL: postgresql+psycopg2://{db_user}:***@{db_host}:{db_port}/{db_name}")
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
db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Create the engine
try:
    engine = create_engine(db_url)
    # Test the connection
    with engine.connect() as conn:
        logging.info("Successfully connected to the database.")
except Exception as e:
    logging.error(f"Error connecting to the database: {e}")
    raise

def load_data():
    """
    Loads intraday and indicator data for each ticker from the database.
    Sentiment data is currently excluded until more comprehensive data is available.
    """
    # SQL query to retrieve all data without the sentiment join
    query = """
    SELECT *
    FROM intraday_{ticker}_intraday
    JOIN indicator_{ticker}_atr_indicator USING ("timestamp")
    JOIN indicator_{ticker}_bbands_indicator USING ("timestamp")
    JOIN indicator_{ticker}_ema_indicator USING ("timestamp")
    JOIN indicator_{ticker}_macd_indicator USING ("timestamp")
    JOIN indicator_{ticker}_obv_indicator USING ("timestamp")
    JOIN indicator_{ticker}_roc_indicator USING ("timestamp")
    JOIN indicator_{ticker}_rsi_indicator USING ("timestamp")
    JOIN indicator_{ticker}_sma_indicator USING ("timestamp")
    """
    
    # Log the query before formatting
    logging.debug("Initial SQL Query Template:")
    logging.debug(query)

    # Count occurrences of 'JOIN sentiment'
    join_sentiment_count = query.count("JOIN sentiment")
    logging.debug(f"'JOIN sentiment' occurs {join_sentiment_count} time(s) in the query template.")

    all_data = []

    for ticker in TICKERS:
        logging.info(f"Loading data for ticker: {ticker}")
        try:
            with engine.connect() as conn:
                logging.info(f"Executing SQL query for ticker: {ticker.lower()}")

                # Format the query
                formatted_query = query.format(
                    ticker=ticker.lower(),
                    ticker_condition=f"ticker = '{ticker.lower()}'"  # Adjust the WHERE condition if necessary
                )
                logging.debug(f"Formatted SQL Query for {ticker}:\n{formatted_query}")

                # Additional debugging
                query_length = len(formatted_query)
                logging.debug(f"Formatted query length: {query_length} characters.")
                join_sentiment_count = formatted_query.count("JOIN sentiment")
                logging.debug(f"'JOIN sentiment' occurs {join_sentiment_count} time(s) in the formatted query.")

                # Optionally, log the first and last 500 characters to avoid overwhelming the logs
                if query_length > 1000:
                    logging.debug(f"Formatted Query Snippet for {ticker}:\n{formatted_query[:500]}...\n...\n{formatted_query[-500:]}")
                else:
                    logging.debug(f"Formatted Query Content:\n{formatted_query}")

                # Execute the query
                sql_query = text(formatted_query)
                df = pd.read_sql_query(sql_query, conn)

                logging.info(f"Rows retrieved for {ticker}: {len(df)}")

        except Exception as e:
            logging.error(f"Error loading data for ticker {ticker}: {e}")
            continue

        # Data preprocessing
        logging.info(f"Preprocessing data for ticker: {ticker}")
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.rename(columns={'timestamp': 'date'})
            logging.info("Renamed 'timestamp' column to 'date'")
        else:
            logging.warning(f"'timestamp' column not found in data for {ticker}")

        if 'date' not in df.columns:
            logging.warning(f"'date' column not found after renaming for {ticker}")
        else:
            df = df.set_index('date')
            logging.info("Set 'date' as index")

        if df.index.nunique() < 50:
            logging.warning("Not enough unique dates for time series modeling.")
            # Optionally, you can choose to continue or skip adding this ticker's data
            continue

        all_data.append(df)

        logging.info(f"Loaded data for {ticker}:")
        logging.info(f"Shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")
        logging.info(f"First few rows:")
        logging.info(f"\n{df.head()}")
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

    return combined_df

if __name__ == "__main__":
    load_data()