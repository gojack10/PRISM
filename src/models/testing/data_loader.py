import os
import yaml
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv

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
print("=== Database Connection Details ===")
print(f"DB_USER: {db_user}")
print(f"DB_PASSWORD: {db_password}")
print(f"DB_HOST: {db_host}")
print(f"DB_PORT: {db_port}")
print(f"DB_NAME: {db_name}")
print(f"Database URL: postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
print("==================================\n")

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
        pass
except Exception as e:
    print(f"Error connecting to the database: {e}")
    raise

def load_data():
    # SQL query to retrieve all data
    query = """
    SELECT * FROM intraday_{ticker}_intraday
    JOIN indicator_{ticker}_atr_indicator USING ("timestamp")
    JOIN indicator_{ticker}_bbands_indicator USING ("timestamp")
    JOIN indicator_{ticker}_ema_indicator USING ("timestamp")
    JOIN indicator_{ticker}_macd_indicator USING ("timestamp")
    JOIN indicator_{ticker}_obv_indicator USING ("timestamp")
    JOIN indicator_{ticker}_roc_indicator USING ("timestamp")
    JOIN indicator_{ticker}_rsi_indicator USING ("timestamp")
    JOIN indicator_{ticker}_sma_indicator USING ("timestamp")
    JOIN sentiment ON intraday_{ticker}_intraday."timestamp" BETWEEN sentiment.timestamp_from AND sentiment.timestamp_to
    WHERE sentiment.ticker = '{ticker}'
    """
    
    all_data = []
    
    for ticker in TICKERS:
        # Load data into DataFrame
        df = pd.read_sql_query(query.format(ticker=ticker.lower()), engine)
        
        # Data preprocessing
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['ticker'] = ticker
        
        all_data.append(df)
        
        print(f"Loaded data for {ticker}:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:")
        print(df.head())
        print("\n" + "="*50 + "\n")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print("Combined DataFrame:")
    print(f"Shape: {combined_df.shape}")
    print(f"Columns: {combined_df.columns.tolist()}")
    print(f"First few rows:")
    print(combined_df.head())
    
    return combined_df

if __name__ == "__main__":
    load_data()