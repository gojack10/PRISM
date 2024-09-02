import os
import json
import requests
import logging
from dotenv import load_dotenv
from datetime import datetime
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
dotenv_path = os.path.join(project_root, 'config', '.env')
load_dotenv(dotenv_path)

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
ALPHA_VANTAGE_BASE_URL = os.getenv('ALPHA_VANTAGE_BASE_URL')

# Load config
with open(os.path.join(project_root, 'config', 'config.yaml'), 'r') as config_file:
    config = yaml.safe_load(config_file)

STOCK_TICKERS = config['tickers']

# Fields to keep in the cleaned JSON
FIELDS_TO_KEEP = [
    "PERatio", "PEGRatio", "PriceToBookRatio",
    "OperatingMarginTTM", "ReturnOnEquityTTM",
    "QuarterlyEarningsGrowthYOY", "QuarterlyRevenueGrowthYOY",
    "DividendYield", "AnalystTargetPrice",
    "50DayMovingAverage", "200DayMovingAverage",
    "Beta", "MarketCapitalization"
]

def fetch_company_overview(symbol: str) -> dict:
    """
    Fetch company overview for a given symbol from Alpha Vantage
    """
    params = {
        'function': 'OVERVIEW',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error occurred: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    
    return None

def clean_data(data: dict) -> dict:
    """
    Clean the data to only include specified fields
    """
    return {key: data[key] for key in FIELDS_TO_KEEP if key in data}

def save_to_json(data: dict, symbol: str):
    output_dir = os.path.join(project_root, 'data', 'raw', 'overview')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{symbol}_overview_{datetime.now().strftime('%Y-%m-%d')}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logging.info(f"Company overview for {symbol} saved to {filepath}")

def main():
    for ticker in STOCK_TICKERS:
        logging.info(f"Fetching company overview for {ticker}")
        data = fetch_company_overview(ticker)
        
        if data:
            cleaned_data = clean_data(data)
            save_to_json(cleaned_data, ticker)
        else:
            logging.warning(f"No data retrieved for {ticker}")

if __name__ == "__main__":
    main()