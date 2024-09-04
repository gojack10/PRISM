import os
import json
import yaml
import requests
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

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

# Define indicators and their parameters
INDICATORS = {
    'SMA': {'function': 'SMA', 'time_period': 20, 'series_type': 'close'},
    'MACD': {'function': 'MACD', 'series_type': 'close'},
    'RSI': {'function': 'RSI', 'time_period': 14, 'series_type': 'close'},
    'BBANDS': {'function': 'BBANDS', 'time_period': 20, 'series_type': 'close'},
    'OBV': {'function': 'OBV'},
    'CCI': {'function': 'CCI', 'time_period': 20}
}

def is_market_day(date):
    # Check if it's a weekday (Monday = 0, Sunday = 6)
    return date.weekday() < 5

def get_last_market_day():
    today = datetime.now()
    last_market_day = today
    while not is_market_day(last_market_day):
        last_market_day -= timedelta(days=1)
    return last_market_day.strftime('%Y-%m-%d')

def fetch_indicator_data(ticker: str, indicator: str, start_date: str, end_date: str) -> Dict[str, Any]:
    params = {
        'function': INDICATORS[indicator]['function'],
        'symbol': ticker,
        'apikey': ALPHA_VANTAGE_API_KEY,
        'interval': 'daily',
        'outputsize': 'full'  # Request full data to ensure we get historical data
    }
    params.update({k: v for k, v in INDICATORS[indicator].items() if k not in ['function']})
    
    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {indicator} data for {ticker}: {str(e)}")
        return None

def filter_last_two_years(data: Dict[str, Any], indicator: str) -> Dict[str, Any]:
    two_years_ago = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    filtered_data = {
        'Meta Data': data.get('Meta Data', {}),
        'Technical Analysis': {
            k: v for k, v in data.get(f"Technical Analysis: {INDICATORS[indicator]['function']}", {}).items() 
            if k >= two_years_ago
        }
    }
    return filtered_data

def filter_single_day(data: Dict[str, Any], indicator: str, date: str) -> Dict[str, Any]:
    filtered_data = {
        'Meta Data': data.get('Meta Data', {}),
        'Technical Analysis': {
            k: v for k, v in data.get(f"Technical Analysis: {INDICATORS[indicator]['function']}", {}).items() 
            if k == date
        }
    }
    return filtered_data

def save_indicator_data(data: Dict[str, Any], ticker: str, indicator: str, start_date: str, end_date: str, data_type: str):
    if data_type == 'historical':
        output_dir = os.path.join(project_root, 'data', 'raw', 'indicators', 'historical', indicator)
    elif data_type == 'daily':
        output_dir = os.path.join(project_root, 'data', 'raw', 'indicators', 'daily_update', indicator)
    else:
        raise ValueError("Invalid data_type. Must be 'historical' or 'daily'.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if data_type == 'historical':
        filename = f"{ticker}_{indicator}_{start_date}_{end_date}.json"
    else:  # daily
        filename = f"{ticker}_{indicator}_{end_date}.json"
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logging.info(f"Saved {data_type} {indicator} data for {ticker} to {filepath}")

def check_and_fetch_historical_data():
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    for ticker in STOCK_TICKERS:
        for indicator in INDICATORS:
            historical_dir = os.path.join(project_root, 'data', 'raw', 'indicators', 'historical', indicator)
            os.makedirs(historical_dir, exist_ok=True)
            
            # Check if any historical data exists for this ticker and indicator
            existing_files = [f for f in os.listdir(historical_dir) if f.startswith(f"{ticker}_{indicator}_")]
            
            if existing_files:
                logging.info(f"Historical data already exists for {ticker} {indicator}. Skipping historical data collection.")
                continue
            
            filename = f"{ticker}_{indicator}_{start_date}_{end_date}.json"
            filepath = os.path.join(historical_dir, filename)
            
            if not os.path.exists(filepath):
                data = fetch_indicator_data(ticker, indicator, start_date, end_date)
                if data:
                    filtered_data = filter_last_two_years(data, indicator)
                    save_indicator_data(filtered_data, ticker, indicator, start_date, end_date, 'historical')

def fetch_daily_indicator_data():
    last_market_day = get_last_market_day()
    
    for ticker in STOCK_TICKERS:
        for indicator in INDICATORS:
            historical_dir = os.path.join(project_root, 'data', 'raw', 'indicators', 'historical', indicator)
            if not os.path.exists(historical_dir):
                logging.info(f"No historical data for {ticker} {indicator}. Skipping daily update.")
                continue
            
            data = fetch_indicator_data(ticker, indicator, last_market_day, last_market_day)
            if data:
                filtered_data = filter_single_day(data, indicator, last_market_day)
                save_indicator_data(filtered_data, ticker, indicator, last_market_day, last_market_day, 'daily')

def main():
    check_and_fetch_historical_data()
    fetch_daily_indicator_data()

if __name__ == "__main__":
    main()
