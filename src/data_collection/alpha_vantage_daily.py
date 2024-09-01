import os
import csv
import yaml
import requests
import time
from dotenv import load_dotenv
from datetime import datetime
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
dotenv_path = os.path.join(project_root, 'config', '.env')
load_dotenv(dotenv_path)

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
ALPHA_VANTAGE_BASE_URL = os.getenv('ALPHA_VANTAGE_BASE_URL')

with open('PRISM/config/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

STOCK_TICKERS = config['tickers']

def fetch_daily_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch daily data for a given symbol from Alpha Vantage
    """
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY,
        'outputsize': 'compact'
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

def save_to_csv(data: Dict[str, Any], symbol: str, date: str):
    output_dir = os.path.join(project_root, 'data', 'raw', 'historical', 'daily_update')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{symbol}_daily_{date}.csv"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'open', 'high', 'low', 'close', 'volume'])
        
        time_series = data['Time Series (Daily)']
        today_data = time_series.get(date)
        
        if today_data:
            writer.writerow([
                date,
                today_data['1. open'],
                today_data['2. high'],
                today_data['3. low'],
                today_data['4. close'],
                today_data['5. volume']
            ])
            logging.info(f"Data for {symbol} on {date} saved to {filepath}")
        else:
            logging.warning(f"No data available for {symbol} on {date}")

def main():
    today = datetime.now().strftime('%Y-%m-%d')

    for ticker in STOCK_TICKERS:
        logging.info(f"Fetching daily data for {ticker} for {today}")
        data = fetch_daily_data(ticker)
        
        if data:
            save_to_csv(data, ticker, today)
        else:
            logging.warning(f"No data retrieved for {ticker} on {today}")

if __name__ == "__main__":
    main()
