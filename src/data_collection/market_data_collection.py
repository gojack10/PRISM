import os
import sys
import time
import json
import yaml
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from io import StringIO

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
dotenv_path = os.path.join(project_root, 'config', '.env')
load_dotenv(dotenv_path)

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
ALPHA_VANTAGE_BASE_URL = os.getenv('ALPHA_VANTAGE_BASE_URL')

# Load config
config_path = os.path.join(project_root, 'config', 'config.yaml')
if not os.path.exists(config_path):
    print(f"Configuration file not found at {config_path}. Please create it and list your tickers.")
    sys.exit(1)

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

TICKERS = config.get('tickers', [])
INTRADAY_DATA_DIR = os.path.join(project_root, 'data', 'raw', 'intraday')
INDICATOR_DATA_DIR = os.path.join(project_root, 'data', 'raw', 'indicator')
OVERVIEW_DATA_DIR = os.path.join(project_root, 'data', 'raw', 'overview')

# Ensure data directories exist
os.makedirs(INTRADAY_DATA_DIR, exist_ok=True)
os.makedirs(INDICATOR_DATA_DIR, exist_ok=True)
os.makedirs(OVERVIEW_DATA_DIR, exist_ok=True)

# Define the interval
INTERVAL = '60min'

# Set up logging to print to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

def save_data(data, data_dir, filename):
    """
    Saves data to the specified directory with the given filename.
    Handles both DataFrame and dictionary data types.
    """
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, filename)
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        elif isinstance(data, dict):
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            logging.error(f"Unsupported data type for saving: {type(data)}")
            return
        logging.info(f"Data saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save data to {file_path}: {e}")

def collect_intraday_data(symbol):
    filename = f"{symbol}_intraday.csv"
    file_path = os.path.join(INTRADAY_DATA_DIR, filename)
    if os.path.exists(file_path):
        logging.info(f"Intraday data for {symbol} already exists. Skipping data collection.")
        return

    logging.info(f'Collecting intraday data for {symbol}')
    all_data = []
    
    # Start and end dates for the last 5 years
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=5*365)
    
    # Generate a list of months between start_date and end_date
    month_list = []
    current_date = start_date.replace(day=1)
    while current_date <= end_date:
        month_list.append(current_date.strftime('%Y-%m'))
        # Move to the next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    for month in month_list:
        logging.info(f'Fetching data for {symbol} for {month}')
        try:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': INTERVAL,
                'month': month,
                'outputsize': 'full',
                'adjusted': 'true',  # Adjusted data
                'apikey': ALPHA_VANTAGE_API_KEY,
            }
            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
            if response.status_code == 200:
                data = response.json()
                
                # Check if 'Time Series (60min)' is in the response
                time_series_key = f'Time Series ({INTERVAL})'
                if time_series_key in data:
                    time_series_data = data[time_series_key]
                    df = pd.DataFrame.from_dict(time_series_data, orient='index')
                    df.reset_index(inplace=True)
                    df.rename(columns={'index': 'timestamp'}, inplace=True)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    # Convert columns to numeric
                    df = df.apply(pd.to_numeric, errors='ignore')
                    all_data.append(df)
                    logging.info(f'Collected intraday data for {symbol} for {month}')
                else:
                    logging.warning(f"No intraday data found for {symbol} for {month}")
                    logging.warning(f"Response message: {data.get('Note') or data.get('Error Message')}")
            else:
                logging.error(f'Error fetching intraday data: {response.status_code} {response.text}')
            # Respect API rate limits
            time.sleep(0.8)
        except Exception as e:
            logging.error(f'Exception occurred while fetching intraday data for {symbol} for {month}: {e}')
            # Wait longer before retrying in case of an error
            time.sleep(5)
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        # Sort data by timestamp
        combined_data.sort_values('timestamp', inplace=True)
        # Reset index after sorting
        combined_data.reset_index(drop=True, inplace=True)
        save_data(combined_data, INTRADAY_DATA_DIR, filename)
    else:
        logging.info(f'No intraday data collected for {symbol}')

def collect_overview_data(symbol):
    filename = f"{symbol}_overview.json"
    file_path = os.path.join(OVERVIEW_DATA_DIR, filename)
    if os.path.exists(file_path):
        logging.info(f"Overview data for {symbol} already exists. Skipping data collection.")
        return

    logging.info(f'Collecting overview data for {symbol}')
    try:
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': ALPHA_VANTAGE_API_KEY,
        }
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            if data:
                save_data(data, OVERVIEW_DATA_DIR, filename)
                logging.info(f'Collected overview data for {symbol}')
            else:
                logging.warning(f'No overview data found for {symbol}')
            time.sleep(0.8)  # Respect API rate limits
        else:
            logging.error(f'Error fetching overview data for {symbol}: {response.status_code} {response.text}')
            time.sleep(5)
    except Exception as e:
        logging.error(f'Exception occurred while fetching overview for {symbol}: {e}')
        time.sleep(5)

def collect_indicator_data(symbol):
    logging.info(f'Collecting technical indicators for {symbol}')
    
    # Define the indicators to collect and their parameters
    indicators = [
        {
            'function': 'RSI',
            'interval': INTERVAL,
            'time_period': '10',
            'series_type': 'open',
            'suffix': 'RSI',
        },
        {
            'function': 'MACD',
            'interval': INTERVAL,
            'series_type': 'open',
            'suffix': 'MACD',
        },
        {
            'function': 'ROC',
            'interval': INTERVAL,
            'time_period': '10',
            'series_type': 'close',
            'suffix': 'ROC',
        },
        {
            'function': 'OBV',
            'interval': INTERVAL,
            'suffix': 'OBV',
        },
        {
            'function': 'ATR',
            'interval': INTERVAL,
            'time_period': '14',
            'suffix': 'ATR',
        },
        {
            'function': 'BBANDS',
            'interval': INTERVAL,
            'time_period': '5',
            'series_type': 'close',
            'nbdevup': '3',
            'nbdevdn': '3',
            'matype': '0',
            'suffix': 'BBANDS',
        },
        {
            'function': 'SMA',
            'interval': INTERVAL,
            'time_period': '10',
            'series_type': 'open',
            'suffix': 'SMA',
        },
        {
            'function': 'EMA',
            'interval': INTERVAL,
            'time_period': '10',
            'series_type': 'open',
            'suffix': 'EMA',
        },
    ]

    for indicator in indicators:
        filename = f"{symbol}_{indicator['suffix']}_indicator.csv"
        file_path = os.path.join(INDICATOR_DATA_DIR, filename)
        if os.path.exists(file_path):
            logging.info(f"{indicator['function']} data for {symbol} already exists. Skipping data collection.")
            continue

        logging.info(f"Collecting {indicator['function']} data for {symbol}")
        try:
            params = {
                'function': indicator['function'],
                'symbol': symbol,
                'interval': indicator['interval'],
                'apikey': ALPHA_VANTAGE_API_KEY,
            }
            # Add optional parameters
            for key in ['time_period', 'series_type', 'fastperiod', 'slowperiod', 'signalperiod', 'nbdevup', 'nbdevdn', 'matype']:
                if key in indicator:
                    params[key] = indicator[key]

            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
            if response.status_code == 200:
                data = response.json()
                # Determine the key for the technical analysis data
                ta_key = None
                for key in data.keys():
                    if key.startswith('Technical Analysis:'):
                        ta_key = key
                        break
                if ta_key:
                    technical_data = data[ta_key]
                    df = pd.DataFrame.from_dict(technical_data, orient='index')
                    df.reset_index(inplace=True)
                    df.rename(columns={'index': 'timestamp'}, inplace=True)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    # Convert data types
                    df = df.apply(pd.to_numeric, errors='ignore')
                    # Sort data by timestamp
                    df.sort_values('timestamp', inplace=True)
                    # Save data
                    save_data(df, INDICATOR_DATA_DIR, filename)
                    logging.info(f"Collected {indicator['function']} data for {symbol}")
                else:
                    logging.warning(f"No technical data found for {symbol} for indicator {indicator['function']}")
                    logging.warning(f"Response message: {data.get('Note') or data.get('Error Message')}")
            else:
                logging.error(f"Error fetching {indicator['function']} data for {symbol}: {response.status_code} {response.text}")
            # Respect API rate limits
            time.sleep(0.8)
        except Exception as e:
            logging.error(f"Exception occurred while fetching {indicator['function']} data for {symbol}: {e}")
            # Wait longer before retrying in case of an error
            time.sleep(5)

def main():
    """
    Main function to orchestrate data collection for all symbols.
    """
    for symbol in TICKERS:
        logging.info(f'Starting data collection for {symbol}')
        collect_intraday_data(symbol)
        collect_overview_data(symbol)
        collect_indicator_data(symbol)
    logging.info('Data collection completed.')

if __name__ == '__main__':
    main()