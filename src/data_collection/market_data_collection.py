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
import glob

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
dotenv_path = os.path.join(project_root, 'config', '.env')
load_dotenv(dotenv_path)

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query'

# Load config
config_path = os.path.join(project_root, 'config', 'config.yaml')
if not os.path.exists(config_path):
    print(f"Configuration file not found at {config_path}. Please create it and list your tickers.")
    sys.exit(1)

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

TICKERS = config.get('tickers', [])

INTERVAL = '60min'

# Set up logging to print to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

# Function to check if the market is open
def is_market_open():
    url = f"{ALPHA_VANTAGE_BASE_URL}?function=MARKET_STATUS&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for market in data.get('markets', []):
            if market['market_type'] == 'Equity' and market['region'] == 'United States':
                return market['current_status'] == 'open'
    return False

# Check if the market is open
if is_market_open():
    print("Error: Market is currently open. This script should only run when the market is closed.")
    sys.exit(1)

# Create output directory with timestamp
run_timestamp = datetime.datetime.now().strftime('RUN %Y-%m-%d %H-%M-%S')
output_dir = os.path.join(project_root, 'data', 'raw', run_timestamp)
os.makedirs(output_dir, exist_ok=True)

# Initialize a counter for API calls
api_call_count = 0
start_time = time.time()

def fetch_data(symbol, function, params):
    global api_call_count, start_time
    api_call_count += 1

    # Implement rate limiting for premium API (75 calls per minute)
    if api_call_count % 75 == 0:
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            sleep_time = 60 - elapsed_time
            logging.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
        start_time = time.time()

    url = f"{ALPHA_VANTAGE_BASE_URL}?function={function}&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    for key, value in params.items():
        url += f"&{key}={value}"

    response = requests.get(url)
    if response.status_code == 200:
        try:
            return response.json()
        except json.JSONDecodeError:
            logging.error(f"JSON decode error for {symbol}, function {function}")
            return None
    else:
        logging.error(f"Error fetching data for {symbol}, function {function}: {response.status_code}")
        return None

def save_csv_data(data, filename):
    file_path = os.path.join(output_dir, f"{filename}.csv")
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        logging.info(f"Saved CSV data to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save CSV data to {file_path}: {e}")

def save_json_data(data, filename):
    file_path = os.path.join(output_dir, f"{filename}.json")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved JSON data to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON data to {file_path}: {e}")

def fetch_news_sentiment(ticker):
    """
    Fetch news sentiment data for a given ticker.
    """
    global api_call_count, start_time
    api_call_count += 1

    # Implement rate limiting for premium API (75 calls per minute)
    if api_call_count % 75 == 0:
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            sleep_time = 60 - elapsed_time
            logging.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
        start_time = time.time()

    params = {
        'function': 'NEWS_SENTIMENT',
        'apikey': ALPHA_VANTAGE_API_KEY,
        'tickers': ticker,
        'sort': 'LATEST',
        # You can set 'limit' if desired, default is 50
        # 'limit': 100
    }

    logging.info(f"Fetching news sentiment for {ticker} with params: {params}")
    
    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        logging.info(f"Received response for {ticker}: {data.keys()}")
        
        # Log any notes or error messages from API response
        if 'Note' in data:
            logging.error(f"API Note: {data['Note']}")
        if 'Error Message' in data:
            logging.error(f"API Error: {data['Error Message']}")
        
        feed = data.get('feed', [])
        if not feed:
            logging.warning(f"No news sentiment data returned for {ticker}.")
        return feed
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
    except json.JSONDecodeError as json_err:
        logging.error(f"JSON decode error: {json_err} - Response content: {response.text[:200]}")
    except Exception as err:
        logging.error(f"Other error occurred: {err}")
    return []

five_years_ago_date = datetime.datetime.now() - datetime.timedelta(days=5*365)

# Function to iterate over each month for intraday data
def get_months_since(start_date):
    current = datetime.datetime.now().replace(day=1)
    while current >= start_date.replace(day=1):
        yield current.strftime('%Y-%m')
        current -= datetime.timedelta(days=1)
        current = current.replace(day=1)

for ticker in TICKERS:
    logging.info(f"Fetching data for {ticker}")

    # Fetch intraday data (hourly for the last 5 years, regular trading hours only)
    for month in get_months_since(five_years_ago_date):
        intraday_data = fetch_data(ticker, 'TIME_SERIES_INTRADAY', {
            'interval': INTERVAL,
            'outputsize': 'full',
            'adjusted': 'true',
            'extended_hours': 'false',
            'month': month,
            'datatype': 'json'
        })
        if intraday_data:
            time_series_key = f'Time Series ({INTERVAL})'
            time_series_data = intraday_data.get(time_series_key, {})
            if time_series_data:
                csv_data = [['ticker', 'timestamp', '1. open', '2. high', '3. low', '4. close', '5. volume']]
                csv_data.extend([[ticker, k] + list(v.values()) for k, v in time_series_data.items()])
                save_csv_data(csv_data, f'intraday_price_volume_data_{month}')
            else:
                logging.warning(f"No intraday time series data found for {ticker} in {month}")
        else:
            logging.warning(f"Failed to fetch intraday data for {ticker} in {month}")
        # Optional: Add a short sleep to respect API rate limits
        time.sleep(1)

    # Fetch earnings data
    earnings_data = fetch_data(ticker, 'EARNINGS', {})
    if earnings_data:
        quarterly_earnings = earnings_data.get('quarterlyEarnings', [])
        if quarterly_earnings:
            csv_data = [['ticker'] + list(quarterly_earnings[0].keys())]
            csv_data.extend([[ticker] + list(d.values()) for d in quarterly_earnings])
            save_csv_data(csv_data, 'quarterly_earnings_data')
        else:
            logging.warning(f"No quarterly earnings data found for {ticker}")
    else:
        logging.warning(f"Failed to fetch earnings data for {ticker}")

    # Fetch income statement data
    income_statement_data = fetch_data(ticker, 'INCOME_STATEMENT', {})
    if income_statement_data:
        annual_reports = income_statement_data.get('annualReports', [])
        if annual_reports:
            csv_data = [['ticker'] + list(annual_reports[0].keys())]
            csv_data.extend([[ticker] + list(d.values()) for d in annual_reports])
            save_csv_data(csv_data, 'annual_income_statement_data')
        else:
            logging.warning(f"No annual income statement data found for {ticker}")
    else:
        logging.warning(f"Failed to fetch income statement data for {ticker}")

    logging.info(f"Fetching news sentiment data for {ticker}")
    sentiment_data = fetch_news_sentiment(ticker)

    if sentiment_data:
        try:
            save_json_data({'feed': sentiment_data}, f'news_sentiment_data_{ticker}')
            logging.info(f"Successfully saved sentiment data for {ticker}")
        except Exception as e:
            logging.error(f"Error saving sentiment data for {ticker}: {e}")
    else:
        logging.warning(f"No news sentiment data available for {ticker}")

    # Fetch technical indicators (hourly data for the last 5 years)
    technical_indicators = {
        'RSI': {'interval': INTERVAL, 'time_period': 14, 'series_type': 'close'},
        'MACD': {'interval': INTERVAL, 'series_type': 'close'},
        'ATR': {'interval': INTERVAL, 'time_period': 14},
        'BBANDS': {'interval': INTERVAL, 'time_period': 20, 'series_type': 'close'},
        'VWAP': {'interval': INTERVAL},
    }

    for indicator, params_indicator in technical_indicators.items():
        params_indicator = params_indicator.copy()  # To avoid mutating the original dict
        params_indicator['outputsize'] = 'full'
        params_indicator['month'] = five_years_ago_date.strftime('%Y-%m')
        params_indicator['datatype'] = 'json'
        data = fetch_data(ticker, indicator, params_indicator)
        if data:
            technical_key = f'Technical Analysis: {indicator}'
            technical_data = data.get(technical_key, {})
            if technical_data:
                first_entry = next(iter(technical_data.values()), {})
                headers = ['ticker', 'timestamp'] + list(first_entry.keys())
                csv_rows = [[ticker, k] + list(v.values()) for k, v in technical_data.items()]
                csv_data = [headers] + csv_rows
                save_csv_data(csv_data, f'hourly_{indicator.lower()}_data')
            else:
                logging.warning(f"No {indicator} data found for {ticker}")
        else:
            logging.warning(f"Failed to fetch {indicator} data for {ticker}")
        time.sleep(1)

# Fetch economic indicators
economic_indicators = {
    'FEDERAL_FUNDS_RATE': {'interval': 'monthly'},
    'CPI': {'interval': 'monthly'},
    'UNEMPLOYMENT': {}
}

for indicator, params_indicator in economic_indicators.items():
    data = fetch_data('', indicator, params_indicator)
    if data:
        economic_data = data.get('data', [])
        if economic_data:
            csv_data = [['ticker', 'date', 'value']]
            csv_data.extend([['ECONOMIC'] + list(d.values()) for d in economic_data])
            save_csv_data(csv_data, f'monthly_{indicator.lower()}_data')
        else:
            logging.warning(f"No data found for economic indicator: {indicator}")
    else:
        logging.warning(f"Failed to fetch economic indicator data for: {indicator}")

def combine_intraday_csv_files(output_dir):
    # Get all intraday CSV files
    intraday_files = glob.glob(os.path.join(output_dir, 'intraday_price_volume_data_*.csv'))
    
    # Combine all dataframes
    combined_df = pd.concat([pd.read_csv(f) for f in intraday_files])
    
    # Sort by ticker and timestamp
    combined_df = combined_df.sort_values(['ticker', 'timestamp'])
    
    # Save combined dataframe
    combined_file = os.path.join(output_dir, 'combined_intraday_data.csv')
    combined_df.to_csv(combined_file, index=False)
    logging.info(f"Saved combined intraday data to {combined_file}")
    
    # Delete individual files
    for file in intraday_files:
        os.remove(file)
        logging.info(f"Deleted {file}")

# After all intraday data has been collected
combine_intraday_csv_files(output_dir)

logging.info("Data collection completed.")