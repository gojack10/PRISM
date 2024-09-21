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
    global api_call_count
    api_call_count += 1

    # Implement rate limiting for premium API (75 calls per minute)
    if api_call_count % 75 == 0:
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            time.sleep(60 - elapsed_time)
        start_time = time.time()

    url = f"{ALPHA_VANTAGE_BASE_URL}?function={function}&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    for key, value in params.items():
        url += f"&{key}={value}"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Error fetching data for {symbol}, function {function}: {response.status_code}")
        return None

def save_csv_data(data, filename):
    file_path = os.path.join(output_dir, f"{filename}.csv")
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    logging.info(f"Saved CSV data to {file_path}")

def fetch_news_sentiment(ticker, params):
    """
    Fetch news sentiment data for a given ticker.
    """
    base_url = "https://api.yournewsapi.com/v1/"
    params['function'] = 'NEWS_SENTIMENT'
    params['tickers'] = ticker

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('feed', [])
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except Exception as err:
        logging.error(f"Other error occurred: {err}")
    return []

five_years_ago = (datetime.datetime.now() - datetime.timedelta(days=5*365)).strftime('%Y-%m')

for ticker in TICKERS:
    logging.info(f"Fetching data for {ticker}")

    # Fetch intraday data (hourly for the last 5 years)
    intraday_data = fetch_data(ticker, 'TIME_SERIES_INTRADAY', {
        'interval': '60min',
        'outputsize': 'full',
        'adjusted': 'true',
        'extended_hours': 'true',
        'month': five_years_ago
    })
    if intraday_data:
        time_series_data = intraday_data.get('Time Series (60min)', {})
        csv_data = [['ticker', 'timestamp', '1. open', '2. high', '3. low', '4. close', '5. volume']]
        csv_data.extend([ticker, k] + list(v.values()) for k, v in time_series_data.items())
        save_csv_data(csv_data, 'intraday_price_volume_data')

    # Fetch earnings data
    earnings_data = fetch_data(ticker, 'EARNINGS', {})
    if earnings_data:
        quarterly_earnings = earnings_data.get('quarterlyEarnings', [])
        csv_data = [['ticker'] + list(quarterly_earnings[0].keys())]
        csv_data.extend([ticker] + list(d.values()) for d in quarterly_earnings)
        save_csv_data(csv_data, 'quarterly_earnings_data')

    # Fetch income statement data
    income_statement_data = fetch_data(ticker, 'INCOME_STATEMENT', {})
    if income_statement_data:
        annual_reports = income_statement_data.get('annualReports', [])
        csv_data = [['ticker'] + list(annual_reports[0].keys())]
        csv_data.extend([ticker] + list(d.values()) for d in annual_reports)
        save_csv_data(csv_data, 'annual_income_statement_data')

    # Fetch news sentiment data for the last 5 years
    params = {
        'topics': 'technology,earnings,economy_macro,economy_monetary,economy_fiscal,financial_markets',
        'time_from': five_years_ago,
        'sort': 'LATEST'
    }
    logging.info(f"Fetching news sentiment data for {ticker}")
    feed = fetch_news_sentiment(ticker, params)

    if feed:
        try:
            headers = list(feed[0].keys())
            rows = [list(d.values()) for d in feed]
            save_csv_data([headers] + rows, 'news_sentiment_data')
        except Exception as e:
            logging.error(f"Error processing feed data for {ticker}: {e}")
    else:
        logging.warning(f"No news sentiment data available for {ticker}")

    # Fetch technical indicators (hourly data for the last 5 years)
    technical_indicators = {
        'RSI': {'interval': '60min', 'time_period': 14, 'series_type': 'close'},
        'MACD': {'interval': '60min', 'series_type': 'close'},
        'ATR': {'interval': '60min', 'time_period': 14},
        'BBANDS': {'interval': '60min', 'time_period': 20, 'series_type': 'close'},
        'VWAP': {'interval': '60min'},
    }

    for indicator, params in technical_indicators.items():
        params['outputsize'] = 'full'
        params['month'] = five_years_ago
        data = fetch_data(ticker, indicator, params)
        if data:
            technical_data = data.get(f'Technical Analysis: {indicator}', {})
            csv_data = [['ticker', 'timestamp'] + list(next(iter(technical_data.values())).keys())]
            csv_data.extend([ticker, k] + list(v.values()) for k, v in technical_data.items())
            save_csv_data(csv_data, f'hourly_{indicator.lower()}_data')

# Fetch economic indicators
economic_indicators = {
    'FEDERAL_FUNDS_RATE': {'interval': 'monthly'},
    'CPI': {'interval': 'monthly'},
    'UNEMPLOYMENT': {}
}

for indicator, params in economic_indicators.items():
    data = fetch_data('', indicator, params)
    if data:
        economic_data = data.get('data', [])
        csv_data = [['ticker', 'date', 'value']]
        csv_data.extend(['ECONOMIC'] + list(d.values()) for d in economic_data)
        save_csv_data(csv_data, f'monthly_{indicator.lower()}_data')

logging.info("Data collection completed.")