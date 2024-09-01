import os
import json
from dotenv import load_dotenv
import requests
import pandas as pd
import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

# path to the .env file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
dotenv_path = os.path.join(project_root, 'config', '.env')

# load environment variables from .env file
load_dotenv(dotenv_path)

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
ALPHA_VANTAGE_BASE_URL = os.getenv('ALPHA_VANTAGE_BASE_URL')

print(f"API Key: {ALPHA_VANTAGE_API_KEY}")
print(f"Base URL: {ALPHA_VANTAGE_BASE_URL}")

# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

STOCK_TICKERS = ['AAPL','NVDA', 'GOOG']

def make_alpha_vantage_call(tickers: List[str], topics: Optional[List[str]] = None, 
                            time_from: Optional[str] = None, time_to: Optional[str] = None, 
                            sort: str = 'LATEST', limit: int = 50) -> Optional[Dict[str, Any]]:
    """
    Make an API call to Alpha Vantage News Sentiment endpoint.
    """
    if not ALPHA_VANTAGE_BASE_URL:
        logging.error("ALPHA_VANTAGE_BASE_URL is not set")
        return None
    
    params = {
        'function': 'NEWS_SENTIMENT',
        'apikey': ALPHA_VANTAGE_API_KEY,
        'tickers': ','.join(tickers),
        'sort': sort,
        'limit': limit
    }
    if topics:
        params['topics'] = ','.join(topics)
    if time_from:
        params['time_from'] = time_from
    if time_to:
        params['time_to'] = time_to

    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            logging.warning("Rate limit reached. Waiting for 60 seconds before retrying.")
            time.sleep(60)
            return make_alpha_vantage_call(tickers, topics, time_from, time_to, sort, limit)
        else:
            logging.error(f"HTTP error occurred: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    
    return None

def process_news_sentiment(json_response: Dict[str, Any]) -> pd.DataFrame:
    """
    Process the JSON response from Alpha Vantage and extract relevant information.
    """
    processed_data = []
    
    for entry in json_response.get('feed', []):
        article_data = {
            'title': entry.get('title'),
            'url': entry.get('url'),
            'time_published': entry.get('time_published'),
            'authors': ', '.join(entry.get('authors', [])),
            'summary': entry.get('summary'),
            'source': entry.get('source'),
            'overall_sentiment_score': entry.get('overall_sentiment_score'),
            'overall_sentiment_label': entry.get('overall_sentiment_label'),
        }
        
        for ticker_sentiment in entry.get('ticker_sentiment', []):
            ticker = ticker_sentiment.get('ticker')
            article_data[f'{ticker}_relevance_score'] = ticker_sentiment.get('relevance_score')
            article_data[f'{ticker}_sentiment_score'] = ticker_sentiment.get('ticker_sentiment_score')
            article_data[f'{ticker}_sentiment_label'] = ticker_sentiment.get('ticker_sentiment_label')
        
        processed_data.append(article_data)
    
    return pd.DataFrame(processed_data)

def prepare_data_for_claude(processed_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare the processed data for Claude API.
    """
    prepared_data = []
    for _, row in processed_data.iterrows():
        article = {
            'content': f"{row['title']} {row['summary']}",
            'sentiment': row['overall_sentiment_label'],
            'score': row['overall_sentiment_score'],
            'tickers': {}
        }
        for col in row.index:
            if '_relevance_score' in col:
                ticker = col.split('_')[0]
                article['tickers'][ticker] = {
                    'relevance': row[f'{ticker}_relevance_score'],
                    'sentiment': row[f'{ticker}_sentiment_label'],
                    'score': row[f'{ticker}_sentiment_score']
                }
        prepared_data.append(article)
    
    return {'articles': prepared_data}

def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the raw data before saving. You can add any transformations here.
    """
    return data

def main() -> None:
    all_data = {}
    
    for ticker in STOCK_TICKERS:
        json_response = make_alpha_vantage_call([ticker])
        if json_response:
            all_data[ticker] = json_response
            logging.info(f"Successfully retrieved data for ticker: {ticker}")
        else:
            logging.error(f"Failed to retrieve data for ticker: {ticker}")
    
    if all_data:
        raw_data_dir = os.path.join(project_root, 'data', 'raw')
        os.makedirs(raw_data_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alpha_vantage_data_{timestamp}.json"
        file_path = os.path.join(raw_data_dir, filename)
        
        # Save the data to a JSON file
        processed_data = process_data(all_data)
        with open(file_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        logging.info(f"Data saved to {file_path}")
    else:
        logging.error("No data was successfully retrieved and processed.")

if __name__ == "__main__":
    main()
