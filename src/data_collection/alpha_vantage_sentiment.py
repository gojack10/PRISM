import os
import json
from dotenv import load_dotenv
import requests
import pandas as pd
import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import yaml

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

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open('PRISM/config/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

STOCK_TICKERS = config['tickers']

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

def simplify_data(data: Dict[str, Any], max_entries: int = 10) -> Dict[str, Any]:
    """
    Simplify the JSON data to reduce token usage.

    Args:
        data: The original JSON data.
        max_entries: The maximum number of entries to keep per ticker.

    Returns:
        A simplified version of the data.
    """
    simplified_data = {}
    
    for ticker, content in data.items():
        simplified_feed = []
        feed = content.get('feed', [])[:max_entries]  # Limit the number of entries
        
        for entry in feed:
            simplified_entry = {
                'title': entry.get('title'),
                'time_published': entry.get('time_published'),
                'source': entry.get('source'),
                'topics': [topic.get('topic') for topic in entry.get('topics', [])],
                'sent_score': entry.get('overall_sentiment_score'),
                'sent_label': entry.get('overall_sentiment_label'),
                'tickers': [
                    {
                        'ticker': ts.get('ticker'),
                        'score': ts.get('ticker_sentiment_score'),
                        'label': ts.get('ticker_sentiment_label'),
                    }
                    for ts in entry.get('ticker_sentiment', [])
                ],
            }
            simplified_feed.append(simplified_entry)
        
        simplified_data[ticker] = {'feed': simplified_feed}
    
    return simplified_data

def aggregate_sentiment(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each ticker, compute the date range and overall sentiment.

    Args:
        data: The original JSON data.

    Returns:
        A dictionary with the ticker as key and date range and overall sentiment as values.
    """
    aggregated_data = {}

    for ticker, content in data.items():
        feed = content.get('feed', [])
        if not feed:
            continue  # Skip if no feed entries

        # Collect dates and sentiment scores relevant to the ticker
        dates = []
        sentiment_scores = []

        for entry in feed:
            time_published = entry.get('time_published')
            if time_published:
                # Convert time_published to datetime object
                try:
                    date_obj = datetime.strptime(time_published, '%Y%m%dT%H%M%S')
                    dates.append(date_obj)
                except ValueError as ve:
                    logging.warning(f"Invalid date format for ticker {ticker}: {time_published}")
                    continue

            # Extract sentiment scores specific to the ticker
            ticker_sentiments = entry.get('ticker_sentiment', [])
            for ts in ticker_sentiments:
                if ts.get('ticker') == ticker:
                    score = ts.get('ticker_sentiment_score')
                    if score is not None:
                        try:
                            sentiment_scores.append(float(score))
                        except ValueError:
                            logging.warning(f"Invalid sentiment score for ticker {ticker}: {score}")
                            continue

        if not dates or not sentiment_scores:
            continue  # Skip if no dates or sentiment scores

        # Determine date range
        date_from = min(dates).strftime('%Y-%m-%d %H:%M:%S')
        date_to = max(dates).strftime('%Y-%m-%d %H:%M:%S')

        # Calculate average sentiment score
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

        aggregated_data[ticker] = {
            'date_from': date_from,
            'date_to': date_to,
            'overall_sentiment_score': avg_sentiment
        }

    return aggregated_data

def main() -> None:
    # Set up the directory and file names
    raw_data_dir = os.path.join(project_root, 'data', 'raw', 'sentiment')
    os.makedirs(raw_data_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"news_sentiment_{date_str}.json"  # Format: news_sentiment_YYYYMMDD.json
    file_path = os.path.join(raw_data_dir, filename)

    # Check if the sentiment data for today's date already exists
    if os.path.exists(file_path):
        logging.info(f"Sentiment data for date {date_str} already exists at {file_path}. Skipping data collection.")
        print(f"Sentiment data for date {date_str} already exists at {file_path}. Skipping data collection.")
        return  # Exit the function

    all_data = {}

    for ticker in STOCK_TICKERS:
        json_response = make_alpha_vantage_call([ticker])
        if json_response:
            all_data[ticker] = json_response
            logging.info(f"Successfully retrieved data for ticker: {ticker}")
        else:
            logging.error(f"Failed to retrieve data for ticker: {ticker}")

    if all_data:
        # Aggregate the data to get date range and overall sentiment
        aggregated_data = aggregate_sentiment(all_data)

        # Save the aggregated data to a JSON file
        with open(file_path, 'w') as f:
            json.dump(aggregated_data, f, indent=2)

        logging.info(f"Aggregated sentiment data saved to {file_path}")
        print(f"Aggregated sentiment data saved to {file_path}")
    else:
        logging.error("No data was successfully retrieved and processed.")
        print("No data was successfully retrieved and processed.")

if __name__ == "__main__":
    main()
