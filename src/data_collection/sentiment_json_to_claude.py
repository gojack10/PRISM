import json
import os
import logging
from dotenv import load_dotenv
import anthropic
from datetime import datetime
import yaml
import time
import random
from ratelimit import limits, sleep_and_retry
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
dotenv_path = os.path.join(project_root, 'config', '.env')
load_dotenv(dotenv_path)

def load_config():
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

@sleep_and_retry
@limits(calls=50, period=60)
def rate_limited_api_call(client, model, max_tokens, temperature, system, messages):
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages
    )

def analyze_with_claude(data, tickers):
    config = load_config()
    tickers = config['tickers']

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment variables")
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    tickers_str = ', '.join(tickers)
    system_prompt = f"""
    You are an AI financial analyst tasked with processing market sentiment data for machine learning input. Analyze the given data from news articles about stocks and provide a structured output.
    Focus only on the following tickers: {tickers_str}. These tickers are specified in the config.yaml file.
    For each of these tickers, follow these instructions:
    1. Use the provided sentiment scores, but also analyze the text to provide a confidence level (0-1) for each sentiment score based on the content of the articles.
    2. Identify the top 5 most important topics for each stock, assigning a relevance score (0-1) to each. Explain briefly why each topic is relevant.
    3. Detect any significant changes in sentiment over time, providing specific percentage changes. Analyze potential reasons for these changes based on the article content.
    4. Extract and quantify any numerical financial metrics mentioned (e.g., revenue growth, EPS). If metrics are mentioned without specific numbers, estimate a range based on the context.
    5. Identify the top 5 market drivers, assigning an impact score (0-1) to each. Provide a brief explanation for each driver's importance.
    6. Provide a numerical comparison (e.g., sentiment difference, topic overlap percentage) between stocks. Additionally, analyze any noteworthy relationships or correlations between the stocks based on the article content.
    7. Based on the overall analysis, provide a short-term (1-3 months) and long-term (6-12 months) outlook for each stock, with a confidence level (0-1) for each prediction.
    Format your output as a JSON object with the following structure:
    {{
    "stocks": {{
    "TICKEREXAMPLE": {{
    "sentiment_score": number,
    "sentiment_confidence": number,
    "key_topics": [{{"topic": string, "relevance": number, "explanation": string}}],
    "sentiment_change": {{"time_period": string, "change_percentage": number, "analysis": string}},
    "financial_metrics": {{"metric_name": number or {{"min": number, "max": number}}}},
    "short_term_outlook": {{"prediction": string, "confidence": number}},
    "long_term_outlook": {{"prediction": string, "confidence": number}}
    }}
    }},
    "market_drivers": [{{"driver": string, "impact_score": number, "explanation": string}}],
    "stock_comparison": {{
    "sentiment_difference": number,
    "topic_overlap_percentage": number,
    "relationship_analysis": string
    }}
    }}
    Ensure all data is numerical where possible. Do not include any text outside of the JSON object in your response.
    If a ticker from the config is not present in the input data, include it in the output with empty or zero values.
    """

    max_retries = 5
    base_delay = 60
    all_analyses = {}

    ticker_batches = [tickers[i:i+3] for i in range(0, len(tickers), 3)]

    for batch_index, batch in enumerate(ticker_batches, 1):
        batch_data = {ticker: data.get(ticker, {}) for ticker in batch}
        
        logger.info(f"Processing batch {batch_index}/{len(ticker_batches)}: {', '.join(batch)}")
        
        for attempt in range(max_retries):
            try:
                message = rate_limited_api_call(
                    client,
                    model="claude-3-sonnet-20240229",
                    max_tokens=4000,
                    temperature=0,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": json.dumps(batch_data)}
                    ]
                )
                
                cleaned_content = message.content[0].text.replace('\x00', '').strip()
                start = cleaned_content.find('{')
                end = cleaned_content.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = cleaned_content[start:end]
                    analysis = json.loads(json_str)
                    
                    for ticker in batch:
                        if ticker not in analysis['stocks']:
                            logger.warning(f"Ticker {ticker} not found in Claude's analysis. Adding empty data.")
                            analysis['stocks'][ticker] = {
                                "sentiment_score": 0,
                                "sentiment_confidence": 0,
                                "key_topics": [],
                                "sentiment_change": {"time_period": "N/A", "change_percentage": 0, "analysis": "No data available"},
                                "financial_metrics": {},
                                "short_term_outlook": {"prediction": "No data", "confidence": 0},
                                "long_term_outlook": {"prediction": "No data", "confidence": 0}
                            }
                    
                    for ticker in batch:
                        all_analyses[ticker] = analysis['stocks'].get(ticker, {})
                    
                    logger.info(f"Successfully processed batch {batch_index}")
                    
                    if batch_index < len(ticker_batches):
                        delay = min(300, base_delay * (2 ** attempt) + random.uniform(0, 30))
                        logger.info(f"Waiting for {delay:.2f} seconds before next batch...")
                        time.sleep(delay)
                    break
                else:
                    logger.error("No valid JSON object found in the response")
                    return None
            except Exception as e:
                if "rate_limit_error" in str(e) or "429" in str(e):
                    delay = min(300, base_delay * (2 ** attempt) + random.uniform(0, 30))
                    logger.warning(f"Rate limit hit for batch {batch_index}. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Error calling Claude API for batch {batch_index}: {str(e)}")
                    raise
        else:
            logger.error(f"Failed to process batch {batch_index} after {max_retries} attempts")
    
    logger.info("All batches processed. Combining results...")

    if not all_analyses:
        logger.error("No analyses were successfully completed.")
        return None

    combined_analysis = {
        "stocks": all_analyses,
        "market_drivers": [],  # You may need to aggregate this from individual analyses
        "stock_comparison": {}  # You may need to compute this based on all ticker data
    }

    return combined_analysis

def get_latest_sentiment_date(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT MAX(date) FROM sentiment_stock_sentiment")
        latest_date = cursor.fetchone()[0]
        return datetime.strptime(latest_date, '%Y-%m-%d').date() if latest_date else None
    except sqlite3.OperationalError:
        # table doesn't exist yet
        return None
    finally:
        conn.close()

def main():
    try:
        db_path = os.path.join(project_root, 'data', 'processed', 'market_data.db')
        latest_db_date = get_latest_sentiment_date(db_path)
        today = datetime.now().date()
        
        raw_data_dir = os.path.join(project_root, 'data', 'raw', 'sentiment')
        json_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.json')]
        
        if not json_files:
            logger.error("No JSON files found in the raw data directory.")
            return

        date_pattern = r'\d{8}'
        latest_json_file = max(json_files, key=lambda x: re.search(date_pattern, x).group() if re.search(date_pattern, x) else '00000000')
        date_match = re.search(date_pattern, latest_json_file)
        
        if not date_match:
            logger.error(f"Could not extract date from filename: {latest_json_file}")
            return

        latest_json_date = datetime.strptime(date_match.group(), '%Y%m%d').date()

        if latest_db_date is None or latest_db_date < latest_json_date:
            if latest_json_date == today:
                logger.info(f"Loading latest JSON from {raw_data_dir}")
                with open(os.path.join(raw_data_dir, latest_json_file), 'r') as f:
                    data = json.load(f)

                config = load_config()
                tickers = config['tickers']
                logger.info(f"Analyzing the following tickers: {', '.join(tickers)}")

                logger.info("Starting analysis with Claude")
                analysis = analyze_with_claude(data, tickers)

                if analysis:
                    processed_data_dir = os.path.join(project_root, 'data', 'raw', 'sentiment', 'claude')
                    os.makedirs(processed_data_dir, exist_ok=True)
                    output_file = f'claude_analysis_{today.strftime("%Y%m%d")}.json'
                    output_path = os.path.join(processed_data_dir, output_file)
                    
                    logger.info(f"Saving analysis to {output_path}")
                    with open(output_path, 'w') as f:
                        json.dump(analysis, f, indent=2)

                    logger.info(f"Analysis completed and saved to {output_file}")
                else:
                    logger.error("Analysis failed or returned no results.")
            else:
                logger.info(f"Latest JSON file is not from today. Latest date: {latest_json_date}")
        else:
            logger.info(f"Sentiment data is up to date. Latest date in DB: {latest_db_date}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
