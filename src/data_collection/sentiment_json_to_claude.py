import json
import os
import logging
from dotenv import load_dotenv
import anthropic
from datetime import datetime, timedelta
import yaml
import sqlite3
import re
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
dotenv_path = os.path.join(project_root, 'config', '.env')
load_dotenv(dotenv_path)

def load_latest_json(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    latest_file = max(json_files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    with open(os.path.join(directory, latest_file), 'r') as file:
        return json.load(file)

def load_config():
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def analyze_with_claude(data):
    config = load_config()
    tickers = config['tickers']

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment variables")
        raise ValueError("ANTHROPIC_API_KEY not set")

    logger.info(f"Initializing Anthropic client with API key: {api_key[:5]}...")
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
    base_delay = 20  
    
    for attempt in range(max_retries):
        try:
            logger.info("Sending request to Claude API...")
            message = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": json.dumps(data)}
                ]
            )
            logger.info("Received response from Claude API")
            
            # log the raw response content
            logger.debug(f"Raw response content: {message.content}")
            
            # check if the content is empty
            if not message.content:
                logger.error("Received empty content from Claude API")
                return None
            try:
                # remove null characters and leading/trailing whitespace
                cleaned_content = message.content[0].text.replace('\x00', '').strip()
                # find the first '{' and last '}' to extract the JSON object
                start = cleaned_content.find('{')
                end = cleaned_content.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = cleaned_content[start:end]
                    analysis = json.loads(json_str)
                    
                    # Ensure all tickers from the config are present in the analysis
                    for ticker in tickers:
                        if ticker not in analysis['stocks']:
                            logger.warning(f"Ticker {ticker} not found in Claude's analysis. Adding empty data.")
                            analysis['stocks'][ticker] = {
                                "sentiment_score": 0,
                                "key_topics": [],
                                "sentiment_change": {"time_period": "N/A", "change_percentage": 0},
                                "financial_metrics": {}
                            }
                    
                    return analysis
                else:
                    logger.error("No valid JSON object found in the response")
                    return None
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse JSON response: {str(json_err)}")
                logger.debug(f"Response content causing JSON parse error: {cleaned_content}")
                return None
        except Exception as e:
            if "rate_limit_error" in str(e):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 10)
                logger.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Error calling Claude API: {str(e)}")
                raise
    
    logger.error("Max retries reached. Unable to complete API call.")
    raise Exception("Max retries reached for Claude API call")

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

                logger.info("Starting analysis with Claude")
                analysis = analyze_with_claude(data)

                processed_data_dir = os.path.join(project_root, 'data', 'raw', 'sentiment', 'claude')
                os.makedirs(processed_data_dir, exist_ok=True)
                output_file = f'claude_analysis_{today.strftime("%Y%m%d")}.json'
                output_path = os.path.join(processed_data_dir, output_file)
                
                logger.info(f"Saving analysis to {output_path}")
                with open(output_path, 'w') as f:
                    json.dump(analysis, f, indent=2)

                logger.info(f"Analysis completed and saved to {output_file}")
            else:
                logger.info(f"Latest JSON file is not from today. Latest date: {latest_json_date}")
        else:
            logger.info(f"Sentiment data is up to date. Latest date in DB: {latest_db_date}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
