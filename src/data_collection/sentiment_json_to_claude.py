import json
import os
import logging
from dotenv import load_dotenv
import anthropic
from datetime import datetime
import yaml

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

SYSTEM_PROMPT = """
You are an AI financial analyst tasked with processing market sentiment data for machine learning input. Analyze the given data from news articles about stocks and provide a structured output.

Follow these instructions:

1. Calculate precise sentiment scores on a scale of -1 to 1 for each stock.
2. Identify the top 5 most important topics for each stock, assigning a relevance score (0-1) to each.
3. Detect any significant changes in sentiment over time, providing specific percentage changes.
4. Extract and quantify any numerical financial metrics mentioned (e.g., revenue growth, EPS).
5. Identify the top 5 market drivers, assigning an impact score (0-1) to each.
6. Provide a numerical comparison (e.g., sentiment difference, topic overlap percentage) between stocks.

Format your output as a JSON object with the following structure:

{
  "stocks": {
    "EXAMPLETICKER1": {
      "sentiment_score": number,
      "key_topics": [{"topic": string, "relevance": number}],
      "sentiment_change": {"time_period": string, "change_percentage": number},
      "financial_metrics": {"metric_name": number}
    },
    "EXAMPLETICKER2": {
      "sentiment_score": number,
      "key_topics": [{"topic": string, "relevance": number}],
      "sentiment_change": {"time_period": string, "change_percentage": number},
      "financial_metrics": {"metric_name": number}
    },
    "EXAMPLETICKER3": {
      "sentiment_score": number,
      "key_topics": [{"topic": string, "relevance": number}],
      "sentiment_change": {"time_period": string, "change_percentage": number},
      "financial_metrics": {"metric_name": number}
    }
  },
  "market_drivers": [{"driver": string, "impact_score": number}],
  "stock_comparison": {"sentiment_difference": number, "topic_overlap_percentage": number}
}

Ensure all data is numerical where possible. Do not include any text outside of the JSON object in your response.
"""

def load_config():
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

# send to claude
def analyze_with_claude(data):
    config = load_config()
    tickers = config['tickers']

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment variables")
        raise ValueError("ANTHROPIC_API_KEY not set")

    logger.info(f"Initializing Anthropic client with API key: {api_key[:5]}...")
    client = anthropic.Anthropic(api_key=api_key)

    # modify the system prompt to include the tickers from config.yaml
    system_prompt = SYSTEM_PROMPT + f"\n\nAnalyze the following tickers: {', '.join(tickers)}. These tickers are specified in the config.yaml file."

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
                return json.loads(json_str)
            else:
                logger.error("No valid JSON object found in the response")
                return None
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse JSON response: {str(json_err)}")
            logger.debug(f"Response content causing JSON parse error: {cleaned_content}")
            return None
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        raise

def main():
    try:
        
        raw_data_dir = os.path.join(project_root, 'data', 'raw')
        logger.info(f"Loading latest JSON from {raw_data_dir}")
        data = load_latest_json(raw_data_dir)

        logger.info("Starting analysis with Claude")
        analysis = analyze_with_claude(data)

        processed_data_dir = os.path.join(project_root, 'data', 'raw')
        os.makedirs(processed_data_dir, exist_ok=True)
        output_file = f'claude_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        output_path = os.path.join(processed_data_dir, output_file)
        
        logger.info(f"Saving analysis to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Analysis completed and saved to {output_file}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
