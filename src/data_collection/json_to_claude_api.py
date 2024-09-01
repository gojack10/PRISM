import json
import os
import logging
from dotenv import load_dotenv
import anthropic
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
dotenv_path = os.path.join(project_root, 'config', '.env')
load_dotenv(dotenv_path)

# Load JSON Data
def load_latest_json(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    latest_file = max(json_files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    with open(os.path.join(directory, latest_file), 'r') as file:
        return json.load(file)

SYSTEM_PROMPT = """
You are an AI financial analyst tasked with processing market sentiment data for machine learning input. Analyze the given data from news articles about stocks and provide a structured output. Follow these instructions:

1. Calculate precise sentiment scores on a scale of -1 to 1 for each stock.
2. Identify the top 5 most important topics for each stock, assigning a relevance score (0-1) to each.
3. Detect any significant changes in sentiment over time, providing specific percentage changes.
4. Extract and quantify any numerical financial metrics mentioned (e.g., revenue growth, EPS).
5. Identify the top 5 market drivers, assigning an impact score (0-1) to each.
6. Provide a numerical comparison (e.g., sentiment difference, topic overlap percentage) between stocks.

Format your output as a JSON object with the following structure:

{
  "stocks": {
    "AAPL": {
      "sentiment_score": number,
      "key_topics": [{"topic": string, "relevance": number}],
      "sentiment_change": {"time_period": string, "change_percentage": number},
      "financial_metrics": {"metric_name": number}
    },
    "NVDA": {
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

# Send to Claude and Process Response
def analyze_with_claude(data):
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment variables")
        raise ValueError("ANTHROPIC_API_KEY not set")

    logger.info(f"Initializing Anthropic client with API key: {api_key[:5]}...")
    client = anthropic.Anthropic(api_key=api_key)

    try:
        logger.info("Sending request to Claude API...")
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": json.dumps(data)}
            ]
        )
        logger.info("Received response from Claude API")
        return json.loads(message.content[0].text)
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        raise

# Main function
def main():
    try:
        # Load data
        raw_data_dir = os.path.join(project_root, 'data', 'raw')
        logger.info(f"Loading latest JSON from {raw_data_dir}")
        data = load_latest_json(raw_data_dir)

        # Analyze with Claude
        logger.info("Starting analysis with Claude")
        analysis = analyze_with_claude(data)

        # Save processed data
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
