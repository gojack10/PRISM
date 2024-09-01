import os
import json
import pandas as pd
import sqlite3
from datetime import datetime

def build_and_save_sentiment_dataframe():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    db_path = os.path.join(processed_data_dir, 'sentiment_data.db')

    # Ensure the processed directory exists
    os.makedirs(processed_data_dir, exist_ok=True)

    # Get all claude_analysis JSON files
    json_files = [f for f in os.listdir(raw_data_dir) if f.startswith('claude_analysis_') and f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError("No claude_analysis JSON files found in the raw data directory.")
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    for json_file in json_files:
        json_path = os.path.join(raw_data_dir, json_file)
        with open(json_path, 'r') as file:
            data = json.load(file)
            file_date = datetime.strptime(json_file.split('_')[2].split('.')[0], "%Y%m%d")
            
            # Process stocks data
            stocks_data = []
            for stock, stock_data in data['stocks'].items():
                stock_info = {
                    'file_date': file_date,
                    'stock': stock,
                    'sentiment_score': stock_data['sentiment_score'],
                    'sentiment_change_period': stock_data['sentiment_change']['time_period'],
                    'sentiment_change_percentage': stock_data['sentiment_change']['change_percentage']
                }
                for topic in stock_data['key_topics']:
                    stock_info[f"topic_{topic['topic']}"] = topic['relevance']
                stocks_data.append(stock_info)
            
            # Save stocks data
            df_stocks = pd.DataFrame(stocks_data)
            df_stocks.to_sql('stocks_sentiment', conn, if_exists='append', index=False)
            
            # Process market drivers data
            market_drivers_data = [{**driver, 'file_date': file_date} for driver in data['market_drivers']]
            df_drivers = pd.DataFrame(market_drivers_data)
            df_drivers.to_sql('market_drivers', conn, if_exists='append', index=False)
            
            # Process stock comparison data
            stock_comparison_data = {**data['stock_comparison'], 'file_date': file_date}
            df_comparison = pd.DataFrame([stock_comparison_data])
            df_comparison.to_sql('stock_comparison', conn, if_exists='append', index=False)
        
        print(f"Processed and saved data from {json_file}")

    # After processing all files, read and sort each table
    tables = ['stocks_sentiment', 'market_drivers', 'stock_comparison']
    for table in tables:
        df = pd.read_sql(f'SELECT * FROM {table}', conn)
        df = df.sort_values('file_date')
        df.to_sql(table, conn, if_exists='replace', index=False)
    
    conn.close()
    
    return pd.read_sql('SELECT * FROM stocks_sentiment', sqlite3.connect(db_path))

if __name__ == "__main__":
    sentiment_df = build_and_save_sentiment_dataframe()
    print(sentiment_df.head())
    print(f"DataFrame shape: {sentiment_df.shape}")
    print(f"Date range: {sentiment_df['file_date'].min()} to {sentiment_df['file_date'].max()}")
