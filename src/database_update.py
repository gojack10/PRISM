import os
import logging
import yaml
from datetime import datetime
import subprocess
import sys
from tqdm import tqdm
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
config_path = os.path.join(project_root, 'config', 'config.yaml')

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

STOCK_TICKERS = config['tickers']

def data_exists_for_date(date, data_type):
    data_dir = os.path.join(project_root, 'data', 'raw', data_type)
    date_str = date.strftime('%Y-%m-%d')
    for file in os.listdir(data_dir):
        if date_str in file:
            return True
    return False

def get_latest_data_date(data_type):
    data_dir = os.path.join(project_root, 'data', 'raw', data_type)
    dates = []
    for file in os.listdir(data_dir):
        try:
            date = datetime.strptime(file.split('_')[1], '%Y-%m-%d')
            dates.append(date)
        except (ValueError, IndexError):
            continue
    return max(dates) if dates else None

def run_script(script_name):
    script_path = os.path.join(current_dir, 'data_collection', script_name)
    logging.info(f"Running {script_name}")
    print(f"\nExecuting {script_name}...")
    try:
        process = subprocess.Popen([sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        for line in process.stdout:
            print(line.strip())
        process.wait()
        if process.returncode == 0:
            logging.info(f"{script_name} completed successfully")
            print(f"{script_name} completed successfully")
            return True
        else:
            logging.error(f"Error running {script_name}: {process.stderr.read()}")
            print(f"Error running {script_name}. Check logs for details.")
            return False
    except Exception as e:
        logging.error(f"Exception running {script_name}: {str(e)}")
        print(f"Exception running {script_name}. Check logs for details.")
        return False

def main():
    today = datetime.now().date()
    
    # List of scripts to run
    scripts = [
        ('overview', 'alpha_vantage_overview.py'),
        ('intraday', 'alpha_vantage_intraday.py'),
        ('indicators', 'alpha_vantage_indicators.py'),
        ('sentiment', 'alpha_vantage_sentiment.py'),
        ('sentiment_analysis', 'sentiment_json_to_claude.py'),
        ('database_update', 'database_builder.py')
    ]
    
    print("Starting PRISM database update process...")
    
    with tqdm(total=len(scripts), desc="Overall Progress", position=0) as pbar:
        for data_type, script in scripts:
            if data_type in ['sentiment_analysis', 'database_update'] or not data_exists_for_date(today, data_type):
                if run_script(script):
                    if data_type not in ['sentiment_analysis', 'database_update']:
                        latest_date = get_latest_data_date(data_type)
                        if latest_date:
                            print(f"Updated {data_type} data. Latest date: {latest_date}")
                        else:
                            print(f"Warning: Failed to get latest date for {data_type}")
                else:
                    print(f"Error: Failed to update {data_type} data. Stopping execution.")
                    return
            else:
                print(f"Data for {data_type} already exists for today. Skipping.")
            
            pbar.update(1)
            time.sleep(0.2)
    
    print("\nAll data collection and processing completed successfully.")

if __name__ == "__main__":
    main()
