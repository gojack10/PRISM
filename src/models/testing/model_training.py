import os
import logging
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.utils.utils import SeasonalityMode, TrendMode, ModelMode
import torch
import matplotlib.pyplot as plt  # Added this line

# Import functions from data_loader.py
from data_loader import (
    load_environment_variables,
    create_db_engine,
    fetch_and_process_data,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def parse_timestamp(ts):
    """
    Attempts to parse a timestamp string into a pandas Timestamp object.
    Supports multiple datetime formats:
    - "%Y-%m-%d %H:%M:%S"
    - "%Y-%m-%d %H:%M"
    - "%Y-%m-%d"

    Parameters:
        ts (str): The timestamp string to parse.

    Returns:
        pd.Timestamp or pd.NaT: The parsed Timestamp or Not-a-Time if parsing fails.
    """
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return pd.to_datetime(ts, format=fmt)
        except ValueError:
            continue
    # If all formats fail, return NaT
    return pd.NaT

def main():
    # Calculate the current directory and project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    dotenv_path = os.path.join(project_root, 'config', '.env')

    # Load environment variables
    load_environment_variables(dotenv_path)

    try:
        # Create database engine
        engine = create_db_engine()
        logging.info(f"Engine type in main: {type(engine)}")  # Added logging

        # Fetch and process all data
        processed_data = fetch_and_process_data(engine)

        # Access the preprocessed intraday data
        df_intraday = processed_data.get('intraday')
        if df_intraday is None:
            logging.error("Intraday data is not available. Exiting.")
            return

        # Verify the frequency
        logging.info(f"Intraday data frequency before setting: {df_intraday.index.freq}")

        # Ensure the frequency is set to hourly
        if df_intraday.index.freq is None:
            df_intraday = df_intraday.asfreq('H')
            logging.info("Intraday data frequency set to hourly ('H').")

        # Create TimeSeries object
        try:
            series = TimeSeries.from_dataframe(
                df_intraday,
                value_cols='close',
                freq='H',  # Ensure the frequency is set to hourly
                fill_missing_dates=True
            )
            logging.info("TimeSeries object created successfully.")
        except ValueError as ve:
            logging.error(f"Error creating TimeSeries: {ve}")
            return

        # Split data into training and validation sets
        train, val = series.split_before(0.8)
        logging.info(f"Training set size: {len(train)}")
        logging.info(f"Validation set size: {len(val)}")

        # Define the model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = NBEATSModel(
            input_chunk_length=30,
            output_chunk_length=7,
            n_epochs=100,
            pl_trainer_kwargs={
                "accelerator": "gpu",       # or "cpu"
                "devices": 1,               # Number of GPUs to use
                "precision": "64-true"      # Use "64-true" for double precision
            }
        )
        logging.info(f"Model initialized on device: {device}")

        # Fit the model
        logging.info("Starting model training...")
        model.fit(train, verbose=True)
        logging.info("Model training completed.")

        # Make forecasts
        logging.info("Starting forecasting...")
        forecast = model.predict(n=7)
        logging.info("Forecasting completed.")

        # Plot the results
        logging.info("Plotting the results.")
        series.plot(label='Actual')
        forecast.plot(label='Forecast')
        plt.legend()
        plt.show()

        # Save the model
        model.save_model("nbeats_model.pth.tar")
        logging.info("Model saved successfully.")

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
