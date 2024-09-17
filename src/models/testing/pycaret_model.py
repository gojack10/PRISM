import os
import time
import logging
from pycaret.time_series import *
from data_loader import load_data_parallel as load_data
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pycaret_model_for_ticker(ticker, data):
    """
    Sets up and trains a PyCaret model for a single ticker.
    """
    try:
        logging.info(f"Initializing PyCaret setup for ticker: {ticker}")
        setup_start = time.time()
        
        s = setup(
            data=data,
            target='4. close',
            fh=10,
            session_id=42,
            fold=3,
            numeric_imputation_target='mean',
            numeric_imputation_exogenous='mean',
            ignore_features=[
                'key_topics', 'sentiment_change', 'financial_metrics',
                'short_term_outlook', 'long_term_outlook', 'ticker'
            ],
            n_jobs=-1,      # Utilize all CPU cores for parallel processing
            use_gpu=True,   
            verbose=True
        )
        logging.info(f"Setup completed in {time.time() - setup_start:.2f} seconds for ticker {ticker}")

        # Log selected features
        processed_data = get_config('X')
        logging.info(f"Features used for modeling: {processed_data.columns.tolist()}")
       

        logging.info("\nComparing models...")
        compare_start = time.time()
        best_model = compare_models(turbo=True, verbose=False)
        logging.info(f"Model comparison completed in {time.time() - compare_start:.2f} seconds for ticker {ticker}")

        if best_model is None:
            logging.error(f"No valid model found during comparison for ticker {ticker}.")
            return

        logging.info(f"Best model for {ticker}: {best_model}")

        logging.info("\nTuning the best model...")
        tune_start = time.time()
        tuned_model = tune_model(best_model)
        logging.info(f"Model tuning completed in {time.time() - tune_start:.2f} seconds for ticker {ticker}")

        logging.info("\nFinalizing the model...")
        final_model = finalize_model(tuned_model)

        logging.info("\nMaking future predictions...")
        future_predictions = predict_model(final_model, fh=10)
        logging.info("Future predictions:")
        logging.info(f"\n{future_predictions}")

        logging.info("\nSaving the model...")
        model_filename = f'pycaret_ts_model_{ticker}'
        save_model(final_model, model_filename)
        logging.info(f"Model saved as '{model_filename}'")

        # Generate plots for the final model only
        logging.info("\nGenerating model plots...")
        plot_model(final_model, plot='forecast')
        plot_model(final_model, plot='decomp_stl')
        logging.info("Plots generated. Check your working directory for saved images.")

    except Exception as e:
        logging.error(f"An error occurred for ticker {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()

def run_pycaret_models():
    """
    Loads the combined data and runs PyCaret models for each ticker separately.
    """
    try:
        logging.info("Starting PyCaret models for all tickers...")
        start_time = time.time()

        logging.info("Loading data from data_loader module...")
        combined_data = load_data()
        logging.info(f"Data loaded successfully. Shape: {combined_data.shape}")

        logging.info("Preprocessing data...")

        # parse 'date' column assuming it's in Unix timestamp (seconds)
        combined_data['date'] = pd.to_datetime(combined_data['date'], unit='s')
        logging.info("Converted 'date' column to datetime.")

        # Check for duplicate dates within each ticker
        duplicate_counts = combined_data.duplicated(subset=['ticker', 'date']).sum()
        if duplicate_counts > 0:
            logging.warning(f"Found {duplicate_counts} duplicate 'date' entries across all tickers. Removing duplicates.")
            combined_data = combined_data.drop_duplicates(subset=['ticker', 'date'], keep='first')
            logging.info(f"Data shape after removing duplicates: {combined_data.shape}")
        else:
            logging.info("No duplicate 'date' entries found across tickers.")

        logging.info(f"Data ready. Columns: {combined_data.columns.tolist()}")
        logging.info("First few rows:")
        logging.info(f"\n{combined_data.head()}")

        # Iterate through each ticker and model separately
        tickers = combined_data['ticker'].unique()
        for ticker in tickers:
            logging.info(f"\nProcessing ticker: {ticker}")
            ticker_data = combined_data[combined_data['ticker'] == ticker].copy()
            
            # Ensure 'date' is sorted
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            logging.info(f"Data for ticker {ticker} sorted by 'date'.")

            # Set 'date' as index and define frequency
            ticker_data.set_index('date', inplace=True)
            ticker_data = ticker_data.asfreq('H')  # Replace 'H' with your data's frequency

            # Optionally, handle missing data after setting frequency
            # ticker_data = ticker_data.fillna(method='ffill')  # Forward fill as an example

            run_pycaret_model_for_ticker(ticker, ticker_data)

        logging.info(f"\nAll PyCaret models completed in {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logging.error(f"An error occurred during PyCaret models run: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pycaret_models()
