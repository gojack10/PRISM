import os
import time
import logging
from pycaret.time_series import *
from data_loader import load_data
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pycaret_model():
    try:
        logging.info("Starting PyCaret model run...")
        start_time = time.time()

        logging.info("Loading data from data_loader module...")
        data = load_data()
        logging.info(f"Data loaded successfully. Shape: {data.shape}")
        
        logging.info("Preprocessing data...")
        if 'timestamp' in data.columns:
            data = data.rename(columns={'timestamp': 'date'})
            logging.info("Renamed 'timestamp' column to 'date'")
        
        if data.index.name != 'date':
            data = data.set_index('date')
            logging.info("Set 'date' as index")
        
        if data.index.nunique() < 50:
            logging.warning("Not enough unique dates for time series modeling.")
            return

        logging.info(f"Data ready. Columns: {data.columns.tolist()}")
        logging.info(f"Index: {data.index.name}")
        logging.info("First few rows:")
        logging.info(f"\n{data.head()}")

        logging.info("\nInitializing PyCaret setup for time series...")
        setup_start = time.time()
        s = setup(
            data=data,
            target='close',
            index='date',
            group_id='ticker',
            fh=10,
            session_id=42,
            fold=3,
            transform_target=True,
            numeric_imputation_target='mean',
            numeric_imputation_exogenous='mean',
            ignore_features=[
                'key_topics', 'sentiment_change', 'financial_metrics',
                'short_term_outlook', 'long_term_outlook'
            ],
            verbose=True
        )
        logging.info(f"Setup completed in {time.time() - setup_start:.2f} seconds")

        logging.info("\nComparing models...")
        compare_start = time.time()
        best_model = compare_models()
        logging.info(f"Model comparison completed in {time.time() - compare_start:.2f} seconds")

        if best_model is None:
            logging.error("No valid model found during comparison.")
            return

        logging.info(f"Best model: {best_model}")

        logging.info("\nTuning the best model...")
        tune_start = time.time()
        tuned_model = tune_model(best_model)
        logging.info(f"Model tuning completed in {time.time() - tune_start:.2f} seconds")

        logging.info("\nEvaluating the model...")
        evaluate_model(tuned_model)

        logging.info("\nGenerating model plots...")
        plot_model(tuned_model, plot='forecast')
        plot_model(tuned_model, plot='decomp_stl')
        logging.info("Plots generated. Check your working directory for saved images.")

        logging.info("\nFinalizing the model...")
        final_model = finalize_model(tuned_model)

        logging.info("\nMaking future predictions...")
        future_predictions = predict_model(final_model, fh=10)
        logging.info("Future predictions:")
        logging.info(f"\n{future_predictions}")

        logging.info("\nSaving the model...")
        save_model(final_model, 'pycaret_ts_model')
        logging.info("Model saved as 'pycaret_ts_model'")

        logging.info(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pycaret_model()
