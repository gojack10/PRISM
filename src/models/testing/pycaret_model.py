from pycaret.time_series import *
from data_loader import load_data
import pandas as pd

def run_pycaret_model():
    try:
        # Load data
        data = load_data()
        
        # Verify data sufficiency
        if data.index.nunique() < 50:
            print("Warning: Not enough unique dates for time series modeling.")
            return

        print("Data loaded successfully. Shape:", data.shape)
        print("Columns:", data.columns.tolist())
        print("Index:", data.index.name)
        print("First few rows:")
        print(data.head())

        # Initialize PyCaret setup for time series
        s = setup(
            data=data,
            target='close',
            index='date',
            group_id='ticker',
            fh=10,  # Adjust forecast horizon as appropriate
            session_id=42,
            fold=3,
            transform_target=True,
            numeric_imputation_target='mean',
            numeric_imputation_exogenous='mean',
            ignore_features=['key_topics', 'sentiment_change', 'financial_metrics', 'short_term_outlook', 'long_term_outlook'],
            verbose=True
        )

        # Compare models
        best_model = compare_models()

        if best_model is None:
            print("Error: No valid model found during comparison.")
            return

        print("Best model:", best_model)

        # Tune the best model
        tuned_model = tune_model(best_model)

        # Evaluate the model
        evaluate_model(tuned_model)

        # Plot model results
        plot_model(tuned_model, plot='forecast')
        plot_model(tuned_model, plot='decomp_stl')  # Decomposition plot

        # Finalize the model
        final_model = finalize_model(tuned_model)

        # Make future predictions
        future_predictions = predict_model(final_model, fh=10)  # Adjust fh as needed
        print("Future predictions:")
        print(future_predictions)

        # Save the model
        save_model(final_model, 'pycaret_ts_model')
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pycaret_model()
