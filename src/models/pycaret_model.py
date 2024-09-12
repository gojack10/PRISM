import pandas as pd
from pycaret.regression import *
from data_loader import load_data
from feature_engineering import engineer_features

def run_pycaret_model():
    # Load data
    data = load_data()
    
    # Apply feature engineering
    data = engineer_features(data)
    
    # Initialize PyCaret setup
    setup(data=data, target='close', session_id=42)
    
    # Compare models
    best_model = compare_models()
    
    # Tune the best model
    tuned_model = tune_model(best_model)
    
    # Evaluate the model
    evaluate_model(tuned_model)
    
    # Plot model results
    plot_model(tuned_model, plot='feature')
    plot_model(tuned_model, plot='residuals')
    
    # Finalize the model
    final_model = finalize_model(tuned_model)
    
    # Save the model
    save_model(final_model, 'pycaret_model')

if __name__ == "__main__":
    run_pycaret_model()
