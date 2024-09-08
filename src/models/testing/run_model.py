from data_loader import load_data
from model import train_model, plot_feature_importance
from eval import evaluate_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np
import pandas as pd
from tqdm import tqdm
from feature_engineering import engineer_features
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_model():
    try:
        # Load data
        data_tuple = load_data()
        
        # Check if data is a tuple and extract the DataFrame
        if isinstance(data_tuple, tuple):
            data = data_tuple[0]  # Assuming the DataFrame is the first element of the tuple
        else:
            data = data_tuple
        
        logger.info("Original data columns:")
        logger.info(data.columns)
        logger.info("\nFirst few rows of original data:")
        logger.info(data.head())
        
        # Sort data by date
        data = data.sort_values(['ticker', 'date'])
        
        # Check if 'close' is in the data
        if 'close' not in data.columns:
            logger.error("'close' column not found in the input data. Cannot proceed with model training.")
            return
        
        # Separate features and target
        X = data.drop(['close'], axis=1)
        y = data['close']
        
        logger.info("\nColumns in X before feature engineering:")
        logger.info(X.columns)
        logger.info("\nFirst few rows of X:")
        logger.info(X.head())
        
        # Apply feature engineering
        X_engineered = engineer_features(X)
        logger.info("\nColumns in X_engineered after feature engineering:")
        logger.info(X_engineered.columns)
        logger.info("\nFirst few rows of X_engineered:")
        logger.info(X_engineered.head())
        
        # Ensure all columns are numeric
        non_numeric_columns = X_engineered.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_columns) > 0:
            logger.warning(f"Dropping non-numeric columns: {non_numeric_columns}")
            X_engineered = X_engineered.select_dtypes(include=[np.number])
        
        logger.info("\nFinal data types:")
        logger.info(X_engineered.dtypes)
        logger.info(f"\nNumber of features: {X_engineered.shape[1]}")
        
        # Ensure y is aligned with X_engineered
        y = y.loc[X_engineered.index]
        
        # Initialize TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize lists to store results
        train_scores = []
        test_scores = []
        feature_importances = []
        
        # Perform cross-validation
        for train_index, test_index in tqdm(tscv.split(X_engineered), desc="Cross-validation", total=5):
            X_train, X_test = X_engineered.iloc[train_index], X_engineered.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train model
            model_tuple = train_model(X_train, y_train)
            model = model_tuple[0]  # Assuming the model is the first element of the tuple
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate scores
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            train_scores.append(train_rmse)
            test_scores.append(test_rmse)
            
            # Get feature importance
            importance = model.feature_importances_
            feature_importances.append(pd.Series(importance, index=X_train.columns))
        
        # Print average scores
        logger.info(f"Average Train RMSE: {np.mean(train_scores):.4f} (+/- {np.std(train_scores):.4f})")
        logger.info(f"Average Test RMSE: {np.mean(test_scores):.4f} (+/- {np.std(test_scores):.4f})")
        
        # Plot average feature importance
        avg_feature_importance = pd.concat(feature_importances, axis=1).mean(axis=1).sort_values(ascending=False)
        plot_feature_importance(avg_feature_importance.values, avg_feature_importance.index)
        
        # Train final model on all data
        final_model_tuple = train_model(X_engineered, y)
        final_model = final_model_tuple[0]  # Assuming the model is the first element of the tuple
        
        # Evaluate final model
        final_predictions = final_model.predict(X_engineered)
        final_rmse = np.sqrt(mean_squared_error(y, final_predictions))
        final_mae = mean_absolute_error(y, final_predictions)
        final_r2 = r2_score(y, final_predictions)
        final_mape = mean_absolute_percentage_error(y, final_predictions)
        
        logger.info("\nFinal Model Performance:")
        logger.info(f"RMSE: {final_rmse:.4f}")
        logger.info(f"MAE: {final_mae:.4f}")
        logger.info(f"R2 Score: {final_r2:.4f}")
        logger.info(f"MAPE: {final_mape:.4f}")
        
        # Plot final feature importance
        plot_feature_importance(final_model.feature_importances_, X_engineered.columns)

    except Exception as e:
        logger.error(f"An error occurred during model execution: {str(e)}")
        raise

if __name__ == "__main__":
    run_model()
