from data_loader import load_data
from model import train_model, plot_feature_importance
from eval import evaluate_model, visualize_feature_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np
import pandas as pd
from tqdm import tqdm
from feature_engineering import engineer_features
import logging
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_features_xgboost(X, y, num_features):
    model = XGBRegressor()
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    
    # Select top features
    selected_features = feature_importances.head(num_features).index.tolist()
    
    return selected_features

def select_features_rfe(X, y, num_features):
    model = XGBRegressor()
    rfe = RFE(estimator=model, n_features_to_select=num_features, step=1)
    rfe = rfe.fit(X, y)
    
    selected_features = X.columns[rfe.support_].tolist()
    
    return selected_features

def get_run_folder():
    return f"run {datetime.now().strftime('%Y-%m-%d %H:%M')}"

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
        
        # Log sentiment-related features
        sentiment_features = [col for col in X_engineered.columns if 'sentiment' in col]
        logger.info("\nSentiment-related features:")
        logger.info(sentiment_features)
        
        # Ensure all columns are numeric
        non_numeric_columns = X_engineered.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_columns) > 0:
            logger.warning(f"Dropping non-numeric columns: {non_numeric_columns}")
            X_engineered = X_engineered.select_dtypes(include=[np.number])
        
        # Check if sentiment features were accidentally dropped
        remaining_sentiment_features = [col for col in X_engineered.columns if 'sentiment' in col]
        if len(remaining_sentiment_features) < len(sentiment_features):
            logger.warning("Some sentiment features were dropped during non-numeric column removal.")
            logger.warning(f"Dropped sentiment features: {set(sentiment_features) - set(remaining_sentiment_features)}")
        
        logger.info("\nFinal data types:")
        logger.info(X_engineered.dtypes)
        logger.info(f"\nNumber of features: {X_engineered.shape[1]}")
        
        # Ensure y is aligned with X_engineered
        y = y.loc[X_engineered.index]
        
        # Initialize TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Feature selection
        num_features = X_engineered.shape[1] // 2  # Select half of the features
        selected_features_xgb = select_features_xgboost(X_engineered, y, num_features)
        selected_features_rfe = select_features_rfe(X_engineered, y, num_features)

        # Train and evaluate model with all features
        results_all = train_and_evaluate(X_engineered, y, tscv)

        # Train and evaluate model with XGBoost selected features
        results_xgb = train_and_evaluate(X_engineered[selected_features_xgb], y, tscv)

        # Train and evaluate model with RFE selected features
        results_rfe = train_and_evaluate(X_engineered[selected_features_rfe], y, tscv)

        # Compare results
        logger.info("\nResults with all features:")
        log_results(results_all)

        logger.info("\nResults with XGBoost selected features:")
        log_results(results_xgb)

        logger.info("\nResults with RFE selected features:")
        log_results(results_rfe)

        # Create a new folder for this run
        run_folder = get_run_folder()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prism_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        output_dir = os.path.join(prism_dir, 'graph-output', run_folder)
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created new output directory: {output_dir}")
        except OSError as e:
            logger.error(f"Error creating output directory: {e}")
            raise

        # Visualize feature importances
        avg_feature_importance = pd.concat(results_all['feature_importances'], axis=1).mean(axis=1).sort_values(ascending=False)
        visualize_feature_importance(avg_feature_importance, "Feature Importance - All Features", output_dir)
        
        xgb_importance = pd.Series(dict(zip(selected_features_xgb, avg_feature_importance[selected_features_xgb])))
        visualize_feature_importance(xgb_importance, "Feature Importance - XGBoost Selected", output_dir)
        
        rfe_importance = pd.Series(dict(zip(selected_features_rfe, avg_feature_importance[selected_features_rfe])))
        visualize_feature_importance(rfe_importance, "Feature Importance - RFE Selected", output_dir)

        logger.info(f"Feature importance plots have been saved to the '{output_dir}' directory.")

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

def train_and_evaluate(X, y, tscv):
    train_scores = []
    test_scores = []
    feature_importances = []

    for train_index, test_index in tqdm(tscv.split(X), desc="Cross-validation", total=5):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model_tuple = train_model(X_train, y_train)
        model = model_tuple[0]

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        train_scores.append(train_rmse)
        test_scores.append(test_rmse)

        importance = model.feature_importances_
        feature_importances.append(pd.Series(importance, index=X_train.columns))

    return {
        'train_scores': train_scores,
        'test_scores': test_scores,
        'feature_importances': feature_importances
    }

def log_results(results):
    logger.info(f"Average Train RMSE: {np.mean(results['train_scores']):.4f} (+/- {np.std(results['train_scores']):.4f})")
    logger.info(f"Average Test RMSE: {np.mean(results['test_scores']):.4f} (+/- {np.std(results['test_scores']):.4f})")

    avg_feature_importance = pd.concat(results['feature_importances'], axis=1).mean(axis=1).sort_values(ascending=False)
    logger.info("\nTop 10 features:")
    logger.info(avg_feature_importance.head(10))

if __name__ == "__main__":
    run_model()
