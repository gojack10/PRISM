from data_loader import load_data
from model import train_ticker_model, ConsolidatedModel, train_consolidated_model
from eval import evaluate_model, visualize_feature_importance, evaluate_consolidated_model
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

logging.basicConfig(level=logging.INFO, format='%(message)s')
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

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def train_model(X_train, y_train):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_and_evaluate(X, y, cv):
    results = []
    for fold, (train_index, val_index) in enumerate(cv.split(X), 1):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model = train_model(X_train, y_train)
        
        predictions = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
        mape = mean_absolute_percentage_error(y_val, predictions)
        
        results.append({
            'fold': fold,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'feature_importances': pd.Series(model.feature_importances_, index=X.columns)
        })
    
    return results

def plot_feature_importance(importances, feature_names, output_dir, title="Feature Importance"):
    feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar')
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}.png"))
    plt.close()
    file_name = f"{title.replace(' ', '_')}.png"
    full_path = os.path.join(output_dir, file_name)
    logger.info(f"Feature importance plot saved to: {full_path}")

def run_model():
    try:
        # Create output directory
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        project_root = get_project_root()
        output_dir = os.path.join(project_root, "graph-output", f"run {timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

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
        
        # Group data by ticker
        grouped_data = data.groupby('ticker')

        # Train ticker-specific models and evaluate
        ticker_models = {}
        ticker_performances = {}
        for ticker, ticker_data in grouped_data:
            X_ticker = engineer_features(ticker_data.drop(['close'], axis=1), include_ticker_columns=False)
            y_ticker = ticker_data['close']
            ticker_models[ticker] = train_ticker_model(X_ticker, y_ticker)
            
            # Evaluate ticker-specific model
            predictions = ticker_models[ticker].predict(X_ticker)
            rmse = np.sqrt(mean_squared_error(y_ticker, predictions))
            mae = mean_absolute_error(y_ticker, predictions)
            r2 = r2_score(y_ticker, predictions)
            mape = mean_absolute_percentage_error(y_ticker, predictions)
            da = calculate_directional_accuracy(y_ticker.values, predictions)
            
            ticker_performances[ticker] = {
                "RMSE": rmse,
                "MAE": mae,
                "R²": r2,
                "MAPE": mape,
                "Directional Accuracy": da
            }
            
            # Log performance for each ticker
            logger.info(f"\nModel Performance for {ticker}:")
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"R²: {r2:.4f}")
            logger.info(f"MAPE: {mape:.4f}")
            logger.info(f"Directional Accuracy: {da:.4f}")
            
            # Plot feature importance for each ticker
            plot_feature_importance(ticker_models[ticker].feature_importances_, X_ticker.columns, output_dir, f"{ticker}_Feature_Importance")

        # Output overall performance summary
        logger.info("\nOverall Performance Summary:\n")
        for ticker, performance in ticker_performances.items():
            logger.info(f"{ticker}:")
            for metric, value in performance.items():
                logger.info(f"  {metric}: {value:.4f}")
            logger.info("")  # Add an empty line between tickers

        logger.info(f"All model evaluations and feature importance plots have been saved to the '{output_dir}' directory.")

    except Exception as e:
        logger.error(f"An error occurred during model execution: {str(e)}")
        raise

def log_results(results):
    rmse_scores = [result['rmse'] for result in results]
    mae_scores = [result['mae'] for result in results]
    r2_scores = [result['r2'] for result in results]
    mape_scores = [result['mape'] for result in results]

    logger.info(f"Average RMSE: {np.mean(rmse_scores):.4f} (+/- {np.std(rmse_scores):.4f})")
    logger.info(f"Average MAE: {np.mean(mae_scores):.4f} (+/- {np.std(mae_scores):.4f})")
    logger.info(f"Average R²: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
    logger.info(f"Average MAPE: {np.mean(mape_scores):.4f} (+/- {np.std(mape_scores):.4f})")

def calculate_directional_accuracy(y_true, y_pred):
    # calculate the direction of change
    y_true_direction = np.sign(np.diff(y_true))
    y_pred_direction = np.sign(np.diff(y_pred))
    
    # compare directions and calculate accuracy
    correct_directions = np.sum(y_true_direction == y_pred_direction)
    total_predictions = len(y_true_direction)
    
    return correct_directions / total_predictions

def evaluate_ticker_model(model, X, y, ticker):
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions)
    da = calculate_directional_accuracy(y.values, predictions)

    logger.info(f"\nModel Performance for {ticker}:")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"MAPE: {mape:.4f}")
    logger.info(f"Directional Accuracy: {da:.4f}")

def evaluate_consolidated_model(model, X, y):
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions)
    da = calculate_directional_accuracy(y.values, predictions)

    logger.info("\nConsolidated Model Performance:")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"MAPE: {mape:.4f}")
    logger.info(f"Directional Accuracy: {da:.4f}")

if __name__ == "__main__":
    run_model()
