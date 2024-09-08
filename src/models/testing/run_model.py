from data_loader import load_data
from model import train_model, plot_feature_importance
from eval import evaluate_model, preprocess_data
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np
import pandas as pd
from tqdm import tqdm

def run_model():
    # Load data
    data_tuple = load_data()
    
    # Check if data is a tuple and extract the DataFrame
    if isinstance(data_tuple, tuple):
        data = data_tuple[0]  # Assuming the DataFrame is the first element of the tuple
    else:
        data = data_tuple
    
    # Sort data by date
    data = data.sort_values(['ticker', 'date'])
    
    # Separate features and target
    X = data.drop(['close'], axis=1)
    y = data['close']
    
    print("Columns in X before preprocessing:")
    print(X.columns)
    
    # Preprocess data
    X_preprocessed = preprocess_data(X)
    
    print("Columns in X_preprocessed after preprocessing:")
    print(X_preprocessed.columns)
    
    # Ensure y is aligned with X_preprocessed
    y = y.loc[X_preprocessed.index]
    
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize lists to store results
    train_scores = []
    test_scores = []
    feature_importances = []
    
    # Perform cross-validation
    for train_index, test_index in tqdm(tscv.split(X_preprocessed), desc="Cross-validation", total=5):
        X_train, X_test = X_preprocessed.iloc[train_index], X_preprocessed.iloc[test_index]
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
    print(f"Average Train RMSE: {np.mean(train_scores):.4f} (+/- {np.std(train_scores):.4f})")
    print(f"Average Test RMSE: {np.mean(test_scores):.4f} (+/- {np.std(test_scores):.4f})")
    
    # Plot average feature importance
    avg_feature_importance = pd.concat(feature_importances, axis=1).mean(axis=1).sort_values(ascending=False)
    plot_feature_importance(avg_feature_importance.values, avg_feature_importance.index)
    
    # Train final model on all data
    final_model_tuple = train_model(X_preprocessed, y)
    final_model = final_model_tuple[0]  # Assuming the model is the first element of the tuple
    
    # Evaluate final model
    final_predictions = final_model.predict(X_preprocessed)
    final_rmse = np.sqrt(mean_squared_error(y, final_predictions))
    final_mae = mean_absolute_error(y, final_predictions)
    final_r2 = r2_score(y, final_predictions)
    final_mape = mean_absolute_percentage_error(y, final_predictions)
    
    print("\nFinal Model Performance:")
    print(f"RMSE: {final_rmse:.4f}")
    print(f"MAE: {final_mae:.4f}")
    print(f"R2 Score: {final_r2:.4f}")
    print(f"MAPE: {final_mape:.4f}")
    
    # Plot final feature importance
    plot_feature_importance(final_model.feature_importances_, X_preprocessed.columns)

if __name__ == "__main__":
    run_model()
