import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import os
import numpy as np

# Get the path to the database file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
db_path = os.path.join(project_root, 'data', 'processed', 'market_data.db')

# Create a SQLAlchemy engine
engine = create_engine(f'sqlite:///{db_path}')

# Read data from the database
query = """
SELECT * FROM intraday_aapl
JOIN indicator_sma_aapl ON intraday_aapl.date = indicator_sma_aapl.date
JOIN indicator_macd_aapl ON intraday_aapl.date = indicator_macd_aapl.date
JOIN indicator_rsi_aapl ON intraday_aapl.date = indicator_rsi_aapl.date
JOIN indicator_bbands_aapl ON intraday_aapl.date = indicator_bbands_aapl.date
JOIN indicator_obv_aapl ON intraday_aapl.date = indicator_obv_aapl.date
JOIN indicator_cci_aapl ON intraday_aapl.date = indicator_cci_aapl.date
"""

data = pd.read_sql(query, engine)

# Remove duplicate columns
data = data.loc[:,~data.columns.duplicated()]

# Print column names
print("Columns in the DataFrame:")
print(data.columns)

# Define features (X) and target variable (y)
y = data['close']
X = data.drop(columns=['ticker', 'close'])

# Convert date to numerical feature
X['date'] = pd.to_datetime(X['date']).astype(int) / 10**9

# Handle non-numeric columns
X = X.select_dtypes(include=[np.number])

print("\nFeatures used for prediction:")
print(X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to DMatrix Format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set up XGBoost parameters
params = {
    "objective": "reg:squarederror", # For regression tasks
    "eval_metric": "rmse", # Root Mean Squared Error
    "learning_rate": 0.01, # Step size shrinkage used to prevent overfitting
    "max_depth": 6, # Maximum depth of a tree
    "n_estimators": 100, # Number of trees
    "subsample": 0.8, # Subsample ratio
    "colsample_bytree": 0.8, # Column subsample ratio
    "seed": 42,  # Random seed for reproducibility
}

# Define watchlist to monitor training and testing error
watchlist = [(dtrain, "train"), (dtest, "test")]

# Train the model with 800 iterations
model = xgb.train(params, dtrain, num_boost_round=800, evals=watchlist, early_stopping_rounds=50)

# Get feature importance
importance = model.get_score(importance_type='weight')

# Convert the importance dict to a DataFrame
importance_df = pd.DataFrame({'feature': list(importance.keys()), 'importance': list(importance.values())})
importance_df = importance_df.sort_values('importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(importance_df['feature'], importance_df['importance'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Print feature importance
print("\nFeature Importance:")
print(importance_df)

# Make predictions on test set
y_pred = model.predict(dtest)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nTest RMSE: {rmse}")

