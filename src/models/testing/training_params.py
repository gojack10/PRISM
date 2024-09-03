import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

# Split the data into features (X) and target (y)
X = data.drop(columns=["target_price"])
y = data["target_price"]

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to DMatrix Format

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set up XGBoost parameters

params = {
    "objective": "reg:squarederror", # For regression tasks
    "eval_metric": "rmse", # Root Mean Squared Error
    "learning_rate": 0.1, # Step size shrinkage used to prevent overfitting
    "max_depth": 6, # Maximum depth of a tree
    "n_estimators": 100, # Number of trees
    "subsample": 0.8, # Subsample ratio
    "colsample_bytree": 0.8, # Column subsample ratio
    "seed": 42,  # Random seed for reproducibility
}

# Define watchlist to monitor training and testing error

watchlist = [(dtrain, "train"), (dtest, "test")]

model = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist, early_stopping_rounds=10)

