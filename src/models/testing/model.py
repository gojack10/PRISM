import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def init_model():
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.01,
        "max_depth": 6,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    return params

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = init_model()
    watchlist = [(dtrain, "train"), (dtest, "test")]

    model = xgb.train(params, dtrain, num_boost_round=800, evals=watchlist, early_stopping_rounds=50)

    plot_feature_importance(model, X)
    evaluate_model(model, X_test, y_test)

def plot_feature_importance(model, X):
    importance = model.get_score(importance_type='weight')
    importance_df = pd.DataFrame({'feature': list(importance.keys()), 'importance': list(importance.values())})
    importance_df = importance_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['feature'], importance_df['importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

    print("\nFeature Importance:")
    print(importance_df)

def evaluate_model(model, X_test, y_test):
    dtest = xgb.DMatrix(X_test, label=y_test)
    y_pred = model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nTest RMSE: {rmse}")
