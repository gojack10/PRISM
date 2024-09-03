from data_loader import load_data
from model import train_model
from eval import evaluate_model
from sklearn.model_selection import train_test_split

def run_model():
    # Load data
    X, y = load_data()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train initial model
    initial_model, _, _ = train_model(X_train, y_train)

    # Evaluate model and get best parameters
    best_params = evaluate_model(initial_model, X_train, y_train, X_test, y_test)

    # Retrain the model with the best parameters
    best_model, _, _ = train_model(X, y, params=best_params)

    return best_model

if __name__ == "__main__":
    best_model = run_model()
