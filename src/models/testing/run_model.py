from data_loader import load_data
from model import train_model

if __name__ == "__main__":
    X, y = load_data()
    train_model(X, y)
