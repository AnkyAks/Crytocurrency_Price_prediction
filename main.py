from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import build_model, train_model, evaluate_model

if __name__ == "__main__":
    df = load_data('data/crypto_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = build_model(input_shape=X_train.shape[1])
    train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)
