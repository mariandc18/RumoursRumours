import argparse
import joblib
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from models import obtain_model
from preprocess import preprocess_data


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    target_names = ["non-rumours", "rumours"]
    report = classification_report(y_test, y_pred, target_names=target_names)
    return report

def save_model(model, output_dir="outputs/models", model_name="model.pkl"):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, f"{output_dir}/{model_name}")


def load_best_params(model_name):
    json_path = f"outputs/metrics/{model_name}_random.json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        best_params = data.get("best_params", None)
        if best_params:
            print(f"Loaded best params from CV: {best_params}")
            return best_params
    print("No CV results found, using default parameters")
    return None


def main(model_name):
    data = pd.read_csv("data/processed/pheme_features.csv")
    X, y, scaler = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = obtain_model(model_name)
    best_params = load_best_params(model_name)
    if best_params:
        model.set_params(**best_params)

    model = train_model(model, X_train, y_train)
    print(evaluate_model(model, X_test, y_test))
    save_model(model, model_name=f"{model_name}.pkl")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="random_forest",
                        help="Modelo definido en models.py (random_forest, decision_tree, logistic_regression, xgboost)")
    args = parser.parse_args()

    main(args.model)