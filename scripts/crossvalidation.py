import argparse
import json
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from models import obtain_model
from preprocess import preprocess_data


PARAM_DISTS = {
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced"]
    },
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l1"],         
        "solver": ["liblinear"],    
        "class_weight": ["balanced"],
        "max_iter": [1000]
    },
    "decision_tree": {
        "max_depth": [None, 10, 20, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
    },
    "xgboost": { 
        "n_estimators": [100, 200, 300], 
        "max_depth": [3, 5, 7], 
        "learning_rate": [0.01, 0.05, 0.1], 
        "subsample": [0.8, 1.0], 
        "colsample_bytree": [0.8, 1.0] 
    }
}

def load_data(path="data/processed/pheme_features.csv"):
    data = pd.read_csv(path)
    X, y, scaler = preprocess_data(data)
    return X, y

def run_random_search(model, model_name, X, y, n_iter=30):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAM_DISTS[model_name],
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(X, y)
    return search

def save_results(search, model_name, output_dir="outputs/metrics"):
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "best_score": search.best_score_,
        "best_params": search.best_params_
    }

    path = os.path.join(output_dir, f"{model_name}_random.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    pd.DataFrame(search.cv_results_).to_csv(
        os.path.join(output_dir, f"{model_name}_cv_results.csv"), index=False
    )

def main(model_name):
    X, y = load_data()
    model = obtain_model(model_name)

    search = run_random_search(model, model_name, X, y)
    save_results(search, model_name)

    print(f"Best F1-macro ({model_name}): {search.best_score_:.3f}")
    print("Best params:", search.best_params_)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="random_forest",
                        help="Modelo definido en models.py (random_forest, logistic_regresion, decision_tree, xgboost)")
    args = parser.parse_args()

    main(args.model)
