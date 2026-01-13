import argparse
import os
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data

def load_test_data(path="data/processed/pheme_features.csv", test_size=0.2, random_state=42):
    data = pd.read_csv(path)
    X, y, scaler = preprocess_data(data)
    feature_names = data.drop(columns=["classification", "text", "tweet_type"]).columns.tolist() 

    _, X_test, _, y_test = train_test_split( X, y, test_size=test_size, stratify=y, random_state=random_state
                                            )
    return X_test, y_test, feature_names


def load_model(model_path):
    return joblib.load(model_path)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    target_names = ["non-rumours", "rumours"]
    print(classification_report(y_test, y_pred,  target_names=target_names))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix")
    print(cm)
    
    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC: {auc:.3f}")
    
    return y_pred


def plot_feature_importance(model, feature_names, model_name="model", top_n=20, save=True, show=True):
    plt.figure(figsize=(10,6))
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print("Feature importance no disponible para este modelo")
        return
    
    indices = np.argsort(importances)[::-1][:top_n]
    plt.barh(range(top_n), importances[indices][::-1], align="center")
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel("Importance / Coefficient magnitude")
    plt.title(f"Top Feature Importances - {model_name}")
    plt.tight_layout()
    
    if save:
        fig_dir = "outputs/figures"
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, f"{model_name}_feature_importance.png"))
        print(f"Feature importance figure saved to {fig_dir}")
    
    if show:
        plt.show()
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, model_name="model", save=True, show=True):
    plt.figure(figsize=(6,6))
    cm_disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    
    if save:
        fig_dir = "outputs/figures"
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, f"{model_name}_confusion_matrix.png"))
        print(f"Confusion matrix saved to {fig_dir}")
    
    if show:
        plt.show()
    plt.close()


def plot_roc(model, X_test, y_test, model_name="model", save=True, show=True):
    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        plt.figure()
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"ROC Curve - {model_name}")
        
        if save:
            fig_dir = "outputs/figures"
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(os.path.join(fig_dir, f"{model_name}_roc_curve.png"))
            print(f"ROC curve saved to {fig_dir}")
        
        if show:
            plt.show()
        plt.close()


def save_metrics(y_test, y_pred, model_name="model", output_dir="outputs/metrics"):
    os.makedirs(output_dir, exist_ok=True)
    metrics = {
        "classification_report": classification_report(y_test, y_pred,  target_names= ["non-rumours", "rumours"], output_dict=True)
    }
    path = os.path.join(output_dir, f"{model_name}_test_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {path}")


def main(model_name="random_forest"):
    model_path = f"outputs/models/{model_name}.pkl"
    model = load_model(model_path)
    
    X_test, y_test, feature_names = load_test_data()
    y_pred = evaluate_model(model, X_test, y_test)
    
    save_metrics(y_test, y_pred, model_name)
    plot_feature_importance(model, feature_names, model_name)
    plot_confusion_matrix(model, X_test, y_test, model_name)
    plot_roc(model, X_test, y_test, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="random_forest",
                        help="Modelo a evaluar: random_forest, decision_tree, logistic_regression")
    args = parser.parse_args()
    
    main(args.model)
