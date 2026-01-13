from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "xgboost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
}

def obtain_model(name: str, **kwargs):
    if name not in MODELS:
        raise ValueError(f"'{name}' no esta dentro de los modelos definidos.")
    model = MODELS[name]
    if kwargs:
        model.set_params(**kwargs)
    return model