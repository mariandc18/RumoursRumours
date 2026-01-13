import subprocess
import sys


models = ["random_forest", "decision_tree", "logistic_regression", "xgboost"]

PYTHON = sys.executable 

for model_name in models:
    print(f"\n=== Ejecutando pipeline para {model_name} ===\n")
    
    print(f"[{model_name}] Cross-validation...")
    subprocess.run([PYTHON, "scripts/crossvalidation.py", "--model", model_name], check=True)
    print(f"[{model_name}] Entrenando modelo...")
    subprocess.run([PYTHON, "scripts/train.py", "--model", model_name], check=True)
    print(f"[{model_name}] Evaluando modelo...")
    subprocess.run( [PYTHON, "scripts/eval.py", "--model", model_name], check=True)
