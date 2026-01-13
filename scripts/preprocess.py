import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame):
   
    X = df.drop(columns=["classification", "text", "tweet_type"])
    y = df["classification"]
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, scaler