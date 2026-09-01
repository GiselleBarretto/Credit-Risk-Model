import pandas as pd
import numpy as np
import joblib

model = joblib.load("data/credit_risk_model.pkl")
feature_names = joblib.load("data/feature_names.pkl")

def assign_risk_bucket(prob: float) -> str:
    if prob < 0.25:
        return "Low Risk"
    elif prob < 0.55:
        return "Medium Risk"
    else:
        return "High Risk"

def predict_batch(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    df_encoded = pd.get_dummies(df.drop("default", axis=1, errors="ignore"), columns=["loan_purpose"], drop_first=True)

    # Align columns
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_names]

    probs = model.predict_proba(df_encoded)[:, 1]
    df["default_probability"] = probs.round(4)
    df["risk_bucket"] = [assign_risk_bucket(p) for p in probs]
    df["recommended_action"] = df["risk_bucket"].map({
        "Low Risk": "Approve",
        "Medium Risk": "Manual Review",
        "High Risk": "Decline",
    })

    df.to_csv(output_csv, index=False)
    print(f"\nPredictions saved to {output_csv}")
    print(df["risk_bucket"].value_counts().to_string())
    print(f"\nEstimated delinquency rate: {(probs >= 0.5).mean():.2%}")

if __name__ == "__main__":
    predict_batch("data/loan_data.csv", "data/predictions.csv")
