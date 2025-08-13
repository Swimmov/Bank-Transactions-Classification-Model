# deploy_model
import pandas as pd
import joblib

from preprocessing import preprocess
from utils import flatten_array
from custom_models import XGBWithLE

# Load pipeline
best_pipeline = joblib.load("best_model_pipeline.pkl")

def predict_transactions(input_df):
    processed_df = preprocess(input_df)
    predictions = best_pipeline.predict(processed_df)
    prediction_probs = None
    try:
        prediction_probs = best_pipeline.predict_proba(processed_df)
    except AttributeError:
        pass
    return predictions, prediction_probs

if __name__ == "__main__":
    new_data = pd.read_csv("Transactions_2024.csv")
    preds, probs = predict_transactions(new_data)
    
    new_data["Predicted Label"] = preds
    if probs is not None:
        new_data["Prediction Confidence"] = probs.max(axis=1)
    
    new_data.to_csv("Transactions_2024_with_predictions.csv", index=False)
    print("✅ Predictions saved to Transactions_2024_with_predictions.csv")
