# Bank Transactions Classification Model

**Model Version**: 1.0
**Author**: Dazmitry Kandrykinski
**Date**: 2025-05-15


## Overview

This model classifies bank transactions into labeled categories (e.g., "Bill Payment", "Deposit: income", etc.) based on transaction description, type (credit/debit), amount, and date-derived features.
It is implemented as a full **scikit-learn preprocessing + classifier pipeline** with built-in text processing (`TfidfVectorizer`), numeric scaling, and categorical encoding.
The best-performing model is **XGBoost** with custom label encoding (`XGBWithLE`) to handle string class labels.

---

## Files Included

| File Name                                | Description                                           |
| ---------------------------------------- | ----------------------------------------------------- |
| `best_model_pipeline.pkl`                | Trained preprocessing + model pipeline                |
| `Transactions_2024.csv`                  | Example input dataset (unlabeled)                     |
| `Transactions_2024_with_predictions.csv` | Example output with predictions                       |
| `preprocessing.py`                       | Preprocessing functions for new input data            |
| `custom_models.py`                       | Custom `XGBWithLE` wrapper for XGBoost label encoding |
| `utils.py`                               | Helper functions (e.g., `flatten_array`)              |
| `deploy_model.py`                        | Deployment script to load model and predict           |
| `requirements.txt`                       | Python dependencies                                   |
| `README.md`                              | Project documentation                                 |

---

## Model Summary

* **Model Type**: Multi-class Classification
* **Technique**: XGBoost (wrapped in `XGBWithLE`)
* **Preprocessing steps**:

  * Convert `Processed Date` to datetime
  * Extract `day_of_week_numeric` and `month_numeric`
  * Drop unused columns (`Processed Date`, `Account Name`, `Check Number`)
  * Impute missing text, numeric, and categorical values
  * Apply `TfidfVectorizer` to text
  * Scale numeric features
  * One-hot encode categorical features
* **Target Variable**: `Label`

---

## Model Performance

| Metric              | Value                       |
| ------------------- | --------------------------- |
| Accuracy            | 0.764                       |
| F1-score (weighted) | 0.763                       |
| Best Model          | XGBoost with label encoding |

---

## Input Data Format

| Column Name       | Type        | Description                                           |
| ----------------- | ----------- | ----------------------------------------------------- |
| `Description`     | string      | Transaction text description                          |
| `Credit or Debit` | string      | Transaction type (`Credit`/`Debit`)                   |
| `Amount`          | float       | Transaction amount                                    |
| `Processed Date`  | string/date | Date transaction was processed                        |
| `Account Name`    | string      | Account name (dropped in preprocessing)               |
| `Check Number`    | string/int  | Check number if applicable (dropped in preprocessing) |

**Note**: Do not include the target variable column (`Label`) for predictions.

**Note**: Do not manually preprocess or encode data — the pipeline handles it automatically.

---

## Model Usage Example

```python
import pandas as pd
from preprocessing_predict import preprocess, predict_transactions

# Load new transaction data
new_data = pd.read_csv("Transactions_2024.csv")

# Get predictions and probabilities
preds, probs = predict_transactions(new_data)

# Add predictions to DataFrame
new_data["Predicted Label"] = preds
if probs is not None:
    new_data["Prediction Confidence"] = probs.max(axis=1)

# Save results
new_data.to_csv("Transactions_2024_with_predictions.csv", index=False)
```

---

## Contact & Support

If you have any questions or want to integrate this into an application, feel free to contact:
📧 [d.kandrykinski@gmail.com](mailto:d.kandrykinski@gmail.com)
💼 [GitHub](https://github.com/Swimmov)

---

