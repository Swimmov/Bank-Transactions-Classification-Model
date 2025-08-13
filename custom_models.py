# custom_models.py
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

class XGBWithLE(BaseEstimator, ClassifierMixin):
    """XGBoost wrapper with LabelEncoder for string labels."""
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)
        self.le = LabelEncoder()
    
    def fit(self, X, y):
        y_enc = self.le.fit_transform(y)
        self.model.fit(X, y_enc)
        return self
    
    def predict(self, X):
        y_pred_enc = self.model.predict(X)
        return self.le.inverse_transform(y_pred_enc)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
