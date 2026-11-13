# src/features.py
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

def _clip(X):
    """Limite les valeurs standardisées à [-3, 3] (atténue les outliers)."""
    return X.clip(-3, 3)

def build_numeric_preprocess():
    """Prétraitement : imputation médiane + standardisation + clipping."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ("clip", FunctionTransformer(_clip)),
    ])