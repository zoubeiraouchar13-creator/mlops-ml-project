# src/model.py
from sklearn.linear_model import LogisticRegression

def build_model(cfg: dict):
    name = cfg["model"].get("name", "logistic_regression")
    if name != "logistic_regression":
        raise ValueError(f"Modèle non supporté dans la baseline : {name}")
    return LogisticRegression(
        max_iter=int(cfg["model"].get("max_iter", 2000)),
        C=float(cfg["model"].get("C", 1.0)), # inverse de la régularisation
    )