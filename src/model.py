# src/model.py
from distro import name
from sklearn.linear_model import LogisticRegression

def build_model(cfg: dict):
    name = cfg["model"].get("name", "logistic_regression")
    if name != "logistic_regression":
        raise ValueError(f"Modèle non supporté dans la baseline : {name}")
    return LogisticRegression(max_iter=int(cfg["model"].get("max_iter", 2000)))