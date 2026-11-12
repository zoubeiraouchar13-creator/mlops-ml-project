# scripts/train.py
import json
import sys
from pathlib import Path
# Rendre le package src/ importable quand on lance : python scripts/train.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import matplotlib
matplotlib.use("Agg") # backend sans écran (terminal, Docker, CI)
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.data import load_dataset
from src.features import build_numeric_preprocess
from src.model import build_model

def load_cfg(path="config/train.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    cfg = load_cfg()
    art_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    art_dir.mkdir(exist_ok=True)
    X, y = load_dataset(cfg)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=float(cfg["split"]["test_size"]),
        random_state=int(cfg["split"]["random_state"]),
        stratify=y,
    )
    pipe = Pipeline(steps=[
        ("preprocess", build_numeric_preprocess()),
        ("model", build_model(cfg)),
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    f1 = float(f1_score(yte, pred, average="macro"))
 # Artefacts
    joblib.dump(pipe, art_dir / "model.joblib")
    metrics = {"accuracy": acc, "f1_macro": f1}
    (art_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_confusion_matrix(yte, pred, art_dir / "confusion_matrix.png")
    print("Train OK:", metrics)
    
if __name__ == "__main__":
 main()