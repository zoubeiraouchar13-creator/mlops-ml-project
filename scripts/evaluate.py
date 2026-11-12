# scripts/evaluate.py
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import yaml
from sklearn.metrics import classification_report
from src.data import load_dataset

def load_cfg(path="config/train.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    art_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    model = joblib.load(art_dir / "model.joblib")
    X, y = load_dataset(cfg)
    pred = model.predict(X)
    report = classification_report(y, pred, output_dict=True)
    (art_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("Evaluate OK: artifacts/report.json")

if __name__ == "__main__":
 main()
