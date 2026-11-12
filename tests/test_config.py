# tests/test_config.py
import yaml
from src.data import load_dataset
def load_cfg():
    with open("config/train.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
def test_config_contient_les_sections():
    cfg = load_cfg()
    for section in ["data", "split", "model", "artifacts_dir"]:
        assert section in cfg, f"Section manquante : {section}"
def test_test_size_valide():
    assert 0 < float(load_cfg()["split"]["test_size"]) < 1
def test_dataset_non_vide():
    X, y = load_dataset(load_cfg())
    assert len(X) == len(y) > 0
