# mlops-ml-project — baseline ML reproductible
## Installation
python -m venv .venv
.venv\Scripts\Activate.ps1 # Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
## Entraînement
python scripts/train.py
## Évaluation
python scripts/evaluate.py
## Tests
python -m pytest -q
## Artefacts (générés dans artifacts/, non versionnés)
- model.joblib : pipeline entraîné (prétraitement + modèle)
- metrics.json : accuracy et f1_macro sur le jeu de test
- confusion_matrix.png : matrice de confusion
- report.json : rapport de classification détaillé
