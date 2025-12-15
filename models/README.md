# Modèles

Ce dossier contient les modèles ML entraînés.

## Format

Utiliser **joblib** pour sauvegarder les modèles scikit-learn :

```python
import joblib

# Sauvegarder
joblib.dump(model, 'models/mon_modele_v1.joblib')

# Charger
model = joblib.load('models/mon_modele_v1.joblib')
```

## Nomenclature

```
model_type_version_date.joblib
```

Exemple : `random_forest_v1_20251112.joblib`

## Métadonnées

Créer un fichier JSON avec les métadonnées du modèle :

```json
{
  "model_name": "Random Forest Classifier",
  "version": "1.0",
  "date": "2025-11-12",
  "accuracy": 0.95,
  "features": ["feature_1", "feature_2"],
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10
  }
}
```

## ⚠️ Important

Les modèles ne sont **PAS versionnés dans Git** car ils sont trop volumineux.

Utilisez un système de versioning de modèles (MLflow, DVC) pour la production.
