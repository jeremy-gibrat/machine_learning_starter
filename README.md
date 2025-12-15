# 🤖 Template ML - Projet Machine Learning Standardisé

## 📋 Description

Ce template fournit une structure standardisée pour les projets Machine Learning à livrer à l'équipe FlightWatching. Il garantit la cohérence, la reproductibilité et la maintenabilité des modèles ML.

---

## 📁 Structure du Projet

```
template_ml/
├── README.md                  # Ce fichier - Documentation principale
├── requirements.txt           # Dépendances Python du projet
├── setup_venv.sh             # Script de création de l'environnement virtuel
├── activate_venv.sh          # Script d'activation rapide
├── .env.example              # Template des variables d'environnement
├── .gitignore                # Fichiers à exclure de Git
│
├── config/                   # Configuration du projet
│   ├── config.py            # Paramètres globaux
│   └── features.py          # Définition des features ML
│
├── data/                     # Données (NON versionnées dans Git)
│   ├── raw/                 # Données brutes originales
│   ├── processed/           # Données nettoyées et transformées
│   └── README.md            # Description des datasets
│
├── models/                   # Modèles entraînés (NON versionnés dans Git)
│   ├── README.md            # Informations sur les modèles
│   └── .gitkeep             # Garde le dossier dans Git
│
├── notebooks/                # Notebooks Jupyter pour l'exploration
│   ├── 01_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
│
├── src/                      # Code source du projet
│   ├── __init__.py
│   ├── data/                # Scripts de traitement des données
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── features/            # Engineering des features
│   │   ├── __init__.py
│   │   └── build_features.py
│   │
│   ├── models/              # Entraînement et prédiction
│   │   ├── __init__.py
│   │   ├── train.py         # Inclut la classe PINN, train_pinn, save/load_pinn_model
│   │   └── predict.py       # Inclut predict_pinn et predict_pinn_from_file
│   │
│   └── utils/               # Fonctions utilitaires
│       ├── __init__.py
│       ├── helpers.py
│       ├── helpers_math.py      # Fonctions mathématiques (safe_divide, arrondi, clamp...)
│       ├── helpers_date.py      # Fonctions sur les dates (format, to_utc, etc)
│       ├── helpers_string.py    # Fonctions sur les chaînes (split, etc)
│       ├── helpers_object.py    # Fonctions sur les dicts
│       └── helpers_array.py     # Fonctions sur les listes
│
├── tests/                    # Tests unitaires
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
└── docs/                     # Documentation supplémentaire
    ├── ARCHITECTURE.md      # Architecture du modèle
    ├── API.md               # Documentation API
    └── DEPLOYMENT.md        # Guide de déploiement
```

---

## 🚀 Quick Start

### 1. Cloner le template

```bash
# Copier le template pour un nouveau projet
cp -r template_ml mon_projet_ml
cd mon_projet_ml
```

### 2. Créer l'environnement virtuel

```bash
# Exécuter le script de setup (crée le venv et installe les dépendances)
bash setup_venv.sh
```

### 3. Activer l'environnement

```bash
# Activer l'environnement virtuel
source activate_venv.sh
```

### 4. Configurer les variables d'environnement

```bash
# Copier le template
cp .env.example .env

# Éditer avec vos valeurs
nano .env
```

### 5. Vérifier l'installation

```bash
# Tester que tout fonctionne
python -c "import pandas, numpy, sklearn; print('✅ Environnement OK')"
```

---

## 📦 Dépendances Standard

Les dépendances suivantes sont incluses dans `requirements.txt` :

- **Data Science** : pandas, numpy
- **Machine Learning** : scikit-learn, joblib
- **Visualisation** : matplotlib, seaborn
- **Utilitaires** : python-dotenv, tqdm
- **Tests** : pytest

### Ajouter des dépendances

```bash
# Installer une nouvelle dépendance
pip install nouvelle-dependance

# Mettre à jour requirements.txt
pip freeze > requirements.txt
```

---

## 📝 Conventions de Code

### Nommage des fichiers

- Scripts Python : `snake_case.py` (ex: `train_model.py`)
- Notebooks : `NN_description.ipynb` (ex: `01_exploration.ipynb`)
- Modèles sauvegardés : `nom_modele_version.joblib` (ex: `rf_classifier_v1.joblib`)

### Structure du code

```python
"""
Description du module
"""

# Imports standards
import os
import sys

# Imports third-party
import pandas as pd
import numpy as np

# Imports locaux
from config import config
from src.utils import helpers

# Constantes
CONSTANT_NAME = "valeur"

# Fonctions
def ma_fonction(param):
    """
    Docstring claire avec Args et Returns
    
    Args:
        param: Description du paramètre
        
    Returns:
        Description du retour
    """
    pass
```

### Gestion des erreurs

```python
# Toujours gérer les exceptions
try:
    result = operation_risquee()
except SpecificException as e:
    logger.error(f"Erreur: {e}")
    return None
```

---

## 🧪 Tests

### Exécuter les tests

```bash
# Tous les tests
pytest tests/

# Un fichier spécifique
pytest tests/test_models.py

# Avec couverture
pytest --cov=src tests/
```

### Écrire des tests

```python
# tests/test_models.py
import pytest
from src.models.train import train_model

def test_train_model():
    """Test de l'entraînement du modèle"""
    model = train_model(data_train)
    assert model is not None
    assert hasattr(model, 'predict')
```

---

## 📊 Workflow Standard

### 1. Exploration des données (notebooks/)

```python
# 01_exploration.ipynb
import pandas as pd
data = pd.read_csv('../data/raw/dataset.csv')
data.describe()
```

### 2. Préparation des données (src/data/)

```python
# src/data/preprocess.py
def clean_data(df):
    """Nettoie les données brutes"""
    df = df.dropna()
    df = df[df['value'] > 0]
    return df
```

### 3. Engineering des features (src/features/)

```python
# src/features/build_features.py
def compute_features(df):
    """Calcule les features pour le ML"""
    df['feature_1'] = df['col_a'] / df['col_b']
    return df
```

### 4. Entraînement (src/models/train.py)

```python
# src/models/train.py
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train):
    """Entraîne le modèle"""
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/model.joblib')
    return model
```

### 5. Prédiction (src/models/predict.py)

```python
# src/models/predict.py
import joblib

def predict(X_new):
    """Fait une prédiction"""
    model = joblib.load('models/model.joblib')
    return model.predict(X_new)
```

---

## 🔒 Sécurité et Confidentialité

### Variables d'environnement

**JAMAIS** committer de secrets dans Git :
- API Keys
- Mots de passe
- URLs sensibles

Utiliser `.env` et `python-dotenv` :

```python
# .env
API_KEY=votre_cle_secrete
DATABASE_URL=postgres://...

# Dans le code
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
```

### Données sensibles

Les dossiers `data/` et `models/` sont exclus de Git par défaut.

---

## 📤 Livraison à l'équipe FlightWatching

### Checklist avant livraison

- [ ] README.md complété avec description du projet
- [ ] requirements.txt à jour
- [ ] .env.example fourni (sans secrets)
- [ ] Code documenté (docstrings)
- [ ] Tests passent (`pytest`)
- [ ] Notebook d'exemple fourni
- [ ] Documentation API/déploiement dans `docs/`
- [ ] Modèle sauvegardé dans `models/` avec métadonnées
- [ ] Données d'exemple dans `data/raw/` (si possible)

### Format de livraison

```bash
# Créer une archive du projet (sans venv, data volumineuses, etc.)
zip -r mon_projet_ml.zip . -x "venv/*" "*.pyc" "__pycache__/*" ".git/*" "data/raw/*"
```

### Documentation requise

1. **README.md** : Description, installation, utilisation
2. **docs/ARCHITECTURE.md** : Architecture du modèle ML
3. **docs/API.md** : Interface de prédiction
4. **docs/DEPLOYMENT.md** : Instructions de déploiement

---

## 📜 Licence

Propriété de FlightWatching - Confidentiel

---

## 🔄 Versions

- **v1.0.0** (2025-11-12) : Version initiale du template

---

## 🎯 Objectifs du Template

1. ✅ **Standardisation** : Structure cohérente pour tous les projets ML
2. ✅ **Reproductibilité** : Environnements virtuels isolés
3. ✅ **Maintenabilité** : Code propre et documenté
4. ✅ **Collaboration** : Facilite le passage de relais entre équipes
5. ✅ **Déploiement** : Prêt pour production FlightWatching

---

**Bonne chance avec votre projet ML ! 🚀**
