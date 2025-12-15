"""
Module d'entraînement du modèle
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from config.config import MODEL_DIR, RANDOM_SEED, TEST_SIZE, MODEL_PARAMS


def train_model(X, y, model_params=None):
    """
    Entraîne un modèle Random Forest
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        model_params (dict): Hyperparamètres du modèle
        
    Returns:
        model: Modèle entraîné
        metrics: Métriques d'évaluation
    """
    # Utiliser les paramètres par défaut si non fournis
    if model_params is None:
        model_params = MODEL_PARAMS
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    print(f"📊 Train set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    
    # Entraînement
    print("\n🚀 Entraînement du modèle...")
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    print("✅ Modèle entraîné")
    
    # Évaluation
    print("\n📈 Évaluation...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    metrics = {
        'accuracy': accuracy,
        'classification_report': report
    }
    
    return model, metrics


def save_model(model, filename):
    """
    Sauvegarde le modèle entraîné
    
    Args:
        model: Modèle à sauvegarder
        filename (str): Nom du fichier
    """
    filepath = MODEL_DIR / filename
    joblib.dump(model, filepath)
    print(f"✅ Modèle sauvegardé: {filepath}")


def load_model(filename):
    """
    Charge un modèle sauvegardé
    
    Args:
        filename (str): Nom du fichier
        
    Returns:
        model: Modèle chargé
    """
    filepath = MODEL_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {filepath}")
    
    model = joblib.load(filepath)
    print(f"✅ Modèle chargé: {filepath}")
    return model
