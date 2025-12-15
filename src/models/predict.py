"""
Module de prédiction
"""

import pandas as pd
from src.models.train import load_model


def predict(X_new, model_filename):
    """
    Fait une prédiction avec le modèle entraîné
    
    Args:
        X_new (pd.DataFrame): Nouvelles données
        model_filename (str): Nom du fichier modèle
        
    Returns:
        predictions: Prédictions du modèle
    """
    # Charger le modèle
    model = load_model(model_filename)
    
    # Prédire
    predictions = model.predict(X_new)
    
    return predictions


def predict_proba(X_new, model_filename):
    """
    Fait une prédiction avec probabilités
    
    Args:
        X_new (pd.DataFrame): Nouvelles données
        model_filename (str): Nom du fichier modèle
        
    Returns:
        probabilities: Probabilités des classes
    """
    # Charger le modèle
    model = load_model(model_filename)
    
    # Prédire les probabilités
    probabilities = model.predict_proba(X_new)
    
    return probabilities
