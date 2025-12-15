"""
Tests pour le module models
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


def test_train_model():
    """Test de l'entraînement du modèle"""
    # Créer un dataset de test
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    y_series = pd.Series(y)
    
    from src.models.train import train_model
    
    model, metrics = train_model(X_df, y_series)
    
    # Vérifier que le modèle existe et a des métriques
    assert model is not None
    assert 'accuracy' in metrics
    assert metrics['accuracy'] > 0
