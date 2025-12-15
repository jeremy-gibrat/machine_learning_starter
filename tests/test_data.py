"""
Tests pour le module data
"""

import pytest
import pandas as pd
from src.data.preprocess import clean_data, handle_missing_values


def test_clean_data():
    """Test du nettoyage des données"""
    # Créer des données de test avec duplicatas
    df = pd.DataFrame({
        'col1': [1, 2, 2, 3],
        'col2': [4, 5, 5, 6]
    })
    
    result = clean_data(df)
    
    # Vérifier que les duplicatas sont supprimés
    assert len(result) == 3


def test_handle_missing_values():
    """Test de la gestion des valeurs manquantes"""
    df = pd.DataFrame({
        'col1': [1, 2, None, 4],
        'col2': [5, None, 7, 8]
    })
    
    result = handle_missing_values(df, strategy='drop')
    
    # Vérifier qu'il ne reste pas de NaN
    assert result.isna().sum().sum() == 0
