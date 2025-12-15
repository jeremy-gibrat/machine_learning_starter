"""
Module de construction des features
"""

import pandas as pd
from config.features import compute_derived_features, validate_features


def build_features(df):
    """
    Construit toutes les features nécessaires pour le modèle
    
    Args:
        df (pd.DataFrame): Données prétraitées
        
    Returns:
        pd.DataFrame: Données avec features calculées
    """
    # Calculer les features dérivées
    df = compute_derived_features(df)
    
    # Valider que toutes les features sont présentes
    validate_features(df)
    
    return df


def select_features(df, feature_list):
    """
    Sélectionne uniquement les features spécifiées
    
    Args:
        df (pd.DataFrame): Données complètes
        feature_list (list): Liste des features à conserver
        
    Returns:
        pd.DataFrame: Données avec features sélectionnées
    """
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        raise ValueError(f"Features manquantes: {missing}")
    
    return df[feature_list]
