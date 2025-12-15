"""
Module de prétraitement des données
"""

import pandas as pd
import numpy as np


def clean_data(df):
    """
    Nettoie les données brutes
    
    Args:
        df (pd.DataFrame): Données brutes
        
    Returns:
        pd.DataFrame: Données nettoyées
    """
    # Supprimer les duplicatas
    df = df.drop_duplicates()
    
    # Supprimer les lignes avec trop de NaN
    df = df.dropna(thresh=len(df.columns) * 0.5)
    
    return df


def handle_missing_values(df, strategy='mean'):
    """
    Gère les valeurs manquantes
    
    Args:
        df (pd.DataFrame): Données avec valeurs manquantes
        strategy (str): Stratégie ('mean', 'median', 'drop')
        
    Returns:
        pd.DataFrame: Données sans valeurs manquantes
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    else:
        raise ValueError(f"Stratégie inconnue: {strategy}")


def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Supprime les outliers d'une colonne
    
    Args:
        df (pd.DataFrame): Données
        column (str): Nom de la colonne
        method (str): Méthode ('iqr' ou 'zscore')
        threshold (float): Seuil pour la détection
        
    Returns:
        pd.DataFrame: Données sans outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        return df[(df[column] >= lower) & (df[column] <= upper)]
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return df[z_scores < threshold]
    
    else:
        raise ValueError(f"Méthode inconnue: {method}")


def normalize_column(df, column, method='minmax'):
    """
    Normalise une colonne
    
    Args:
        df (pd.DataFrame): Données
        column (str): Nom de la colonne
        method (str): Méthode ('minmax' ou 'standard')
        
    Returns:
        pd.DataFrame: Données avec colonne normalisée
    """
    if method == 'minmax':
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    elif method == 'standard':
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    return df
