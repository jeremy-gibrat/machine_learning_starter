"""
Module de chargement des données
"""

import pandas as pd
from pathlib import Path
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_raw_data(filename):
    """
    Charge les données brutes depuis data/raw/
    
    Args:
        filename (str): Nom du fichier à charger
        
    Returns:
        pd.DataFrame: Données chargées
    """
    filepath = RAW_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    
    # Détecter le format et charger
    if filepath.suffix == '.csv':
        return pd.read_csv(filepath)
    elif filepath.suffix == '.parquet':
        return pd.read_parquet(filepath)
    elif filepath.suffix == '.json':
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Format non supporté: {filepath.suffix}")


def save_processed_data(df, filename):
    """
    Sauvegarde les données traitées dans data/processed/
    
    Args:
        df (pd.DataFrame): Données à sauvegarder
        filename (str): Nom du fichier
    """
    filepath = PROCESSED_DATA_DIR / filename
    
    if filepath.suffix == '.csv':
        df.to_csv(filepath, index=False)
    elif filepath.suffix == '.parquet':
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Format non supporté: {filepath.suffix}")
    
    print(f"✅ Données sauvegardées: {filepath}")


def load_processed_data(filename):
    """
    Charge les données traitées depuis data/processed/
    
    Args:
        filename (str): Nom du fichier
        
    Returns:
        pd.DataFrame: Données chargées
    """
    filepath = PROCESSED_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    
    if filepath.suffix == '.csv':
        return pd.read_csv(filepath)
    elif filepath.suffix == '.parquet':
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Format non supporté: {filepath.suffix}")
