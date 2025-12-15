"""
Fonctions utilitaires pour la gestion de fichiers CSV
"""
import csv
import pandas as pd

def read_csv(filepath, delimiter=",", **kwargs):
    """
    Lit un fichier CSV et retourne un DataFrame pandas
    """
    return pd.read_csv(filepath, delimiter=delimiter, **kwargs)

def write_csv(data, filepath, delimiter=",", index=False, **kwargs):
    """
    Écrit un DataFrame pandas ou une liste de dicts dans un fichier CSV
    """
    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, sep=delimiter, index=index, **kwargs)
    elif isinstance(data, list) and all(isinstance(row, dict) for row in data):
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys(), delimiter=delimiter)
            writer.writeheader()
            writer.writerows(data)
    else:
        raise ValueError("data doit être un DataFrame pandas ou une liste de dictionnaires")

def add_column(df, col_name, values):
    """
    Ajoute une colonne à un DataFrame
    """
    df[col_name] = values
    return df

def remove_column(df, col_name):
    """
    Supprime une colonne d'un DataFrame
    """
    return df.drop(columns=[col_name])

def filter_csv(df, condition):
    """
    Filtre un DataFrame selon une condition (lambda ou booléen pandas)
    """
    return df[condition(df)] if callable(condition) else df[condition]

def merge_csv(df1, df2, on=None, how="inner", suffixes=("_x", "_y")):
    """
    Fusionne deux DataFrames (issus de CSV) selon une ou plusieurs colonnes
    Args:
        df1, df2: DataFrames à fusionner
        on: colonne(s) clé(s) pour la jointure
        how: type de jointure ('inner', 'outer', 'left', 'right')
        suffixes: suffixes pour colonnes dupliquées
    Returns:
        DataFrame fusionné
    """
    return pd.merge(df1, df2, on=on, how=how, suffixes=suffixes)