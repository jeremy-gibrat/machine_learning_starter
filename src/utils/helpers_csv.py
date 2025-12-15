"""
Fonctions utilitaires pour la gestion de fichiers CSV
"""
import csv
import pandas as pd
import numpy as np
import os

def read_csv(filepath, delimiter=",", **kwargs):
    """
    Lit un fichier CSV et retourne un DataFrame pandas
    Exemple :
        df = read_csv('data.csv')
    """
    try:
        return pd.read_csv(filepath, delimiter=delimiter, **kwargs)
    except FileNotFoundError:
        print(f"[ERROR] Fichier non trouvé : {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Erreur lecture CSV : {e}")
        return pd.DataFrame()

def write_csv(data, filepath, delimiter=",", index=False, **kwargs):
    """
    Écrit un DataFrame pandas ou une liste de dicts dans un fichier CSV
    Exemple :
        write_csv(df, 'out.csv')
    """
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, sep=delimiter, index=index, **kwargs)
        elif isinstance(data, list) and all(isinstance(row, dict) for row in data):
            with open(filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys(), delimiter=delimiter)
                writer.writeheader()
                writer.writerows(data)
        else:
            raise ValueError("data doit être un DataFrame pandas ou une liste de dictionnaires")
    except Exception as e:
        print(f"[ERROR] Erreur écriture CSV : {e}")

def concat_csv(dfs, axis=0, ignore_index=True):
    """
    Concatène plusieurs DataFrames (issus de CSV).
    Args:
        dfs: liste de DataFrames à concaténer
        axis: 0 pour concaténer les lignes, 1 pour les colonnes
        ignore_index: réindexe le résultat si True
    Returns:
        DataFrame concaténé
    """
    return pd.concat(dfs, axis=axis, ignore_index=ignore_index)

def concat_csv_from_folder(folder_path, delimiter=",", axis=0, ignore_index=True, **kwargs):
    """
    Concatène tous les fichiers CSV d'un dossier en un seul DataFrame.
    Args:
        folder_path: chemin du dossier contenant les CSV
        delimiter: séparateur CSV
        axis: 0 pour concaténer les lignes, 1 pour les colonnes
        ignore_index: réindexe le résultat si True
        kwargs: arguments supplémentaires pour read_csv
    Returns:
        DataFrame concaténé
    """
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    dfs = [read_csv(f, delimiter=delimiter, **kwargs) for f in csv_files]
    return concat_csv(dfs, axis=axis, ignore_index=ignore_index)

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

def rename_columns(df, columns_map):
    """
    Renomme les colonnes d'un DataFrame selon un mapping.
    Args:
        df: DataFrame source
        columns_map: dict {ancien_nom: nouveau_nom}
    Returns:
        DataFrame avec colonnes renommées
    """
    return df.rename(columns=columns_map)

def select_columns(df, columns):
    """
    Retourne un DataFrame ne contenant que les colonnes spécifiées.
    Si une colonne n'existe pas, elle est créée avec des valeurs NaN.
    Args:
        df: DataFrame source
        columns: liste des noms de colonnes à conserver
    Returns:
        DataFrame filtré
    """
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[columns]

