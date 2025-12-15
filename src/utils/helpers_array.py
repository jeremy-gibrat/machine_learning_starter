"""
Fonctions utilitaires pour les tableaux (listes)
"""

def concat_lists(*lists):
    """
    Concatène plusieurs listes.
    """
    result = []
    for l in lists:
        if isinstance(l, list):
            result += l
    return result

def is_empty_list(l):
    """
    Vérifie si une liste est vide ou None.
    """
    return not l or len(l) == 0
