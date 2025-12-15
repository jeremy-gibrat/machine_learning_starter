"""
Fonctions utilitaires pour les chaînes de caractères
"""

def is_empty(s):
    """
    Vérifie si une chaîne est vide ou None.
    """
    return not s or s.strip() == ""

def to_lower(s):
    """
    Met en minuscules.
    """
    return s.lower() if isinstance(s, str) else s

def to_upper(s):
    """
    Met en majuscules.
    """
    return s.upper() if isinstance(s, str) else s

def capitalize_first(s):
    """
    Met la première lettre en majuscule.
    """
    return s.capitalize() if isinstance(s, str) else s

def split_string(s, sep=None, maxsplit=-1):
    """
    Découpe une chaîne selon un séparateur.
    Args:
        s: chaîne à découper
        sep: séparateur (None = espaces)
        maxsplit: nombre max de splits (-1 = illimité)
    Returns:
        list: liste des sous-chaînes
    """
    if not isinstance(s, str):
        return []
    return s.split(sep, maxsplit)

def split_at_index(s, index):
    """
    Coupe une chaîne en deux à l'index donné.
    Args:
        s: chaîne à couper
        index: position de coupe
    Returns:
        tuple: (avant, après)
    """
    if not isinstance(s, str) or not isinstance(index, int):
        return s, ''
    return s[:index], s[index:]

def remove_whitespace(s):
    """
    Retire tous les espaces et caractères blancs d'une chaîne.
    """
    if not isinstance(s, str):
        return s
    return ''.join(s.split())

def replace_char(s, old, new):
    """
    Remplace un caractère par un autre dans une chaîne.
    """
    if not isinstance(s, str):
        return s
    return s.replace(old, new)