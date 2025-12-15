"""
Fonctions utilitaires pour les objets (dict)
"""
import re
import json

def get_keys(d):
    """
    Retourne la liste des clés d'un dictionnaire.
    """
    return list(d.keys()) if isinstance(d, dict) else []

def get_values(d):
    """
    Retourne la liste des valeurs d'un dictionnaire.
    """
    return list(d.values()) if isinstance(d, dict) else []

def merge_dicts(a, b):
    """
    Fusionne deux dictionnaires (b écrase a en cas de conflit).
    """
    return {**a, **b}

def correct_and_parse_json(invalid_json_string):
    """
    Corrige une chaîne JSON mal formée (sans virgules entre les paires clé/valeur) et la parse.
    Remplace aussi les doubles quotes par des simples quotes sur les attributs ou valeurs.
    Retourne un dict Python ou None si parsing impossible.
    """
    def add_commas_and_fix_quotes(s):
        # Si la chaîne commence et finit par un guillemet, on l'enlève
        s = s.strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        # Remplace les doubles quotes par une seule quote
        s = s.replace('""', '"')
        # Ajoute une virgule entre } et " (rare, mais pour sécurité)
        s = re.sub(r'}\s+"', '}, "', s)
        # Ajoute une virgule entre " et " (clé/clé)
        s = re.sub(r'"\s+"', '", "', s)
        # Ajoute une virgule après un nombre (int ou float) suivi d'un espace et d'une clé
        s = re.sub(r'(\d)(\.\d+)?\s+"', r'\1\2, "', s)
        return s
    corrected_json_string = add_commas_and_fix_quotes(invalid_json_string)
    try:
        return json.loads(corrected_json_string)
    except json.JSONDecodeError as e:
        print(f"Erreur lors du parsing du JSON corrigé : {e}\n -Chaîne corrigée : {corrected_json_string} ÷ \n -Chaîne originale : {invalid_json_string}")
        return None